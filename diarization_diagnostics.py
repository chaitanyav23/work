import json
import torch
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import argparse
import sys
import os

# Add project root to path
sys.path.append(str(Path.cwd()))

from scripts.utils.device import get_device

def load_ground_truth(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def get_der_stats(ref_segments, pred_segments, duration):
    # Simplified DER calculation for diagnostics
    # ref/pred: list of {"start": s, "end": e, "speaker": spk}
    
    # Time grid (0.1s resolution)
    grid_size = 0.1
    num_steps = int(duration / grid_size)
    
    miss = 0
    fa = 0
    conf = 0
    total_speech = 0
    
    errors = []
    
    for i in range(num_steps):
        t = i * grid_size
        ref_spks = [s['speaker'] for s in ref_segments if s['start'] <= t < s['end']]
        pred_spks = [s['speaker'] for s in pred_segments if s.get('speaker') and s['start'] <= t < s['end']]
        
        if ref_spks: total_speech += 1
        
        if ref_spks and not pred_spks:
            miss += 1
            errors.append({"type": "miss", "time": t, "ref": ref_spks[0]})
        elif not ref_spks and pred_spks:
            fa += 1
            errors.append({"type": "fa", "time": t, "pred": pred_spks[0]})
        elif ref_spks and pred_spks and ref_spks[0] != pred_spks[0]:
            # This is complex because of mapping, but for diagnostics we use raw labels first
            # We will handle mapping in a more advanced way if needed
            conf += 1
            errors.append({"type": "conf", "time": t, "ref": ref_spks[0], "pred": pred_spks[0]})
            
    return {
        "miss": (miss * grid_size),
        "fa": (fa * grid_size),
        "conf": (conf * grid_size),
        "total_speech": (total_speech * grid_size),
        "errors": errors
    }

def extract_error_clips(audio_path, errors, output_dir, stem):
    audio, sr = librosa.load(audio_path, sr=None)
    duration = len(audio) / sr
    
    # Group errors by type and continuous blocks
    if not errors: return
    
    # Simple strategy: take the first few of each type
    error_types = ["miss", "fa", "conf"]
    for etype in error_types:
        type_errors = [e for e in errors if e['type'] == etype]
        if not type_errors: continue
        
        # Take up to 5 examples
        for idx, err in enumerate(type_errors[:5]):
            start = max(0, err['time'] - 2)
            end = min(duration, err['time'] + 2)
            
            clip = audio[int(start*sr):int(end*sr)]
            out_name = f"{stem}_{etype}_{idx}.wav"
            sf.write(output_dir / out_name, clip, sr)
            
            # Save metadata
            meta = {
                "type": etype,
                "start": err['time'],
                "context_start": start,
                "context_end": end,
                "ref_speaker": err.get("ref"),
                "pred_speaker": err.get("pred")
            }
            with open(output_dir / f"{stem}_{etype}_{idx}.json", 'w') as f:
                json.dump(meta, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Phase 5: Diarization Diagnostics")
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--gt-dir", default="ground_truth/parsed")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    
    audio_dir = Path(args.audio_dir)
    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for pred_path in pred_dir.glob("*.json"):
        stem = pred_path.stem.replace("_hardened", "")
        audio_path = audio_dir / f"{stem}.wav"
        gt_path = gt_dir / f"{stem}.json"
        
        if not (audio_path.exists() and gt_path.exists()): 
            print(f"Missing GT or Audio for {stem}")
            continue
            
        print(f"Analyzing: {stem}")
        
        with open(pred_path, 'r') as f:
            pred_data = json.load(f)
        pred_segments = pred_data.get("segments", [])
        
        gt_segments = load_ground_truth(gt_path)
        
        duration = librosa.get_duration(path=str(audio_path))
        stats = get_der_stats(gt_segments, pred_segments, duration)
        
        # Speaker Counts
        ref_spks = set([s['speaker'] for s in gt_segments])
        pred_spks = set([s['speaker'] for s in pred_segments if s.get('speaker')])
        
        res = {
            "file": stem,
            "total_speech": stats['total_speech'],
            "miss": stats['miss'],
            "fa": stats['fa'],
            "conf": stats['conf'],
            "der": (stats['miss'] + stats['fa'] + stats['conf']) / stats['total_speech'] if stats['total_speech'] > 0 else 0,
            "ref_spk_count": len(ref_spks),
            "pred_spk_count": len(pred_spks)
        }
        results.append(res)
        
        # Extract Clips
        extract_error_clips(audio_path, stats['errors'], output_dir, stem)
        
    df = pd.DataFrame(results)
    df.to_csv(output_dir / "diarization_diagnostics.csv", index=False)
    print(f"Diagnostics saved to {output_dir}")

if __name__ == "__main__":
    main()
