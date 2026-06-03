import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from dotenv import dotenv_values
import whisperx
from pyannote.audio import Pipeline
from nemo.collections.asr.models import NeuralDiarizer
from omegaconf import OmegaConf

# Path setups
BASE_DIR = Path("experiments/phase11_dfn_ablation")
AUDIO_DIR = BASE_DIR / "audio_variants"
GROUNDING_DIR = BASE_DIR / "grounding"

OUT_11A = BASE_DIR / "outputs_phase11A"
OUT_11B = BASE_DIR / "outputs_phase11B"
OUT_11C = BASE_DIR / "outputs_phase11C"

NORM_WAV = AUDIO_DIR / "bwc_normalized.wav"
DFN_WAV = AUDIO_DIR / "bwc_deepfilternet.wav"

GROUNDING_11A = GROUNDING_DIR / "11A_grounding.json"
GROUNDING_11B = GROUNDING_DIR / "11B_grounding.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ENV_PATH = Path("external/WhisperX-Audio-Intelligence-Platform/.env")
ENV_VALUES = dotenv_values(ENV_PATH)
HF_TOKEN = ENV_VALUES.get("HF_TOKEN")

# Custom Fusion Logic from Phase 10
def assign_speakers_to_words(aligned_res, msdd_segments, tolerance=0.5):
    word_metadata = []
    for seg in aligned_res["segments"]:
        for word_info in seg.get("words", []):
            if "start" not in word_info or "end" not in word_info:
                continue
            w_start, w_end = word_info["start"], word_info["end"]
            
            overlaps = []
            for msdd in msdd_segments:
                if msdd["start"] <= w_end and msdd["end"] >= w_start:
                    overlap_start = max(w_start, msdd["start"])
                    overlap_end = min(w_end, msdd["end"])
                    overlap_dur = overlap_end - overlap_start
                    if overlap_dur > 0:
                        overlaps.append({
                            "speaker": msdd["speaker"],
                            "overlap_dur": overlap_dur
                        })
            
            assigned_spk = None
            if len(overlaps) >= 1:
                overlaps.sort(key=lambda x: x["overlap_dur"], reverse=True)
                assigned_spk = overlaps[0]["speaker"]
            else:
                nearest_spk = None
                min_dist = float('inf')
                for msdd in msdd_segments:
                    dist = min(abs(msdd["start"] - w_end), abs(msdd["end"] - w_start))
                    if dist < min_dist:
                        min_dist = dist
                        nearest_spk = msdd["speaker"]
                
                if min_dist <= tolerance and nearest_spk:
                    assigned_spk = nearest_spk
                else:
                    assigned_spk = "UNK"
            word_info["speaker"] = assigned_spk
            
    for seg in aligned_res["segments"]:
        speakers = []
        for w in seg.get("words", []):
            if "speaker" in w and w["speaker"] != "UNK":
                speakers.append(w["speaker"])
        if speakers:
            from collections import Counter
            seg["speaker"] = Counter(speakers).most_common(1)[0][0]
        else:
            seg["speaker"] = "UNK"
    return aligned_res

def write_rttm(segments, out_path, file_id="bwc"):
    with open(out_path, "w") as f:
        for seg in segments:
            duration = seg["end"] - seg["start"]
            f.write(f"SPEAKER {file_id} 1 {seg['start']:.3f} {duration:.3f} <NA> <NA> speaker <NA> <NA>\n")

def get_msdd_segments_from_rttm(rttm_path):
    segments = []
    with open(rttm_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8 and parts[0] == "SPEAKER":
                start = float(parts[3])
                duration = float(parts[4])
                speaker = parts[7]
                segments.append({"start": start, "end": start+duration, "speaker": speaker})
    return segments

def run_msdd(audio_path: Path, oracle_rttm_path: Path, output_dir: Path):
    rttm_out_path = output_dir / audio_path.stem / "pred_rttms" / f"{audio_path.stem}.rttm"
    if rttm_out_path.exists():
        print(f"Loading MSDD from cache: {rttm_out_path}")
        return get_msdd_segments_from_rttm(rttm_out_path)

    manifest_path = output_dir / f"{audio_path.stem}_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "audio_filepath": str(audio_path.absolute()),
            "offset": 0, "duration": None, "label": "infer", "text": "-",
            "num_speakers": 2, "rttm_filepath": str(oracle_rttm_path.absolute()),
        }, f)

    config = OmegaConf.create({
        "diarizer": {
            "manifest_filepath": str(manifest_path),
            "out_dir": str(output_dir / audio_path.stem),
            "oracle_vad": True, "collar": 0.25, "ignore_overlap": False,
            "vad": {"model_path": None, "external_vad_manifest": None, "parameters": {}},
            "speaker_embeddings": {
                "model_path": "titanet_large",
                "parameters": {
                    "window_length_in_sec": [1.5, 1.2, 1.0, 0.8, 0.5],
                    "shift_length_in_sec": [0.75, 0.6, 0.5, 0.4, 0.25],
                    "multiscale_weights": [1, 2, 3, 3, 3], "save_embeddings": True,
                },
            },
            "msdd_model": {
                "model_path": "diar_msdd_telephonic",
                "parameters": {
                    "use_backim_for_multiscale": True, "sigmoid_threshold": [0.7],
                    "use_adaptive_thres": False, "overlap_infer_spk_limit": 3,
                    "diar_window_length": 50, "infer_batch_size": 16,
                    "seq_eval_mode": False, "split_infer": True, "split_batch_size": 1,
                },
            },
            "clustering": {
                "parameters": {"max_num_speakers": 2, "oracle_num_speakers": True, "max_rp_threshold": 0.25, "sparse_search_volume": 30},
            },
        },
        "batch_size": 16, "num_workers": 4, "sample_rate": 16000, "verbose": True, "device": DEVICE,
    })

    print(f"Running MSDD: {audio_path.name}")
    msdd_model = NeuralDiarizer(cfg=config).to(torch.device(DEVICE))
    msdd_model.diarize()
    return get_msdd_segments_from_rttm(rttm_out_path)

def run_asr_and_align(audio_path, vad_segments, asr_model, align_model, align_metadata):
    audio = whisperx.load_audio(str(audio_path))
    sr = 16000
    transcript_segments = []
    
    print(f"Running WhisperX ASR on {len(vad_segments)} segments from {audio_path.name}...")
    for seg in tqdm(vad_segments, desc="ASR"):
        audio_slice = audio[int(seg["start"]*sr):int(seg["end"]*sr)]
        res, _ = asr_model.model.transcribe(
            audio_slice, language="en", task="transcribe", condition_on_previous_text=False
        )
        text = "".join([s.text for s in res]).strip()
        transcript_segments.append({"start": seg["start"], "end": seg["end"], "text": text})
        
    print("Running WhisperX Alignment...")
    aligned_res = whisperx.align(transcript_segments, align_model, align_metadata, audio, DEVICE)
    return aligned_res

def save_result(output_dir, final_result, stem="bwc"):
    out_json = output_dir / f"{stem}_final.json"
    out_txt = output_dir / f"{stem}_final.txt"
    with open(out_json, "w") as f:
        json.dump(final_result, f, indent=2, default=str)
    with open(out_txt, "w") as f:
        for s in final_result["segments"]:
            f.write(f"[{s.get('start', 0.0):06.2f}s] {s.get('speaker', 'UNK'):10}: {s.get('text', '')}\n")

def main():
    print("Loading Models...")
    asr_model = whisperx.load_model("large-v3", DEVICE, compute_type="float16")
    align_model, align_metadata = whisperx.load_align_model(language_code="en", device=DEVICE)
    
    with open(GROUNDING_11A, "r") as f: vad_11A = json.load(f)["segments"]
    with open(GROUNDING_11B, "r") as f: vad_11B = json.load(f)["segments"]
    
    oracle_rttm_11A = GROUNDING_DIR / "11A_oracle.rttm"
    oracle_rttm_11B = GROUNDING_DIR / "11B_oracle.rttm"
    write_rttm(vad_11A, oracle_rttm_11A)
    write_rttm(vad_11B, oracle_rttm_11B)

    # =========================================================
    # PIPELINE 11A: Control (Norm Grounding -> Norm ASR -> Norm MSDD)
    # =========================================================
    print("\n--- Running Pipeline 11A (Control) ---")
    OUT_11A.mkdir(parents=True, exist_ok=True)
    msdd_11A = run_msdd(NORM_WAV, oracle_rttm_11A, OUT_11A)
    aligned_11A = run_asr_and_align(NORM_WAV, vad_11A, asr_model, align_model, align_metadata)
    final_11A = assign_speakers_to_words(aligned_11A, msdd_11A)
    save_result(OUT_11A, final_11A)

    # =========================================================
    # PIPELINE 11B: Full DFN (DFN Grounding -> DFN ASR -> Norm MSDD)
    # Note: We must use Norm MSDD because DFN breaks MSDD completely (proven in Phase 4).
    # =========================================================
    print("\n--- Running Pipeline 11B (Full DFN) ---")
    OUT_11B.mkdir(parents=True, exist_ok=True)
    msdd_11B = run_msdd(NORM_WAV, oracle_rttm_11B, OUT_11B)
    aligned_11B = run_asr_and_align(DFN_WAV, vad_11B, asr_model, align_model, align_metadata)
    final_11B = assign_speakers_to_words(aligned_11B, msdd_11B)
    save_result(OUT_11B, final_11B)

    # =========================================================
    # PIPELINE 11C: Fixed-Grounding DFN (Norm Grounding -> DFN ASR -> Norm MSDD)
    # This isolates exactly the effect DFN has on Whisper.
    # =========================================================
    print("\n--- Running Pipeline 11C (Fixed-Grounding DFN) ---")
    OUT_11C.mkdir(parents=True, exist_ok=True)
    # Use exact same MSDD from 11A to ensure perfect isolation
    msdd_11C = msdd_11A 
    aligned_11C = run_asr_and_align(DFN_WAV, vad_11A, asr_model, align_model, align_metadata)
    final_11C = assign_speakers_to_words(aligned_11C, msdd_11C)
    save_result(OUT_11C, final_11C)

    print("\nPhase 11 Pipelines Execution Complete!")

if __name__ == "__main__":
    main()
