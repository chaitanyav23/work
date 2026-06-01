import json
import torch
import numpy as np
import pandas as pd
import whisperx
from pathlib import Path
from tqdm import tqdm
import argparse
from dotenv import dotenv_values

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    parser = argparse.ArgumentParser(description="Phase 3 Runner: Hardened ASR + Custom Diar")
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--grounding-dir", required=True)
    parser.add_argument("--diar-json-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    grounding_dir = Path(args.grounding_dir)
    diar_json_dir = Path(args.diar_json_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- Loading Hardened Engine (Device: {DEVICE}) ---")
    asr_model = whisperx.load_model("large-v3", DEVICE, compute_type="float16")

    for g_path in grounding_dir.glob("*.json"):
        stem = g_path.stem.replace("_grounding", "")
        audio_path = audio_dir / f"{stem}.wav"
        diar_path = diar_json_dir / f"{stem}_diarization.json"
        
        if not (audio_path.exists() and diar_path.exists()): continue
        
        print(f"\nProcessing: {stem}")
        
        # 1. Load Data
        audio = whisperx.load_audio(str(audio_path))
        with open(g_path, "r") as f:
            grounding = json.load(f)
        segments = grounding["segments"]
        sr = 16000

        # 2. Load Custom Diarization
        with open(diar_path, "r") as f:
            diar_list = json.load(f)
        diar_df = pd.DataFrame(diar_list)

        # 3. Hardened ASR
        transcript_segments = []
        for seg in tqdm(segments, desc="Transcribing"):
            audio_slice = audio[int(seg["start"]*sr):int(seg["end"]*sr)]
            res, _ = asr_model.model.transcribe(audio_slice, language="en", task="transcribe", condition_on_previous_text=False)
            text = "".join([s.text for s in res]).strip()
            transcript_segments.append({"start": seg["start"], "end": seg["end"], "text": text})

        # 4. Alignment & Speaker Merge
        model_a, metadata = whisperx.load_align_model(language_code="en", device=DEVICE)
        aligned_res = whisperx.align(transcript_segments, model_a, metadata, audio, DEVICE)
        final_result = whisperx.assign_word_speakers(diar_df, aligned_res)

        # 5. Output
        out_json = output_dir / f"{stem}_hardened.json"
        with open(out_json, "w") as f:
            json.dump(final_result, f, indent=2, default=str)
            
    print(f"[SUCCESS] Saved to: {output_dir}")

if __name__ == "__main__":
    main()
