"""
run_final_hardened_pipeline.py
==============================
The production-ready 'Hardened' pipeline for BWC.
Integrates Grounded VAD, Muted Pyannote 3.1, and Manual Slice ASR.
"""

import json
import torch
import numpy as np
import pandas as pd
import whisperx
from pathlib import Path
from tqdm import tqdm
from pyannote.audio import Pipeline
from dotenv import dotenv_values
import argparse
import importlib.util

# Paths
PROJECT_ROOT = Path.cwd()
RAW_AUDIO_DIR = PROJECT_ROOT / "raw_audio"
GROUNDING_DIR = PROJECT_ROOT / "vad_custom_grounding"
EXTERNAL_WX_DIR = PROJECT_ROOT / "external" / "WhisperX-Audio-Intelligence-Platform"
OUTPUT_DIR = PROJECT_ROOT / "final_bwc_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Auth & Device
ENV_PATH = EXTERNAL_WX_DIR / ".env"
ENV_VALUES = dotenv_values(ENV_PATH)
HF_TOKEN = ENV_VALUES.get("HF_TOKEN")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

wx_utils = _load_module("wx_utils", EXTERNAL_WX_DIR / "utils.py")

def process_file(audio_path, grounding_path, asr_model, diarizer, args):
    stem = audio_path.stem
    out_json = OUTPUT_DIR / f"{stem}_hardened.json"
    out_txt = OUTPUT_DIR / f"{stem}_hardened.txt"

    if not args.overwrite and out_json.exists() and out_txt.exists():
        print(f"Skipping {stem} (outputs already exist).")
        return

    print(f"\n[PHASE 1] Initializing Grounded Pipeline for: {stem}")
    
    # 1. Load Data
    audio = whisperx.load_audio(str(audio_path))
    with open(grounding_path, "r") as f:
        grounding = json.load(f)
    segments = grounding["segments"]
    sr = 16000

    # 2. Grounded Diarization (The 'Muting' Trick)
    print(f"[PHASE 2] Running Grounded Diarization (Pyannote 3.1 + Muting)")
    mask = np.zeros_like(audio)
    for seg in segments:
        mask[int(seg["start"]*sr):int(seg["end"]*sr)] = 1.0
    muted_audio = audio * mask
    
    audio_tensor = torch.from_numpy(muted_audio[None, :]).to(DEVICE)
    diar_res = diarizer(
        {"waveform": audio_tensor, "sample_rate": sr},
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers
    )
    
    diar_list = []
    for turn, _, speaker in diar_res.speaker_diarization.itertracks(yield_label=True):
        diar_list.append({"start": turn.start, "end": turn.end, "speaker": speaker})
    diar_df = pd.DataFrame(diar_list)

    # 3. Hardened ASR (Manual Slicing, No Conditioning)
    print(f"[PHASE 3] Running Hardened ASR (Manual Slices, language={args.language})")
    transcript_segments = []
    for seg in tqdm(segments, desc="Transcribing"):
        audio_slice = audio[int(seg["start"]*sr):int(seg["end"]*sr)]
        res, _ = asr_model.model.transcribe(
            audio_slice, language=args.language, task="transcribe", 
            condition_on_previous_text=False
        )
        text = "".join([s.text for s in res]).strip()
        transcript_segments.append({"start": seg["start"], "end": seg["end"], "text": text})

    # 4. Alignment & Speaker Merge
    print("[PHASE 4] Running Word-Level Alignment & Speaker Tagging")
    model_a, metadata = whisperx.load_align_model(language_code=args.language, device=DEVICE)
    aligned_res = whisperx.align(transcript_segments, model_a, metadata, audio, DEVICE)
    final_result = whisperx.assign_word_speakers(diar_df, aligned_res)

    # 5. Output Generation
    with open(out_json, "w") as f:
        json.dump(final_result, f, indent=2, default=str)
    
    with open(out_txt, "w") as f:
        for s in final_result["segments"]:
            spk = s.get("speaker", "UNKNOWN")
            f.write(f"[{s['start']:06.2f}s] {spk:10}: {s['text']}\n")

    print(f"[SUCCESS] Final Intelligence saved to: {OUTPUT_DIR}")

def main():
    parser = argparse.ArgumentParser(description="Final Hardened BWC Pipeline")
    parser.add_argument("--language", default="en", help="Language code (default: en)")
    parser.add_argument("--min-speakers", type=int, default=None, help="Minimum number of speakers")
    parser.add_argument("--max-speakers", type=int, default=None, help="Maximum number of speakers")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    args = parser.parse_args()

    if not HF_TOKEN:
        print("Error: HF_TOKEN required for Pyannote 3.1")
        return

    print(f"--- Loading Hardened Engines (Device: {DEVICE}) ---")
    asr_model = whisperx.load_model("large-v3", DEVICE, compute_type="float16")
    diarizer = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=HF_TOKEN).to(torch.device(DEVICE))

    grounding_files = sorted(list(GROUNDING_DIR.glob("*.json")))
    for g_path in grounding_files:
        stem = g_path.stem.replace("_grounding", "")
        audio_path = RAW_AUDIO_DIR / f"{stem}.wav"
        if audio_path.exists():
            process_file(audio_path, g_path, asr_model, diarizer, args)

if __name__ == "__main__":
    main()
