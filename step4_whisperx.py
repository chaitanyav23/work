import json
import argparse
import torch
import whisperx
from pathlib import Path
from tqdm import tqdm
import importlib.util
import os

PROJECT_ROOT = Path.cwd()
EXTERNAL_WHISPERX_DIR = PROJECT_ROOT / "external" / "WhisperX-Audio-Intelligence-Platform"

def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

whisperx_utils = _load_module("whisperx_utils", EXTERNAL_WHISPERX_DIR / "utils.py")

def main():
    BASE_DIR = Path("experiments/phase13_karen_generalization")
    AUDIO_PATH = BASE_DIR / "audio" / "karen_normalized.wav"
    GROUNDING_PATH = BASE_DIR / "grounding" / "karen_grounding.json"
    OUTPUT_JSON = BASE_DIR / "outputs" / "aligned_words.json"
    
    # 1. Load data
    print(f"Loading audio from {AUDIO_PATH}...")
    audio = whisperx.load_audio(str(AUDIO_PATH))
    with open(GROUNDING_PATH, "r") as f:
        grounding = json.load(f)
    segments = grounding["segments"]

    # 2. Load Whisper Model
    model_size = "large-v3"
    print(f"Loading Whisper {model_size}...")
    model = whisperx.load_model(
        model_size, 
        whisperx_utils.DEVICE, 
        compute_type="float16",
        language="en"
    )

    # 3. Transcribe
    print(f"Transcribing {len(segments)} segments...")
    transcription_results = []
    sample_rate = 16000
    
    for i, seg in enumerate(tqdm(segments, desc="Transcribing")):
        start_samp = int(seg["start"] * sample_rate)
        end_samp = int(seg["end"] * sample_rate)
        audio_slice = audio[start_samp:end_samp]
        
        # transcribe
        gen_result, _ = model.model.transcribe(
            audio_slice,
            language="en",
            task="transcribe",
            word_timestamps=False,
            condition_on_previous_text=False
        )
        
        full_text = ""
        for s in gen_result:
            full_text += s.text
            
        transcription_results.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": full_text.strip()
        })

    # 4. Align
    print("Loading Alignment Model...")
    model_a, metadata = whisperx.load_align_model(
        language_code="en", 
        device=whisperx_utils.DEVICE
    )

    print("Aligning...")
    alignment_result = whisperx.align(
        transcription_results, 
        model_a, 
        metadata, 
        audio, 
        whisperx_utils.DEVICE, 
        return_char_alignments=False
    )

    # 5. Extract and flatten words
    all_words = []
    for seg in alignment_result["word_segments"]:
        # Each segment has a 'word', 'start', 'end', 'score'
        # whisperx.align might return word_segments which are already flattened?
        # Let's check the structure of alignment_result
        all_words.append({
            "word": seg.get("word"),
            "start": seg.get("start"),
            "end": seg.get("end"),
            "score": seg.get("score")
        })

    # Save
    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_words, f, indent=2)
    
    print(f"Saved {len(all_words)} words to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
