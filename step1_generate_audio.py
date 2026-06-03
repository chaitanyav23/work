import os
import subprocess
from pathlib import Path
import hashlib
import librosa
import numpy as np
import pyloudnorm as pyln
import csv

BASE_DIR = Path("experiments/phase11_dfn_ablation")
AUDIO_DIR = BASE_DIR / "audio_variants"
REPORT_DIR = BASE_DIR / "reports"

RAW_BWC = Path("raw_audio/bwc.wav")
NORM_WAV = AUDIO_DIR / "bwc_normalized.wav"
DFN_WAV = AUDIO_DIR / "bwc_deepfilternet.wav"

def compute_hash(filepath, algo='md5'):
    h = hashlib.new(algo)
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def analyze_audio(filepath):
    print(f"Analyzing {filepath.name}...")
    y, sr = librosa.load(filepath, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    
    meter = pyln.Meter(sr)
    lufs = meter.integrated_loudness(y)
    
    rms = np.sqrt(np.mean(y**2))
    peak = np.max(np.abs(y))
    
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    
    return {
        "file": filepath.name,
        "md5": compute_hash(filepath, 'md5'),
        "sha256": compute_hash(filepath, 'sha256'),
        "duration": float(duration),
        "sample_rate": int(sr),
        "LUFS": float(lufs),
        "RMS": float(rms),
        "Peak": float(peak),
        "Spectral_Centroid_Mean": float(np.mean(cent)),
        "Spectral_Rolloff_Mean": float(np.mean(rolloff))
    }

def main():
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("1. Generating Normalized Audio (Phase 6 V2 settings)...")
    if not NORM_WAV.exists():
        cmd = [
            "ffmpeg", "-y", "-i", str(RAW_BWC),
            "-ar", "16000", "-ac", "1",
            "-af", "loudnorm=I=-20:TP=-1.5:LRA=11",
            str(NORM_WAV)
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        print("Normalized audio exists, skipping generation.")

    print("2. Generating DeepFilterNet Enhanced Audio...")
    if not DFN_WAV.exists():
        cmd = [
            "python", "-m", "df.enhance", 
            str(NORM_WAV), 
            "--output-dir", str(AUDIO_DIR)
        ]
        subprocess.run(cmd, check=True)
        # DeepFilterNet outputs as {stem}_DeepFilterNet3.wav
        dfn_out = AUDIO_DIR / f"{NORM_WAV.stem}_DeepFilterNet3.wav"
        if dfn_out.exists():
            os.rename(dfn_out, DFN_WAV)
    else:
        print("DeepFilterNet audio exists, skipping generation.")

    print("3. Generating Audio Audit Report...")
    audit_results = [analyze_audio(NORM_WAV), analyze_audio(DFN_WAV)]
    
    csv_path = REPORT_DIR / "audio_audit.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=audit_results[0].keys())
        writer.writeheader()
        writer.writerows(audit_results)
        
    print(f"Audio generation complete. Audit saved to {csv_path}")

if __name__ == "__main__":
    main()
