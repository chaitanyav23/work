import json
import torch
import numpy as np
import argparse
import os
from pathlib import Path
from nemo.collections.asr.models import NeuralDiarizer
from omegaconf import OmegaConf
import sys

def main():
    parser = argparse.ArgumentParser(description="Phase 3: NeMo MSDD Diarization")
    parser.add_argument("--audio-dir", default="raw_audio")
    parser.add_argument("--output-dir", default="experiments/phase3_diarization/outputs_msdd")
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for audio_path in audio_dir.glob("*.wav"):
        print(f"Diarizing: {audio_path.name}")
        
        manifest_path = output_dir / f"{audio_path.stem}_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump({
                "audio_filepath": str(audio_path.absolute()),
                "offset": 0,
                "duration": None,
                "label": "infer",
                "text": "-",
                "num_speakers": None,
                "rttm_filepath": None
            }, f)

        config = OmegaConf.create({
            "diarizer": {
                "manifest_filepath": str(manifest_path),
                "out_dir": str(output_dir / audio_path.stem),
                "oracle_vad": False,
                "collar": 0.25,
                "ignore_overlap": False,
                "vad": {
                    "model_path": "vad_multilingual_marblenet",
                    "parameters": {
                        "window_length_in_sec": 0.15,
                        "shift_length_in_sec": 0.01,
                        "threshold": 0.5,
                        "activation_threshold": 0.5,
                        "deactivation_threshold": 0.4,
                        "smoothing": "mean",
                        "overlap": 0.5
                    }
                },
                "speaker_embeddings": {
                    "model_path": "titanet_large",
                    "parameters": {
                        "window_length_in_sec": [1.5, 1.2, 1.0, 0.8, 0.5],
                        "shift_length_in_sec": [0.75, 0.6, 0.5, 0.4, 0.25],
                        "multiscale_weights": [1, 1, 1, 1, 1],
                        "save_embeddings": False
                    }
                },
                "msdd_model": {
                    "model_path": "diar_msdd_telephonic",
                    "parameters": {
                        "use_backim_for_multiscale": True,
                        "sigmoid_threshold": 0.7,
                        "diar_window_length": 50,
                        "infer_batch_size": 16,
                        "seq_eval_mode": False,
                        "split_batch_size": 1
                    }
                },
                "clustering": {
                    "parameters": {
                        "max_num_speakers": 20
                    }
                }
            },
            "batch_size": 16,
            "num_workers": 4,
            "sample_rate": 16000,
            "verbose": True,
            "device": device
        })
        
        msdd_model = NeuralDiarizer(cfg=config).to(torch.device(device))
        msdd_model.diarize()
        
        rttm_path = output_dir / audio_path.stem / "pred_rttms" / f"{audio_path.stem}.rttm"
        
        diar_list = []
        if rttm_path.exists():
            with open(rttm_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    start = float(parts[3])
                    end = start + float(parts[4])
                    speaker = parts[7]
                    diar_list.append({"start": start, "end": end, "speaker": speaker})
        
        with open(output_dir / f"{audio_path.stem}_diarization.json", 'w') as f:
            json.dump(diar_list, f)

if __name__ == "__main__":
    main()
