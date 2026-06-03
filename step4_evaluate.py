import sys, os; sys.path.insert(0, os.getcwd())
import os
import json
import subprocess
import pandas as pd
from pathlib import Path
import numpy as np

BASE_DIR = Path("experiments/phase11_dfn_ablation")
REPORTS_DIR = BASE_DIR / "reports"
DIAG_DIR = BASE_DIR / "diagnostics"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

for d in ["vad_errors", "asr_errors", "diarization_errors"]:
    (DIAG_DIR / d).mkdir(parents=True, exist_ok=True)

PIPELINES = {
    "11A": BASE_DIR / "outputs_phase11A",
    "11B": BASE_DIR / "outputs_phase11B",
    "11C": BASE_DIR / "outputs_phase11C"
}

def generate_vad_diagnostics(gt_segments, pred_segments, label, duration=300, step=0.01):
    n_frames = int(duration / step) + 1
    gt_frames = np.zeros(n_frames, dtype=int)
    pred_frames = np.zeros(n_frames, dtype=int)
    
    for s in gt_segments:
        gt_frames[int(s[0]/step):int(s[1]/step)] = 1
    for s in pred_segments:
        pred_frames[int(s[0]/step):int(s[1]/step)] = 1
        
    miss_frames = (gt_frames == 1) & (pred_frames == 0)
    fa_frames = (gt_frames == 0) & (pred_frames == 1)
    
    def extract_regions(mask):
        regions = []
        in_region = False
        start_idx = 0
        for i in range(len(mask)):
            if mask[i] and not in_region:
                start_idx = i
                in_region = True
            elif not mask[i] and in_region:
                regions.append({"start": start_idx*step, "end": i*step})
                in_region = False
        if in_region:
            regions.append({"start": start_idx*step, "end": len(mask)*step})
        return regions
        
    miss_regions = extract_regions(miss_frames)
    fa_regions = extract_regions(fa_frames)
    
    out_file = DIAG_DIR / "vad_errors" / f"{label}_vad_errors.json"
    with open(out_file, "w") as f:
        json.dump({"miss": miss_regions, "false_alarm": fa_regions}, f, indent=2)

def evaluate_vad():
    from scripts.evaluate_vad import parse_ground_truth, segments_to_frame_labels, compute_segment_metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    gt_path = Path("ground_truth/bwc.txt")
    gt_segments = parse_ground_truth(gt_path)
    duration = 180.0 # Approximation
    if gt_segments:
        duration = max([s[1] for s in gt_segments]) + 10.0
    y_true = segments_to_frame_labels(gt_segments, duration)
    
    records = []
    groundings = {"11A": BASE_DIR / "grounding" / "11A_grounding.json", 
                  "11B": BASE_DIR / "grounding" / "11B_grounding.json"}
    
    for label, path in groundings.items():
        with open(path) as f:
            pred_segments = json.load(f)["segments"]
            # pred_segments are dicts {"start": x, "end": y}
            pred_segments_tuples = [(s["start"], s["end"]) for s in pred_segments]
        
        generate_vad_diagnostics(gt_segments, pred_segments_tuples, label, duration)
        
        y_pred = segments_to_frame_labels(pred_segments_tuples, duration)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        miss_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        fa_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        records.append({
            "Pipeline": label,
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1": f1_score(y_true, y_pred, zero_division=0),
            "Miss Rate": miss_rate,
            "False Alarm Rate": fa_rate
        })
        
    df = pd.DataFrame(records)
    df.to_csv(REPORTS_DIR / "vad_metrics.csv", index=False)

def export_asr_errors(label, out_dir):
    hyp_json = out_dir / "bwc_final.json"
    if not hyp_json.exists(): return
    with open(hyp_json, "r") as f:
        data = json.load(f)
    
    # Just dump segments as context for errors
    out_file = DIAG_DIR / "asr_errors" / f"{label}_asr_segments.json"
    with open(out_file, "w") as f:
        json.dump(data.get("segments", []), f, indent=2)

def evaluate_all():
    print("Evaluating VAD...")
    evaluate_vad()
    
    for label, out_dir in PIPELINES.items():
        print(f"\nEvaluating {label}...")
        
        # 1. ASR Eval
        subprocess.run([
            "python", "evaluate_asr_improved.py",
            "--hyp-dir", str(out_dir),
            "--is-json",
            "--output-name", f"phase11_{label}_asr"
        ], check=False)
        
        export_asr_errors(label, out_dir)
        
        # 2. Diarization Eval
        subprocess.run([
            "python", "evaluate_diarization_improved.py",
            "--pred-dir", str(out_dir),
            "--output-name", f"phase11_{label}_diar"
        ], check=False)
        
        # 3. Conversational Eval
        subprocess.run([
            "python", "evaluate_conversational_timeline.py",
            "--pred-dir", str(out_dir),
            "--output-name", f"phase11_{label}_conv"
        ], check=False)
        
        # 4. Error Diagnostics for Diarization
        err_dir = DIAG_DIR / "diarization_errors" / label
        err_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            "python", "analyze_diarization_errors.py",
            "--pred-dir", str(out_dir),
            "--audio-dir", str(BASE_DIR / "audio_variants"),
            "--output-dir", str(err_dir)
        ], check=False)

def compile_metrics():
    # Consolidate standard reports
    print("\nConsolidating Metrics...")
    import shutil
    
    # Just copy/aggregate the generated csvs into the requested names
    for label in PIPELINES.keys():
        pass # Will handle in step5_compile_report.py
    
if __name__ == "__main__":
    evaluate_all()
    compile_metrics()
