import os
import json
import pickle
import glob
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.neighbors import NearestNeighbors
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import torch

# --- Global Analysis Functions (Original) ---

def setup_directories(base_dir, is_local=False):
    if is_local:
        dirs = ['local_windows/pca', 'local_windows/umap', 'local_windows/trajectories', 
                'local_windows/csv', 'local_windows/metrics', 'local_windows/reports']
    else:
        dirs = ['pca', 'umap', 'combined', 'trajectories', 'metrics', 'reports']
    
    for d in dirs:
        (base_dir / d).mkdir(parents=True, exist_ok=True)

def load_data(scale_num, input_dir):
    pkl_path = input_dir / f"embeddings/subsegments_scale{scale_num}_embeddings.pkl"
    json_path = input_dir / f"subsegments_scale{scale_num}.json"
    label_path = input_dir / f"subsegments_scale{scale_num}_cluster.label"

    if not pkl_path.exists() or not json_path.exists():
        print(f"Skipping scale {scale_num}: Data not found.")
        return None

    print(f"Loading scale {scale_num}...")
    
    with open(pkl_path, 'rb') as f:
        emb_dict = pickle.load(f)
    
    metadata = []
    with open(json_path, 'r') as f:
        for line in f:
            metadata.append(json.loads(line.strip()))
            
    from collections import defaultdict
    meta_by_audio = defaultdict(list)
    for m in metadata:
        audio_id = Path(m['audio_filepath']).stem
        meta_by_audio[audio_id].append(m)
        
    all_embeddings = []
    all_metadata = []
    
    for audio_id, emb_tensor in emb_dict.items():
        if audio_id in meta_by_audio:
            metas = meta_by_audio[audio_id]
            metas = sorted(metas, key=lambda x: x['offset'])
            
            num_emb = emb_tensor.shape[0]
            if len(metas) != num_emb:
                min_len = min(len(metas), num_emb)
                all_embeddings.append(emb_tensor[:min_len].cpu().numpy())
                all_metadata.extend(metas[:min_len])
            else:
                all_embeddings.append(emb_tensor.cpu().numpy())
                all_metadata.extend(metas)
                
    if not all_embeddings:
        return None
        
    embeddings_mat = np.vstack(all_embeddings)
    
    labels = None
    if label_path.exists():
        label_df = pd.read_csv(label_path, sep='\s+', header=None, names=['audio_id', 'start', 'end', 'label'])
        label_list = []
        for m in all_metadata:
            audio_id = Path(m['audio_filepath']).stem
            start = float(m['offset'])
            match = label_df[(label_df['audio_id'] == audio_id) & (np.abs(label_df['start'] - start) < 0.01)]
            if len(match) > 0:
                label_list.append(match.iloc[0]['label'])
            else:
                label_list.append('UNKNOWN')
        labels = np.array(label_list)
        
    return {
        'embeddings': embeddings_mat,
        'metadata': all_metadata,
        'labels': labels
    }

def plot_projection(proj, times, title, xlabel, ylabel, save_path):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(proj[:, 0], proj[:, 1], c=times, cmap='viridis', s=15, alpha=0.7)
    plt.colorbar(scatter, label='Temporal Progression (s)')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_trajectory(proj, times, title, xlabel, ylabel, save_path):
    plt.figure(figsize=(10, 8))
    plt.plot(proj[:, 0], proj[:, 1], color='gray', alpha=0.3, linewidth=1)
    scatter = plt.scatter(proj[:, 0], proj[:, 1], c=times, cmap='viridis', s=20, alpha=0.8, zorder=5)
    plt.colorbar(scatter, label='Temporal Progression (s)')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def compute_metrics(emb_mat, pca, labels):
    N, D = emb_mat.shape
    exp_var = float(np.sum(pca.explained_variance_ratio_))
    dist_mat = pairwise_distances(emb_mat, metric='cosine')
    triu_indices = np.triu_indices(N, k=1)
    dists = dist_mat[triu_indices]
    
    if N > 1:
        nn = NearestNeighbors(n_neighbors=min(6, N), metric='cosine')
        nn.fit(emb_mat)
        distances, _ = nn.kneighbors(emb_mat)
        mean_intra_nn = float(np.mean(distances[:, 1:]))
    else:
        mean_intra_nn = 0.0
        
    sil_score = None
    if labels is not None:
        valid_idx = labels != 'UNKNOWN'
        if np.sum(valid_idx) > 1 and len(np.unique(labels[valid_idx])) > 1:
            sil_score = float(silhouette_score(emb_mat[valid_idx], labels[valid_idx], metric='cosine'))

    global_mean_dist = float(np.mean(dists)) if len(dists) > 0 else 0.0
    compactness = "compact" if (mean_intra_nn < 0.5 * global_mean_dist) else "diffuse"

    return {
        'num_embeddings': N,
        'dimensionality': D,
        'pca_explained_variance_2d': exp_var,
        'silhouette_score': sil_score,
        'mean_intra_neighbor_distance': mean_intra_nn,
        'pairwise_cosine_distance': {
            'mean': global_mean_dist,
            'std': float(np.std(dists)) if len(dists) > 0 else 0.0,
            'min': float(np.min(dists)) if len(dists) > 0 else 0.0,
            'max': float(np.max(dists)) if len(dists) > 0 else 0.0
        },
        'distribution': compactness
    }

# --- Local Analysis Functions ---

def load_rttm(rttm_path):
    rttm_data = []
    if not os.path.exists(rttm_path):
        print(f"Warning: RTTM file not found at {rttm_path}")
        return rttm_data
    with open(rttm_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8 and parts[0] == "SPEAKER":
                start = float(parts[3])
                duration = float(parts[4])
                speaker = parts[7]
                rttm_data.append({'speaker': speaker, 'start': start, 'end': start + duration})
    return rttm_data

def assign_speaker(emb_start, emb_end, rttm_data):
    emb_duration = emb_end - emb_start
    if emb_duration <= 0: return "UNKNOWN", 0.0, {}
    
    overlaps = {}
    for r in rttm_data:
        o_start = max(emb_start, r['start'])
        o_end = min(emb_end, r['end'])
        if o_end > o_start:
            spk = r['speaker']
            overlaps[spk] = overlaps.get(spk, 0.0) + (o_end - o_start)
    
    if not overlaps:
        return "UNKNOWN", 0.0, {}
        
    overlap_ratios = {spk: dur / emb_duration for spk, dur in overlaps.items()}
    sorted_spks = sorted(overlap_ratios.items(), key=lambda x: x[1], reverse=True)
    
    best_spk, best_ratio = sorted_spks[0]
    
    if len(sorted_spks) > 1:
        second_spk, second_ratio = sorted_spks[1]
        if second_ratio > 0.2 and (best_ratio - second_ratio) < 0.15:
            return "MULTI_SPEAKER", best_ratio, overlap_ratios
            
    if best_ratio < 0.1:
        return "UNKNOWN", best_ratio, overlap_ratios
    elif best_ratio < 0.4:
        return f"LOW_CONFIDENCE_{best_spk}", best_ratio, overlap_ratios
    
    return best_spk, best_ratio, overlap_ratios

def plot_local_trajectory(proj, times, labels, title, save_path, xlabel, ylabel):
    plt.figure(figsize=(10, 8))
    
    # Pre-define color palette
    unique_labels = list(set(labels))
    palette = sns.color_palette("tab10", 10)
    color_map = {}
    for l in unique_labels:
        if l == "UNKNOWN": color_map[l] = "black"
        elif l == "MULTI_SPEAKER": color_map[l] = "gray"
        elif l.startswith("LOW_CONFIDENCE_"): color_map[l] = "pink"
        else:
            # Hash to tab10
            idx = sum(ord(c) for c in l) % 10
            color_map[l] = palette[idx]
            
    # Trajectory lines
    plt.plot(proj[:, 0], proj[:, 1], color='gray', alpha=0.5, linewidth=1.5, zorder=1)
    
    # Points by speaker
    for l in unique_labels:
        idx = [i for i, x in enumerate(labels) if x == l]
        plt.scatter(proj[idx, 0], proj[idx, 1], c=[color_map[l]], label=l, s=80, edgecolor='white', zorder=5)
        
    # Annotate times
    for i, (x, y) in enumerate(proj):
        plt.annotate(f"{times[i]:.1f}s", (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)
        
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def process_local_window(w_start, w_end, embeddings, metadata, rttm_data, output_dir, scale):
    indices = [i for i, m in enumerate(metadata) if w_start <= m['offset'] <= w_end]
    if len(indices) < 3:
        return None
        
    local_embs = embeddings[indices]
    local_meta = [metadata[i] for i in indices]
    times = [m['offset'] for m in local_meta]
    
    labels = []
    confidences = []
    
    for m in local_meta:
        spk, conf, _ = assign_speaker(m['offset'], m['offset'] + m['duration'], rttm_data)
        labels.append(spk)
        confidences.append(conf)
        
    # Transitions count
    transitions = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[i-1]:
            transitions += 1
            
    active_speakers = [l for l in set(labels) if not l.startswith("LOW_CONFIDENCE") and l not in ["UNKNOWN", "MULTI_SPEAKER"]]
            
    # PCA
    pca = PCA(n_components=2)
    pca_proj = pca.fit_transform(local_embs)
    exp_var = float(np.sum(pca.explained_variance_ratio_))
    
    # UMAP
    n_samples = len(local_embs)
    if n_samples >= 4:
        n_neighbors = min(15, n_samples - 2)
        reducer = umap.UMAP(n_components=2, n_neighbors=max(2, n_neighbors), random_state=42)
        umap_proj = reducer.fit_transform(local_embs)
    else:
        # Fallback if too few points for UMAP
        umap_proj = pca_proj
        
    # Velocity (smoothness)
    dists = [np.linalg.norm(local_embs[i] - local_embs[i-1]) for i in range(1, len(local_embs))]
    velocity = float(np.mean(dists)) if dists else 0.0
    
    # Metrics
    metrics = {
        'window_start': w_start,
        'window_end': w_end,
        'num_embeddings': len(local_embs),
        'transitions': transitions,
        'active_speakers': active_speakers,
        'pca_explained_var': exp_var,
        'mean_velocity': velocity,
        'fragmentation_score': transitions / max(1, len(local_embs))
    }
    
    # Save Outputs
    w_id = f"{w_start:.1f}_{w_end:.1f}"
    
    # CSV
    df = pd.DataFrame({
        'offset': times,
        'duration': [m['duration'] for m in local_meta],
        'assigned_speaker': labels,
        'confidence': confidences,
        'pca_1': pca_proj[:, 0],
        'pca_2': pca_proj[:, 1],
        'umap_1': umap_proj[:, 0],
        'umap_2': umap_proj[:, 1]
    })
    df.to_csv(output_dir / f"csv/local_scale{scale}_{w_id}.csv", index=False)
    
    # JSON
    with open(output_dir / f"metrics/local_scale{scale}_{w_id}.json", 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # Plots
    title_base = f"Local [{w_start:.1f}s - {w_end:.1f}s] | Spks: {len(active_speakers)} | Trans: {transitions}"
    plot_local_trajectory(pca_proj, times, labels, f"PCA {title_base}", output_dir / f"pca/local_scale{scale}_{w_id}_pca.png", "PCA 1", "PCA 2")
    plot_local_trajectory(umap_proj, times, labels, f"UMAP {title_base}", output_dir / f"umap/local_scale{scale}_{w_id}_umap.png", "UMAP 1", "UMAP 2")
    
    return metrics

def run_local_analysis(args):
    base_dir = Path("experiments/phase13_karen_generalization")
    input_dir = base_dir / "diarization/msdd_outputs/speaker_outputs"
    output_dir = base_dir / "embedding_analysis"
    rttm_path = base_dir / "diarization/msdd_outputs/pred_rttms/karen_normalized.rttm"
    
    setup_directories(output_dir, is_local=True)
    out_local = output_dir / "local_windows"
    
    rttm_data = load_rttm(rttm_path)
    print(f"Loaded {len(rttm_data)} RTTM segments.")
    
    data = load_data(args.scale, input_dir)
    if data is None: return
    
    embs = data['embeddings']
    meta = data['metadata']
    
    max_time = max([m['offset'] for m in meta]) if meta else 0.0
    
    windows = []
    start_t = args.start if args.start is not None else 0.0
    
    if args.auto_slide:
        end_t = args.end if args.end is not None else max_time
        curr = start_t
        while curr < end_t:
            windows.append((curr, curr + args.window_size))
            curr += args.stride
    elif args.end is not None and args.end > start_t:
        curr = start_t
        while curr < args.end:
            windows.append((curr, min(curr + args.window_size, args.end)))
            curr += args.stride
    else:
        end_t = args.end if args.end is not None else (start_t + args.window_size)
        windows.append((start_t, end_t))
        
    print(f"Processing {len(windows)} local windows for Scale {args.scale}...")
    
    all_metrics = []
    for w_start, w_end in windows:
        print(f"  -> Window [{w_start:.1f}s - {w_end:.1f}s]")
        m = process_local_window(w_start, w_end, embs, meta, rttm_data, out_local, args.scale)
        if m:
            all_metrics.append(m)
            
    # Global report
    if all_metrics:
        print("Generating local summary report...")
        report_lines = [
            "# Local Conversational Trajectory Analysis",
            f"- Scale: {args.scale}",
            f"- Window Size: {args.window_size}s",
            f"- Total Windows Analyzed: {len(all_metrics)}",
            "",
            "## Summary Metrics",
            f"- Average Transitions per Window: {np.mean([m['transitions'] for m in all_metrics]):.2f}",
            f"- Average Embedding Velocity: {np.mean([m['mean_velocity'] for m in all_metrics]):.4f}",
            f"- Average Fragmentation Score: {np.mean([m['fragmentation_score'] for m in all_metrics]):.4f}",
            "",
            "## Highest Transition Windows (Fragmentation Artifacts)",
        ]
        
        # Sort by transitions
        sorted_by_trans = sorted(all_metrics, key=lambda x: x['transitions'], reverse=True)
        for m in sorted_by_trans[:5]:
            report_lines.append(f"- [{m['window_start']:.1f}s - {m['window_end']:.1f}s]: {m['transitions']} transitions, {len(m['active_speakers'])} active speakers")
            
        with open(out_local / "reports/local_analysis_summary.md", 'w') as f:
            f.write("\n".join(report_lines))
            
        # Summary CSV
        pd.DataFrame(all_metrics).to_csv(out_local / "reports/all_windows_metrics.csv", index=False)

def run_global_analysis():
    base_dir = Path("experiments/phase13_karen_generalization")
    input_dir = base_dir / "diarization/msdd_outputs/speaker_outputs"
    output_dir = base_dir / "embedding_analysis"
    setup_directories(output_dir, is_local=False)
    
    window_sizes = {0: 1.5, 1: 1.2, 2: 1.0, 3: 0.8, 4: 0.5}
    
    all_scale_metrics = {}
    pca_projections = {}
    umap_projections = {}
    times_dict = {}
    
    print("Starting global processing...")
    
    for scale in range(5):
        data = load_data(scale, input_dir)
        if data is None:
            continue
            
        emb_mat = data['embeddings']
        metadata = data['metadata']
        labels = data['labels']
        N = emb_mat.shape[0]
        
        times = [m['offset'] for m in metadata]
        times_dict[scale] = times
        
        print(f"Computing PCA for scale {scale}...")
        pca = PCA(n_components=2)
        pca_proj = pca.fit_transform(emb_mat)
        pca_projections[scale] = pca_proj
        
        print(f"Computing UMAP for scale {scale}...")
        reducer = umap.UMAP(n_components=2, random_state=42)
        umap_proj = reducer.fit_transform(emb_mat)
        umap_projections[scale] = umap_proj
        
        win_size = window_sizes.get(scale, "unknown")
        title_base = f"Scale {scale} (Window: {win_size}s, N={N})"
        
        plot_projection(pca_proj, times, f"PCA - {title_base}", "PCA Component 1", "PCA Component 2", output_dir / f"pca/scale{scale}_pca.png")
        plot_projection(umap_proj, times, f"UMAP - {title_base}", "UMAP Dimension 1", "UMAP Dimension 2", output_dir / f"umap/scale{scale}_umap.png")
        plot_trajectory(pca_proj, times, f"PCA Trajectory - {title_base}", "PCA Component 1", "PCA Component 2", output_dir / f"trajectories/scale{scale}_pca_trajectory.png")
        plot_trajectory(umap_proj, times, f"UMAP Trajectory - {title_base}", "UMAP Dimension 1", "UMAP Dimension 2", output_dir / f"trajectories/scale{scale}_umap_trajectory.png")
        
        df = pd.DataFrame({
            'audio_filepath': [m['audio_filepath'] for m in metadata],
            'offset': times,
            'duration': [m['duration'] for m in metadata],
            'pca_1': pca_proj[:, 0],
            'pca_2': pca_proj[:, 1],
            'umap_1': umap_proj[:, 0],
            'umap_2': umap_proj[:, 1]
        })
        if labels is not None: df['label'] = labels
        df.to_csv(output_dir / f"pca/scale{scale}_projections.csv", index=False)
        
        print(f"Computing metrics for scale {scale}...")
        metrics = compute_metrics(emb_mat, pca, labels)
        all_scale_metrics[f"scale_{scale}"] = metrics
        
        with open(output_dir / f"metrics/scale{scale}_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=4)

    print("Generating combined comparison figures...")
    scales_found = list(pca_projections.keys())
    if len(scales_found) > 0:
        fig, axes = plt.subplots(2, len(scales_found), figsize=(5 * len(scales_found), 10))
        if len(scales_found) == 1:
            axes = np.array([[axes[0]], [axes[1]]])
            
        for i, scale in enumerate(scales_found):
            win_size = window_sizes.get(scale, "unknown")
            times = times_dict[scale]
            
            ax = axes[0, i]
            ax.scatter(pca_projections[scale][:, 0], pca_projections[scale][:, 1], c=times, cmap='viridis', s=10, alpha=0.7)
            ax.set_title(f"Scale {scale} PCA ({win_size}s window)")
            ax.set_xlabel("PCA 1")
            ax.set_ylabel("PCA 2")
            
            ax = axes[1, i]
            ax.scatter(umap_projections[scale][:, 0], umap_projections[scale][:, 1], c=times, cmap='viridis', s=10, alpha=0.7)
            ax.set_title(f"Scale {scale} UMAP ({win_size}s window)")
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            
        plt.tight_layout()
        plt.savefig(output_dir / "combined/all_scales_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    print("Generating markdown report...")
    report_lines = ["# Multiscale TitaNet Embedding Analysis Report", ""]
    for scale, metrics in all_scale_metrics.items():
        win = window_sizes.get(int(scale.split('_')[1]), "unknown")
        report_lines.extend([
            f"## {scale.capitalize()} (Window Size: {win}s)",
            f"- **Number of Embeddings**: {metrics['num_embeddings']}",
            f"- **Dimensionality**: {metrics['dimensionality']}",
            f"- **PCA Explained Variance (2D)**: {metrics['pca_explained_variance_2d']:.4f}",
            f"- **Distribution Shape**: Evaluated as **{metrics['distribution']}**\n"
        ])
    with open(output_dir / "reports/analysis_report.md", 'w') as f:
        f.write("\n".join(report_lines))

def main():
    parser = argparse.ArgumentParser(description="Multiscale Embedding Analysis Pipeline")
    parser.add_argument("--local", action="store_true", help="Run local trajectory analysis")
    parser.add_argument("--scale", type=int, default=1, help="Scale to analyze (default 1)")
    parser.add_argument("--window_size", type=float, default=5.0, help="Local window size in seconds")
    parser.add_argument("--stride", type=float, default=2.5, help="Sliding window stride in seconds")
    parser.add_argument("--start", type=float, default=None, help="Start time in seconds")
    parser.add_argument("--end", type=float, default=None, help="End time in seconds")
    parser.add_argument("--auto_slide", action="store_true", help="Slide across the entire audio")
    
    args = parser.parse_args()
    
    if args.local:
        run_local_analysis(args)
    else:
        run_global_analysis()

if __name__ == "__main__":
    main()
