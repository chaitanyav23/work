import os
import json
import pickle
import glob
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

def setup_directories(base_dir):
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
                print(f"WARNING: scale{scale_num} audio {audio_id} mismatch: {len(metas)} metadata, {num_emb} embeddings")
                min_len = min(len(metas), num_emb)
                all_embeddings.append(emb_tensor[:min_len].cpu().numpy())
                all_metadata.extend(metas[:min_len])
            else:
                all_embeddings.append(emb_tensor.cpu().numpy())
                all_metadata.extend(metas)
        else:
            print(f"WARNING: Audio ID {audio_id} found in pickle but not in JSON.")
            
    if not all_embeddings:
        print(f"No embeddings matched for scale {scale_num}.")
        return None
        
    embeddings_mat = np.vstack(all_embeddings)
    
    # Check for NaNs and variance
    if np.isnan(embeddings_mat).any():
        print(f"WARNING: NaN found in embeddings for scale {scale_num}.")
    variance = np.var(embeddings_mat, axis=0)
    if np.any(variance == 0):
        print(f"WARNING: Zero variance dimensions found in scale {scale_num}.")
    
    # Detect duplicates
    unique_embs = np.unique(embeddings_mat, axis=0)
    if len(unique_embs) != len(embeddings_mat):
        print(f"WARNING: Duplicate embeddings found in scale {scale_num}. Total: {len(embeddings_mat)}, Unique: {len(unique_embs)}")
        
    labels = None
    if label_path.exists():
        print(f"Loading labels from {label_path.name}...")
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
    # Line connecting points
    plt.plot(proj[:, 0], proj[:, 1], color='gray', alpha=0.3, linewidth=1)
    # Points on top
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
    
    # Cosine distance
    dist_mat = pairwise_distances(emb_mat, metric='cosine')
    # Upper triangle excluding diagonal
    triu_indices = np.triu_indices(N, k=1)
    dists = dist_mat[triu_indices]
    
    # Nearest Neighbors
    if N > 1:
        nn = NearestNeighbors(n_neighbors=min(6, N), metric='cosine')
        nn.fit(emb_mat)
        distances, _ = nn.kneighbors(emb_mat)
        # distances[:, 1:] because the 0th neighbor is the point itself (dist=0)
        mean_intra_nn = float(np.mean(distances[:, 1:]))
    else:
        mean_intra_nn = 0.0
        
    sil_score = None
    if labels is not None:
        valid_idx = labels != 'UNKNOWN'
        if np.sum(valid_idx) > 1 and len(np.unique(labels[valid_idx])) > 1:
            sil_score = float(silhouette_score(emb_mat[valid_idx], labels[valid_idx], metric='cosine'))

    # Diffuse vs Compact Check (Heuristic)
    # If the mean neighbor distance is close to the global mean distance, it's diffuse.
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

def main():
    base_dir = Path("experiments/phase13_karen_generalization")
    input_dir = base_dir / "diarization/msdd_outputs/speaker_outputs"
    output_dir = base_dir / "embedding_analysis"
    setup_directories(output_dir)
    
    window_sizes = {0: 1.5, 1: 1.2, 2: 1.0, 3: 0.8, 4: 0.5}
    
    all_scale_metrics = {}
    pca_projections = {}
    umap_projections = {}
    times_dict = {}
    
    print("Starting processing...")
    
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
        
        # PCA
        print(f"Computing PCA for scale {scale}...")
        pca = PCA(n_components=2)
        pca_proj = pca.fit_transform(emb_mat)
        pca_projections[scale] = pca_proj
        
        # UMAP
        print(f"Computing UMAP for scale {scale}...")
        reducer = umap.UMAP(n_components=2, random_state=42)
        umap_proj = reducer.fit_transform(emb_mat)
        umap_projections[scale] = umap_proj
        
        # Plots
        win_size = window_sizes.get(scale, "unknown")
        title_base = f"Scale {scale} (Window: {win_size}s, N={N})"
        
        plot_projection(
            pca_proj, times, 
            f"PCA - {title_base}", "PCA Component 1", "PCA Component 2", 
            output_dir / f"pca/scale{scale}_pca.png"
        )
        
        plot_projection(
            umap_proj, times, 
            f"UMAP - {title_base}", "UMAP Dimension 1", "UMAP Dimension 2", 
            output_dir / f"umap/scale{scale}_umap.png"
        )
        
        plot_trajectory(
            pca_proj, times, 
            f"PCA Trajectory - {title_base}", "PCA Component 1", "PCA Component 2", 
            output_dir / f"trajectories/scale{scale}_pca_trajectory.png"
        )
        
        plot_trajectory(
            umap_proj, times, 
            f"UMAP Trajectory - {title_base}", "UMAP Dimension 1", "UMAP Dimension 2", 
            output_dir / f"trajectories/scale{scale}_umap_trajectory.png"
        )
        
        # Save CSVs
        df = pd.DataFrame({
            'audio_filepath': [m['audio_filepath'] for m in metadata],
            'offset': times,
            'duration': [m['duration'] for m in metadata],
            'pca_1': pca_proj[:, 0],
            'pca_2': pca_proj[:, 1],
            'umap_1': umap_proj[:, 0],
            'umap_2': umap_proj[:, 1]
        })
        if labels is not None:
            df['label'] = labels
        df.to_csv(output_dir / f"pca/scale{scale}_projections.csv", index=False)
        
        # Metrics
        print(f"Computing metrics for scale {scale}...")
        metrics = compute_metrics(emb_mat, pca, labels)
        all_scale_metrics[f"scale_{scale}"] = metrics
        
        with open(output_dir / f"metrics/scale{scale}_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=4)

    # Combined figure
    print("Generating combined comparison figures...")
    scales_found = list(pca_projections.keys())
    if len(scales_found) > 0:
        fig, axes = plt.subplots(2, len(scales_found), figsize=(5 * len(scales_found), 10))
        # Handle 1D axes case if only 1 scale is found
        if len(scales_found) == 1:
            axes = np.array([[axes[0]], [axes[1]]])
            
        for i, scale in enumerate(scales_found):
            win_size = window_sizes.get(scale, "unknown")
            times = times_dict[scale]
            
            # PCA
            ax = axes[0, i]
            sc = ax.scatter(pca_projections[scale][:, 0], pca_projections[scale][:, 1], c=times, cmap='viridis', s=10, alpha=0.7)
            ax.set_title(f"Scale {scale} PCA ({win_size}s window)")
            ax.set_xlabel("PCA 1")
            ax.set_ylabel("PCA 2")
            
            # UMAP
            ax = axes[1, i]
            ax.scatter(umap_projections[scale][:, 0], umap_projections[scale][:, 1], c=times, cmap='viridis', s=10, alpha=0.7)
            ax.set_title(f"Scale {scale} UMAP ({win_size}s window)")
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            
        plt.tight_layout()
        plt.savefig(output_dir / "combined/all_scales_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    # Generate Markdown Report
    print("Generating markdown report...")
    report_lines = [
        "# Multiscale TitaNet Embedding Analysis Report",
        "",
        "This report summarizes the geometric, temporal, and clustering properties of NeMo MSDD TitaNet speaker embeddings across multiple scales.",
        ""
    ]
    
    for scale, metrics in all_scale_metrics.items():
        win = window_sizes.get(int(scale.split('_')[1]), "unknown")
        report_lines.extend([
            f"## {scale.capitalize()} (Window Size: {win}s)",
            f"- **Number of Embeddings**: {metrics['num_embeddings']}",
            f"- **Dimensionality**: {metrics['dimensionality']}",
            f"- **PCA Explained Variance (2D)**: {metrics['pca_explained_variance_2d']:.4f}",
            f"- **Mean Pairwise Cosine Distance**: {metrics['pairwise_cosine_distance']['mean']:.4f} ± {metrics['pairwise_cosine_distance']['std']:.4f}",
            f"- **Mean Intra-Neighbor Distance (k=5)**: {metrics['mean_intra_neighbor_distance']:.4f}",
            f"- **Distribution Shape**: Evaluated as **{metrics['distribution']}**",
        ])
        if metrics.get('silhouette_score') is not None:
            report_lines.append(f"- **Silhouette Score (from clustering labels)**: {metrics['silhouette_score']:.4f}")
        else:
            report_lines.append("- **Silhouette Score**: N/A (No clustering labels available)")
        report_lines.append("")
        
    report_lines.extend([
        "## General Observations",
        "- **Multiscale Behavior**: As window size decreases, embeddings typically become noisier and more diffuse, affecting speaker separability.",
        "- **Temporal Trajectories**: Trajectory plots show how embeddings transition smoothly or abruptly during speaker turns.",
        "- **BWC Robustness**: Body-worn camera audio presents significant noise challenges, potentially reflected in low explained variance or highly diffuse distributions in shorter window scales."
    ])
    
    with open(output_dir / "reports/analysis_report.md", 'w') as f:
        f.write("\n".join(report_lines))

    print("Done. Analysis outputs saved to", output_dir)

if __name__ == "__main__":
    main()
