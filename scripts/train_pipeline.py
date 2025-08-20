"""
Train full pipeline:
- load dataset
- filter common labels
- embed
- fit UMAP
- cluster HDBSCAN
- remove noise / small clusters
- train IsolationForest per cluster
Saves: models/umap_model.pkl, models/hdbscan_model.pkl, models/iforest_clusters/*
"""
import os
import joblib
import numpy as np
import pandas as pd

from src.config import (DATA_DIR, MODELS_DIR, UMAP_MODEL_PATH, HDBSCAN_MODEL_PATH,
                        IFOREST_DIR, EMBEDDING_MODEL, EMBEDDING_USE_MOCK,
                        UMAP_N_COMPONENTS, UMAP_N_NEIGHBORS, UMAP_MIN_DIST,
                        HDBSCAN_MIN_CLUSTER_SIZE, HDBSCAN_MIN_SAMPLES, TEXT_COL, LABEL_COL)
from src.data_loader import load_csv, filter_common_labels
from src.embedding import Embedder
from src.reduce_dim import fit_umap
from src.clustering import fit_hdbscan, remove_noise_by_cluster_size
from src.iforest_detector import train_iforest_per_cluster

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    # Expect user to put a dataset at data/raw.csv (or adjust path)
    csv_path = DATA_DIR / "abc.csv"
    if not csv_path.exists():
        # fallback: try data/raw.csv or user must provide data
        csv_path = DATA_DIR / "raw.csv"
    if not csv_path.exists():
        raise FileNotFoundError("Please provide dataset at data/abc.csv or data/raw.csv with columns 'text' and 'label'.")

    df = load_csv(str(csv_path))
    if TEXT_COL not in df.columns or LABEL_COL not in df.columns:
        raise ValueError(f"Dataset must contain columns '{TEXT_COL}' and '{LABEL_COL}'")

    # choose threshold: keep labels >= min_count (example: labels >= 500 or configurable)
    min_count = max(1, int(len(df) * 0.01))  # example: labels that occupy at least 1% of data
    print(f"Filtering to common labels with min_count={min_count}")
    df_common = filter_common_labels(df, LABEL_COL, min_count)
    print(f"Using {len(df_common)} samples across {df_common[LABEL_COL].nunique()} labels")

    texts = df_common[TEXT_COL].astype(str).tolist()

    embedder = Embedder(model_name=EMBEDDING_MODEL, use_mock=EMBEDDING_USE_MOCK)
    if EMBEDDING_USE_MOCK:
        embedder.fit(texts)
    embeddings = embedder.encode(texts, show_progress_bar=True)
    print(f"Embeddings shape: {embeddings.shape}")

    # fit umap
    print("Fitting UMAP...")
    X_reduced = fit_umap(embeddings, str(UMAP_MODEL_PATH), n_components=UMAP_N_COMPONENTS,
                         n_neighbors=UMAP_N_NEIGHBORS, min_dist=UMAP_MIN_DIST)
    print(f"UMAP reduced shape: {X_reduced.shape}")

    # clustering
    print("Running HDBSCAN...")
    labels = fit_hdbscan(X_reduced, min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE, min_samples=HDBSCAN_MIN_SAMPLES, save_path=str(HDBSCAN_MODEL_PATH))
    print(f"Clusters found: {len(set(labels)) - (1 if -1 in labels else 0)} (excluding noise)")

    # remove noise & small clusters
    print("Filtering noise & small clusters...")
    X_filtered, labels_filtered = remove_noise_by_cluster_size(X_reduced, labels, min_cluster_size=10)
    print(f"Kept {len(labels_filtered)} samples after noise/small-cluster removal")

    # save labels mapping for traceability
    joblib.dump({"orig_idx": np.arange(len(labels))[labels != -1], "labels": labels_filtered}, str(MODELS_DIR / "cluster_index_labels.pkl"))

    # train IsolationForest per cluster
    print("Training IsolationForest per cluster...")
    models = train_iforest_per_cluster(X_filtered, labels_filtered, save_dir=str(IFOREST_DIR))
    print(f"Trained {len(models)} iforest models saved under {IFOREST_DIR}")

if __name__ == "__main__":
    main()