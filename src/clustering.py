"""
Clustering functions. Default uses CPU hdbscan.
If you want GPU in notebooks: replace with cuML/HDBSCAN there.
"""
import joblib
import numpy as np

try:
    import hdbscan as HDBSCAN_LIB
except Exception:
    HDBSCAN_LIB = None

def fit_hdbscan(X: np.ndarray, min_cluster_size: int = 1000, min_samples: int = 5, save_path: str = None) -> np.ndarray:
    """
    Fit HDBSCAN on X. Returns labels (ndarray).
    """
    if HDBSCAN_LIB is None:
        raise ImportError("hdbscan not installed. pip install hdbscan")
    clusterer = HDBSCAN_LIB.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    labels = clusterer.fit_predict(X)
    if save_path:
        joblib.dump(clusterer, save_path)
    return labels

def remove_noise_by_cluster_size(X: np.ndarray, labels: np.ndarray, min_cluster_size: int = 10):
    """
    Remove points with label == -1 and clusters with size < min_cluster_size.
    Returns filtered X and labels.
    """
    import pandas as pd
    df = pd.DataFrame({"idx": np.arange(len(labels)), "label": labels})
    # remove -1
    df = df[df["label"] != -1]
    counts = df["label"].value_counts()
    good_labels = counts[counts >= min_cluster_size].index.tolist()
    df = df[df["label"].isin(good_labels)]
    keep_idx = df["idx"].values.astype(int)
    return X[keep_idx], labels[keep_idx]
