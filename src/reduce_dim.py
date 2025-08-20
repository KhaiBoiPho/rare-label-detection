"""
UMAP reduction utilities.
This module uses CPU `umap-learn` by default but attempts to detect cuML if you replace calls.
You can directly swap to cuml.UMAP in your notebooks for GPU acceleration.
"""
import joblib
import numpy as np

try:
    # CPU UMAP
    from umap import UMAP as UMAP_CPU
except Exception:
    UMAP_CPU = None

# Optional: if cuML present and you want GPU, you can update to use cuml. Here we keep CPU by default.
def fit_umap(X: np.ndarray, path: str, n_components: int = 64, n_neighbors: int = 15, min_dist: float = 0.1):
    """
    Fit UMAP on X and save the reducer.
    Returns X_reduced (ndarray).
    """
    if UMAP_CPU is None:
        raise ImportError("umap-learn is not installed. pip install umap-learn")
    reducer = UMAP_CPU(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    X_reduced = reducer.fit_transform(X)
    joblib.dump(reducer, path)
    return X_reduced

def load_umap(path: str):
    """
    Load a saved UMAP reducer (joblib).
    """
    return joblib.load(path)

def transform_umap(X: np.ndarray, path: str):
    reducer = load_umap(path)
    return reducer.transform(X)
