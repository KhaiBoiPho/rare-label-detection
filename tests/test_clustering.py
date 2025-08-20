# tests/test_clustering.py
import numpy as np
from src.reduce_dim import fit_umap, transform_umap
from src.clustering import fit_hdbscan, remove_noise_by_cluster_size
import tempfile
import os

def test_umap_hdbscan_pipeline():
    # random data small
    X = np.random.RandomState(42).rand(100, 128)
    tmpdir = tempfile.mkdtemp()
    umap_path = os.path.join(tmpdir, "umap_test.pkl")
    Xr = fit_umap(X, umap_path, n_components=8, n_neighbors=5, min_dist=0.1)
    assert Xr.shape[0] == 100
    labels = fit_hdbscan(Xr, min_cluster_size=5, min_samples=2)
    # remove noise and tiny clusters
    Xf, lf = remove_noise_by_cluster_size(Xr, labels, min_cluster_size=3)
    assert len(lf) <= 100