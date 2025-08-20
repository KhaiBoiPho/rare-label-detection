"""
Global configuration for the pipeline.
Adjust paths / hyperparams here.
"""
from pathlib import Path

# Paths
ROOT = Path(".")
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
IFOREST_DIR = MODELS_DIR / "iforest_clusters"
UMAP_MODEL_PATH = MODELS_DIR / "umap_model.pkl"
HDBSCAN_MODEL_PATH = MODELS_DIR / "hdbscan_model.pkl"
LABELS_PATH = MODELS_DIR / "cluster_labels.pkl"

# Embedding
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_USE_MOCK = False      # set True for tests to use TF-IDF mock
EMBEDDING_MOCK_FEATURES = 256

# UMAP params
UMAP_N_COMPONENTS = 64
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1

# HDBSCAN params
HDBSCAN_MIN_CLUSTER_SIZE = 1000
HDBSCAN_MIN_SAMPLES = 5

# IsolationForest params
IFOREST_N_ESTIMATORS = 100
IFOREST_CONTAMINATION = 0.05
IFOREST_RANDOM_STATE = 42

# Data columns
TEXT_COL = "text"
LABEL_COL = "label"
