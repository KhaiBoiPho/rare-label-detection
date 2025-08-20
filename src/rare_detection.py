from typing import List, Tuple
import numpy as np
import os
from .embedding import Embedder
from .reduce_dim import transform_umap, load_umap
from .iforest_detector import load_iforest_models, predict_rare_sequential

def detect_rare_emails(texts: List[str],
                       umap_model_path: str,
                       iforest_dir: str,
                       embed_model_name: str,
                       use_mock: bool = False) -> np.ndarray:
    """
    Given list of texts, return indices (ints) of rare candidates after sequential filtering.
    """
    embedder = Embedder(model_name=embed_model_name, use_mock=use_mock)
    embeddings = embedder.encode(texts)

    # reduce
    X_reduced = transform_umap(embeddings, umap_model_path)

    # load models
    models = load_iforest_models(iforest_dir)

    rare_idx = predict_rare_sequential(X_reduced, models)
    return rare_idx