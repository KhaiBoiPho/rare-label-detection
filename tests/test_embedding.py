# tests/test_embedding.py
import numpy as np
from src.embedding import Embedder

def test_embedding_mock():
    texts = ["hello world", "this is a test"]
    emb = Embedder(use_mock=True, mock_features=64)
    emb.fit(texts)
    vectors = emb.encode(texts)
    assert isinstance(vectors, np.ndarray)
    assert vectors.shape[0] == len(texts)
    assert vectors.shape[1] <= 64