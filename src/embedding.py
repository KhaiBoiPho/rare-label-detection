from typing import List, Optional
import numpy as np

# If you want to avoid heavy model in tests, use mock TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

class Embedder:
    """
    Wrapper for embedding. Default uses sentence-transformers.
    If use_mock=True, uses TF-IDF vectorizer for quick tests.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_mock: bool = False, mock_features: int = 256):
        self.use_mock = use_mock
        if self.use_mock:
            self.vectorizer = TfidfVectorizer(max_features=mock_features)
            self.fitted = False
        else:
            # lazy import so tests that set use_mock don't need heavy deps
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)

    def fit(self, texts: List[str]) -> None:
        if self.use_mock:
            self.vectorizer.fit(texts)
            self.fitted = True

    def encode(self, texts: List[str], show_progress_bar: bool = False) -> np.ndarray:
        if self.use_mock:
            if not self.fitted:
                self.fit(texts)
            mat = self.vectorizer.transform(texts)
            return mat.toarray().astype(float)
        else:
            return np.array(self.model.encode(texts, show_progress_bar=show_progress_bar))
