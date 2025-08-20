# tests/test_detection.py
import numpy as np
from sklearn.ensemble import IsolationForest
from src.iforest_detector import predict_rare_sequential

def test_predict_rare_sequential():
    rng = np.random.RandomState(0)
    X = rng.rand(50, 8)
    # build two iforest models trained on X (toy)
    clf1 = IsolationForest(contamination=0.2, random_state=0).fit(X)
    clf2 = IsolationForest(contamination=0.2, random_state=1).fit(X)
    models = {0: clf1, 1: clf2}
    rare_idx = predict_rare_sequential(X, models)
    assert hasattr(rare_idx, "__iter__")
    assert all(isinstance(i, (int, np.integer)) for i in rare_idx)