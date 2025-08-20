import os
import joblib
from typing import Dict, Any
import numpy as np
from sklearn.ensemble import IsolationForest

def train_iforest_per_cluster(X: np.ndarray, labels: np.ndarray, save_dir: str,
                              n_estimators: int = 100,
                              contamination: float = 0.05,
                              random_state: int = 42) -> Dict[int, Any]:
    """
    Train IsolationForest per cluster (labels != -1) and save models as joblib.
    Returns dict cluster_id -> model
    """
    os.makedirs(save_dir, exist_ok=True)
    cluster_ids = sorted(set(labels) - {-1})
    models = {}
    for cid in cluster_ids:
        Xc = X[labels == cid]
        if len(Xc) < 10:
            # skip clusters too small to train reliably
            continue
        clf = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=random_state)
        clf.fit(Xc)
        p = os.path.join(save_dir, f"iforest_cluster_{cid}.pkl")
        joblib.dump(clf, p)
        models[cid] = clf
    return models

def load_iforest_models(dirpath: str):
    """
    Load all iforest models from dir. Returns dict cid->model.
    """
    models = {}
    if not os.path.exists(dirpath):
        return models
    for fname in sorted(os.listdir(dirpath)):
        if fname.endswith(".pkl") and fname.startswith("iforest_cluster_"):
            cid = int(fname.split("_")[-1].split(".")[0])
            models[cid] = joblib.load(os.path.join(dirpath, fname))
    return models

def predict_rare_sequential(X_new: np.ndarray, models: dict):
    """
    Sequential filtering:
    - Start with all indices
    - For each model in models (iteration order of keys):
      keep only those flagged as outlier (-1) and pass them to next model
    - Return array of indices (relative to X_new) that remain after all models
    """
    if len(models) == 0:
        # no models -> everything is rare (or nothing trained)
        return np.arange(len(X_new), dtype=int)

    remaining = np.arange(len(X_new), dtype=int)
    for cid in sorted(models.keys()):
        clf = models[cid]
        if remaining.size == 0:
            break
        preds = clf.predict(X_new[remaining])  # 1 = inlier, -1 = outlier
        # keep outliers
        remaining = remaining[preds == -1]
    return remaining
