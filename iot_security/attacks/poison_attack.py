"""
Poison Attack
"""
import numpy as np
from sklearn.ensemble import IsolationForest

def label_flip_attack(X: np.ndarray, y: np.ndarray, source_class: int, target_class: int, poison_rate: float = 0.2, seed: int = 42):
    np.random.seed(seed)
    y_poisoned = y.copy()
    source_idx = np.where(y == source_class)[0]
    n_poison = int(len(source_idx) * poison_rate)
    chosen = np.random.choice(source_idx, n_poison, replace=False)
    y_poisoned[chosen] = target_class
    poison_mask = np.zeros(len(y), dtype=bool)
    poison_mask[chosen] = True
    print(f"[POISON] Label Flip: {n_poison} samples flipped")
    return X, y_poisoned, poison_mask

def detect_poison_isolation_forest(X: np.ndarray, contamination: float = 0.1):
    clf = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    preds = clf.fit_predict(X)
    suspicious = preds == -1
    return suspicious
