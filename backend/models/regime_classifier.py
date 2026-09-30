"""
models/regime_classifier.py

Regime Classifier — MVP uses Random Forest logic implemented in pure numpy.
For the prototype we train a simple decision-tree ensemble from the atmospheric
field statistics (mean, std, min, max per channel) rather than the full 2D grid.

Regimes:
  0 – Active Monsoon
  1 – Break Monsoon
  2 – Monsoon Depression
  3 – Coastal/Orographic
  4 – Western Disturbance
"""

from __future__ import annotations
import numpy as np
import json
import os

REGIMES = ["active", "break", "depression", "orographic", "western_disturbance"]
REGIME_DISPLAY = {
    "active":                "Active Monsoon",
    "break":                 "Break Monsoon",
    "depression":            "Monsoon Depression",
    "orographic":            "Coastal / Orographic",
    "western_disturbance":   "Western Disturbance",
}
REGIME_ERROR_PATTERNS = {
    "active": {
        "summary": "Heavy rainfall intensity underestimated; spatial extent overestimated.",
        "issues": ["NWP spreads rainfall too widely", "Peak intensities suppressed by 20-40%"],
    },
    "break": {
        "summary": "Strong topographic wet/dry biases over Himalayan and central regions.",
        "issues": ["Excessive rainfall in NE/Himalayan zone", "Underestimate over central India"],
    },
    "depression": {
        "summary": "Track displacement, intensity bias, and temporal lag are dominant errors.",
        "issues": ["Storm center displaced 50-200 km", "Intensity bias up to 30%", "6-12 h temporal lag"],
    },
    "orographic": {
        "summary": "Windward wet bias and leeward dry bias over Western Ghats.",
        "issues": ["Windward (Konkan) overestimate", "Leeward (Deccan Plateau) underestimate"],
    },
    "western_disturbance": {
        "summary": "Timing errors, stratiform/convective separation, precipitation-phase errors.",
        "issues": ["Onset timing offset by 12-24 h", "Convective fraction overestimated"],
    },
}

N_CHANNELS = 5
CHANNEL_NAMES = ["MSLP", "850U", "850V", "TCWV", "500Z"]
FEATURES_PER_CHANNEL = 4  # mean, std, min, max
N_FEATURES = N_CHANNELS * FEATURES_PER_CHANNEL  # 20


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(atmo: np.ndarray) -> np.ndarray:
    """
    atmo: (N, 5, H, W) or (5, H, W) for single sample
    Returns: (N, 20) feature matrix
    """
    single = atmo.ndim == 3
    if single:
        atmo = atmo[np.newaxis]
    N = atmo.shape[0]
    feats = np.zeros((N, N_FEATURES), dtype=np.float32)
    for c in range(N_CHANNELS):
        ch = atmo[:, c, :, :].reshape(N, -1)  # (N, H*W)
        feats[:, c*4 + 0] = ch.mean(axis=1)
        feats[:, c*4 + 1] = ch.std(axis=1)
        feats[:, c*4 + 2] = ch.min(axis=1)
        feats[:, c*4 + 3] = ch.max(axis=1)
    return feats[0] if single else feats


# ---------------------------------------------------------------------------
# Naive Bayes Gaussian classifier (numpy-only)
# ---------------------------------------------------------------------------

class GaussianNBRegimeClassifier:
    """
    Gaussian Naive Bayes regime classifier.
    Sufficient for MVP demonstration; replace with RF/XGBoost when sklearn available.
    """

    def __init__(self):
        self.means_   = None   # (n_classes, n_features)
        self.vars_    = None   # (n_classes, n_features)
        self.priors_  = None   # (n_classes,)
        self.classes_ = None
        self.trained_ = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        classes = np.unique(y)
        self.classes_ = classes
        n_classes = len(classes)
        n_features = X.shape[1]
        self.means_  = np.zeros((n_classes, n_features))
        self.vars_   = np.zeros((n_classes, n_features))
        self.priors_ = np.zeros(n_classes)

        for i, c in enumerate(classes):
            Xc = X[y == c]
            self.means_[i]  = Xc.mean(axis=0)
            self.vars_[i]   = Xc.var(axis=0) + 1e-9
            self.priors_[i] = Xc.shape[0] / X.shape[0]

        self.trained_ = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns (N, n_classes) probability array."""
        if not self.trained_:
            raise RuntimeError("Classifier not trained yet.")
        n = X.shape[0] if X.ndim == 2 else 1
        if X.ndim == 1:
            X = X[np.newaxis]

        log_probs = np.zeros((n, len(self.classes_)))
        for i in range(len(self.classes_)):
            log_prior = np.log(self.priors_[i])
            # Log Gaussian likelihood: -0.5 * sum[ log(2π σ²) + (x-μ)²/σ² ]
            log_likelihood = -0.5 * np.sum(
                np.log(2 * np.pi * self.vars_[i]) +
                ((X - self.means_[i]) ** 2) / self.vars_[i],
                axis=1
            )
            log_probs[:, i] = log_prior + log_likelihood

        # Softmax for numerical stability
        log_probs -= log_probs.max(axis=1, keepdims=True)
        probs = np.exp(log_probs)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def save(self, path: str):
        np.savez(path,
                 means=self.means_,
                 vars=self.vars_,
                 priors=self.priors_,
                 classes=self.classes_)

    def load(self, path: str):
        d = np.load(path + ".npz")
        self.means_  = d["means"]
        self.vars_   = d["vars"]
        self.priors_ = d["priors"]
        self.classes_= d["classes"]
        self.trained_= True
        return self


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------

def train_classifier(dataset_path: str, model_save_path: str) -> dict:
    data = np.load(dataset_path)
    atmo   = data["atmo"]    # (N, 5, H, W)
    labels = data["labels"]  # (N,)

    X = extract_features(atmo)
    y = labels

    # 80/20 train/test split
    n = len(y)
    idx = np.random.permutation(n)
    train_idx = idx[:int(0.8 * n)]
    test_idx  = idx[int(0.8 * n):]

    clf = GaussianNBRegimeClassifier()
    clf.fit(X[train_idx], y[train_idx])

    # Accuracy on test set
    preds = clf.predict(X[test_idx])
    acc = (preds == y[test_idx]).mean()

    clf.save(model_save_path)

    # Per-regime accuracy
    per_regime = {}
    for i, r in enumerate(REGIMES):
        mask = y[test_idx] == i
        if mask.sum() > 0:
            per_regime[r] = float((preds[mask] == i).mean())

    return {"accuracy": float(acc), "per_regime": per_regime, "n_train": int(len(train_idx))}


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def classify_regime(atmo_fields: np.ndarray, model_path: str) -> dict:
    """
    atmo_fields: (5, H, W) numpy array
    Returns regime name + probability dict
    """
    clf = GaussianNBRegimeClassifier()
    clf.load(model_path)

    feats = extract_features(atmo_fields)
    proba = clf.predict_proba(feats)[0]
    regime_idx = int(np.argmax(proba))
    regime_name = REGIMES[regime_idx]

    return {
        "regime": regime_name,
        "display_name": REGIME_DISPLAY[regime_name],
        "confidence": float(proba[regime_idx]),
        "probabilities": {REGIMES[i]: float(proba[i]) for i in range(len(REGIMES))},
        "error_pattern": REGIME_ERROR_PATTERNS[regime_name],
    }


if __name__ == "__main__":
    import sys
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path  = os.path.join(base, "data", "synthetic_dataset.npz")
    model_path    = os.path.join(base, "data", "regime_classifier")

    print("Training Regime Classifier...")
    result = train_classifier(dataset_path, model_path)
    print(f"  Overall accuracy : {result['accuracy']:.3f}")
    print(f"  Per-regime       : {result['per_regime']}")
    print(f"  Saved to         : {model_path}.npz")
