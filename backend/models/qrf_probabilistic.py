"""
models/qrf_probabilistic.py

Quantile Regression Forest (QRF) — numpy-only implementation.

A QRF extends a Random Forest by retaining the actual target values in each
leaf node (rather than averaging them). For a new query, the conditional
empirical CDF F̂(y|X) is formed from all leaf-node observations across all
trees, from which arbitrary quantiles and P(R>T) can be computed.

For the MVP prototype, we implement a simplified QRF using bootstrap
aggregation + leaf-node empirical distributions, entirely in numpy.

Verification functions evaluate at gridpoint level (flattened 2-D arrays),
NOT on domain means, to produce non-degenerate contingency metrics.
"""

from __future__ import annotations
import numpy as np

# ---------------------------------------------------------------------------
# IMD Official Rainfall Classification Thresholds (mm/day)
# ---------------------------------------------------------------------------
IMD_THRESHOLDS = {
    "light":        2.5,    # >= 2.5 mm
    "moderate":     15.6,   # >= 15.6 mm
    "heavy":        64.5,   # >= 64.5 mm — standard operational threshold
    "very_heavy":   115.6,  # >= 115.6 mm
    "extreme":      204.4,  # >= 204.4 mm
}

THRESHOLDS = {
    "light":      10.0,   # kept for backward compat (>=10 mm = "rainy day")
    "heavy":      64.5,
    "very_heavy": 115.5,
}

# FSS neighbourhood scales in grid cells (IMD 0.25 deg ~= 25 km per cell)
FSS_SCALES_KM    = [25, 50, 100, 200]         # km
FSS_SCALES_CELLS = [s // 25 for s in FSS_SCALES_KM]  # [1, 2, 4, 8]


# ---------------------------------------------------------------------------
# Single Decision Stump (depth-limited tree node)
# ---------------------------------------------------------------------------

class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "leaf_values")

    def __init__(self):
        self.feature     = None
        self.threshold   = None
        self.left        = None
        self.right       = None
        self.leaf_values = None   # np.ndarray when leaf


def _build_tree(X: np.ndarray, y: np.ndarray, max_depth: int, min_samples: int, rng):
    node = _Node()
    if max_depth == 0 or len(y) <= min_samples:
        node.leaf_values = y
        return node

    n_features = X.shape[1]
    n_try = max(1, int(np.sqrt(n_features)))
    feature_idxs = rng.choice(n_features, size=n_try, replace=False)

    best_score = np.inf
    best_feat  = None
    best_thr   = None

    for feat in feature_idxs:
        vals = X[:, feat]
        thrs = np.unique(vals)
        if len(thrs) < 2:
            continue
        candidate_thrs = rng.choice(thrs, size=min(10, len(thrs)), replace=False)
        for thr in candidate_thrs:
            mask_l = vals <= thr
            mask_r = ~mask_l
            if mask_l.sum() < 1 or mask_r.sum() < 1:
                continue
            score = (
                mask_l.sum() * y[mask_l].var() +
                mask_r.sum() * y[mask_r].var()
            )
            if score < best_score:
                best_score = score
                best_feat  = feat
                best_thr   = thr

    if best_feat is None:
        node.leaf_values = y
        return node

    node.feature    = best_feat
    node.threshold  = best_thr
    mask_l = X[:, best_feat] <= best_thr
    node.left  = _build_tree(X[mask_l],  y[mask_l],  max_depth-1, min_samples, rng)
    node.right = _build_tree(X[~mask_l], y[~mask_l], max_depth-1, min_samples, rng)
    return node


def _predict_leaf(node: _Node, x: np.ndarray) -> np.ndarray:
    if node.leaf_values is not None:
        return node.leaf_values
    if x[node.feature] <= node.threshold:
        return _predict_leaf(node.left, x)
    return _predict_leaf(node.right, x)


# ---------------------------------------------------------------------------
# QRF
# ---------------------------------------------------------------------------

class QuantileRegressionForest:
    """
    Lightweight numpy-only QRF.
    Uses bootstrap + shallow trees. For prototype demo accuracy.
    """

    def __init__(
        self,
        n_estimators: int = 50,
        max_depth: int = 6,
        min_samples_leaf: int = 5,
        seed: int = 42,
    ):
        self.n_estimators      = n_estimators
        self.max_depth         = max_depth
        self.min_samples_leaf  = min_samples_leaf
        self.rng               = np.random.default_rng(seed)
        self.trees_            = []
        self.fitted_           = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """X: (N, n_features), y: (N,) rainfall values"""
        N = X.shape[0]
        self.trees_ = []
        for _ in range(self.n_estimators):
            boot_idx = self.rng.integers(0, N, size=N)
            Xb, yb   = X[boot_idx], y[boot_idx]
            tree = _build_tree(Xb, yb, self.max_depth, self.min_samples_leaf, self.rng)
            self.trees_.append(tree)
        self.fitted_ = True
        return self

    def predict_quantiles(
        self, X: np.ndarray, quantiles: list[float]
    ) -> np.ndarray:
        """
        Returns (N, len(quantiles)) array of rainfall quantile predictions.
        quantiles: list of values in [0, 1] e.g. [0.1, 0.5, 0.9]
        """
        if not self.fitted_:
            raise RuntimeError("QRF not fitted.")
        N = X.shape[0]
        q_arr = np.array(quantiles)
        result = np.zeros((N, len(quantiles)))

        for i in range(N):
            leaf_vals = np.concatenate([
                _predict_leaf(tree, X[i]) for tree in self.trees_
            ])
            result[i] = np.percentile(leaf_vals, q_arr * 100)

        return result

    def predict_exceedance(
        self, X: np.ndarray, thresholds: list[float]
    ) -> np.ndarray:
        """
        Returns (N, len(thresholds)) array where each element is P(R > T).
        """
        if not self.fitted_:
            raise RuntimeError("QRF not fitted.")
        N = X.shape[0]
        T = len(thresholds)
        result = np.zeros((N, T))

        for i in range(N):
            leaf_vals = np.concatenate([
                _predict_leaf(tree, X[i]) for tree in self.trees_
            ])
            for j, thr in enumerate(thresholds):
                result[i, j] = float((leaf_vals > thr).mean())

        return result

    def predict_median(self, X: np.ndarray) -> np.ndarray:
        return self.predict_quantiles(X, [0.5])[:, 0]


# ---------------------------------------------------------------------------
# Verification metrics — GRIDPOINT-LEVEL (Phase 1 fix)
# ---------------------------------------------------------------------------

def _contingency(obs_flat: np.ndarray, fcst_flat: np.ndarray, threshold: float) -> dict:
    """
    Compute hits/misses/false_alarms/correct_negatives from flat 1-D arrays.
    """
    obs_yes  = obs_flat  >= threshold
    fcst_yes = fcst_flat >= threshold

    hits = int(( obs_yes &  fcst_yes).sum())
    miss = int(( obs_yes & ~fcst_yes).sum())
    fas  = int((~obs_yes &  fcst_yes).sum())
    cns  = int((~obs_yes & ~fcst_yes).sum())
    return {"hits": hits, "misses": miss, "false_alarms": fas, "correct_neg": cns}


def _scores_from_contingency(c: dict) -> dict:
    """Derive POD, FAR, CSI, ETS from contingency dict."""
    hits, miss, fas, cns = c["hits"], c["misses"], c["false_alarms"], c["correct_neg"]
    total  = hits + miss + fas + cns
    hits_r = (hits + miss) * (hits + fas) / max(total, 1)

    pod = hits / max(hits + miss, 1)
    far = fas  / max(hits + fas,  1)
    csi = hits / max(hits + miss + fas, 1)
    ets = (hits - hits_r) / max(hits + miss + fas - hits_r, 1e-9)

    return {
        "pod": round(pod, 4),
        "far": round(far, 4),
        "csi": round(csi, 4),
        "ets": round(ets, 4),
        **c,
    }


def compute_metrics(
    obs: np.ndarray,
    fcst: np.ndarray,
    threshold: float = 64.5,
) -> dict:
    """
    Compute gridpoint-level verification metrics.

    PHASE 1 FIX: obs and fcst are evaluated as FLAT arrays of individual
    gridpoints, NOT domain-mean scalars. This ensures that heavy-rain events
    at any single grid cell contribute correctly to contingency counts.

    obs, fcst: any shape — flattened internally.
    threshold: IMD rainfall category boundary (mm/day).
    """
    obs_f  = obs.ravel().astype(float)
    fcst_f = fcst.ravel().astype(float)

    rmse = float(np.sqrt(np.mean((fcst_f - obs_f) ** 2)))
    mae  = float(np.mean(np.abs(fcst_f - obs_f)))
    bias = float(np.mean(fcst_f - obs_f))

    c = _contingency(obs_f, fcst_f, threshold)
    scores = _scores_from_contingency(c)

    return {
        "rmse":         round(rmse, 4),
        "mae":          round(mae,  4),
        "bias":         round(bias, 4),
        **scores,
    }


def compute_metrics_multithreshold(
    obs: np.ndarray,
    fcst: np.ndarray,
    thresholds: dict | None = None,
) -> dict:
    """
    Compute verification metrics at ALL standard IMD rainfall thresholds.

    Returns: { "continuous": {...}, "heavy": {...}, "very_heavy": {...}, ... }

    obs, fcst: any shape — flattened internally (gridpoint evaluation).
    thresholds: dict of {label: mm_value}. Defaults to IMD_THRESHOLDS.
    """
    if thresholds is None:
        thresholds = IMD_THRESHOLDS

    obs_f  = obs.ravel().astype(float)
    fcst_f = fcst.ravel().astype(float)

    # Continuous stats (threshold-independent)
    rmse     = float(np.sqrt(np.mean((fcst_f - obs_f) ** 2)))
    mae      = float(np.mean(np.abs(fcst_f - obs_f)))
    bias     = float(np.mean(fcst_f - obs_f))
    corr_arr = np.corrcoef(obs_f, fcst_f)
    corr     = float(corr_arr[0, 1]) if len(obs_f) > 1 else 0.0
    wet_bias = float(np.mean(fcst_f > 0.1) - np.mean(obs_f > 0.1))

    result = {
        "continuous": {
            "rmse":     round(rmse,     4),
            "mae":      round(mae,      4),
            "bias":     round(bias,     4),
            "corr":     round(corr,     4),
            "wet_bias": round(wet_bias, 4),
            "n_points": int(len(obs_f)),
        }
    }

    for label, thr in thresholds.items():
        c = _contingency(obs_f, fcst_f, thr)
        s = _scores_from_contingency(c)
        obs_freq = float((obs_f >= thr).mean())
        result[label] = {
            "threshold_mm":  thr,
            "obs_frequency": round(obs_freq, 5),
            **s,
        }

    return result


# ---------------------------------------------------------------------------
# FSS — Fractions Skill Score (vectorised with SAT trick)
# ---------------------------------------------------------------------------

def _sliding_fractions(binary: np.ndarray, n: int) -> np.ndarray:
    """
    Compute local exceedance fractions via 2-D Summed Area Table (O(HW)).
    binary: (H, W) 0/1 array
    n: window side length in grid cells
    Returns: (H, W) fraction array in [0, 1]
    """
    if n <= 1:
        return binary.astype(float)

    H, W = binary.shape
    pad  = n // 2
    # Pad so every output pixel has a full n x n neighbourhood
    padded = np.pad(binary.astype(float), pad, mode="constant", constant_values=0)
    Ph, Pw = padded.shape  # (H + 2*pad, W + 2*pad)

    # Build (Ph+1) x (Pw+1) prefix-sum with a zero border for clean SAT indexing
    cs = np.zeros((Ph + 1, Pw + 1), dtype=np.float64)
    cs[1:, 1:] = np.cumsum(np.cumsum(padded, axis=0), axis=1)

    # For output pixel (r, c), the padded window spans rows [r, r+n), cols [c, c+n).
    # In 1-indexed cs: top=r, bot=r+n, left=c, right=c+n
    rows = np.arange(H, dtype=np.intp)
    cols = np.arange(W, dtype=np.intp)
    top   = rows[:, None]
    bot   = rows[:, None] + n
    left  = cols[None, :]
    right = cols[None, :] + n

    window_sums = (
          cs[bot, right]
        - cs[top, right]
        - cs[bot, left]
        + cs[top, left]
    )
    return (window_sums / (n * n)).astype(np.float64)



def compute_fss(
    obs: np.ndarray,
    fcst: np.ndarray,
    threshold: float,
    scale: int,
) -> float:
    """
    Fractions Skill Score at a given neighbourhood scale.

    obs, fcst: (H, W) 2-D arrays (mm/day)
    threshold: rainfall exceedance threshold (mm/day)
    scale: neighbourhood size in grid cells (square window)
    Returns FSS in [0, 1].

    Usefulness criterion:  FSS >= 0.5 + f_o/2
    """
    obs_bin  = (obs  >= threshold).astype(float)
    fcst_bin = (fcst >= threshold).astype(float)

    Po = _sliding_fractions(obs_bin,  scale)
    Pf = _sliding_fractions(fcst_bin, scale)

    mse     = float(np.mean((Pf - Po) ** 2))
    mse_ref = float(np.mean(Pf ** 2) + np.mean(Po ** 2))

    if mse_ref < 1e-10:
        return 1.0
    return float(np.clip(1.0 - mse / mse_ref, 0.0, 1.0))


def compute_fss_multiscale(
    obs_fields: np.ndarray,
    fcst_fields: np.ndarray,
    threshold: float,
    scales_cells: list[int] | None = None,
) -> dict:
    """
    Compute FSS at multiple neighbourhood scales over a set of daily fields.

    obs_fields, fcst_fields: (N, H, W)
    threshold: exceedance threshold in mm/day
    scales_cells: list of window sizes in grid cells. Defaults to FSS_SCALES_CELLS.

    Returns: {
        "25km": {"fss": 0.42, "useful": False, "f_o": 0.03, ...},
        "50km": {...},
        ...
    }
    """
    if scales_cells is None:
        scales_cells = FSS_SCALES_CELLS

    N = obs_fields.shape[0]
    f_o = float((obs_fields >= threshold).mean())
    usefulness_thr = 0.5 + f_o / 2.0

    result = {}
    for cells in scales_cells:
        km = cells * 25
        fss_vals = [
            compute_fss(obs_fields[i], fcst_fields[i], threshold, cells)
            for i in range(N)
        ]
        mean_fss = float(np.mean(fss_vals))
        result[f"{km}km"] = {
            "fss":                  round(mean_fss,       4),
            "useful":               bool(mean_fss >= usefulness_thr),
            "usefulness_threshold": round(usefulness_thr, 4),
            "f_o":                  round(f_o,            6),
            "scale_km":             km,
            "scale_cells":          cells,
        }

    return result


# ---------------------------------------------------------------------------
# CRPS & twCRPS
# ---------------------------------------------------------------------------

def compute_crps(obs: np.ndarray, ensemble: np.ndarray) -> float:
    """
    Mean CRPS from an ensemble forecast.
    obs: (N,)  observed values
    ensemble: (N, M)  ensemble members
    """
    N, M = ensemble.shape
    crps_vals = []
    for i in range(N):
        y   = obs[i]
        ens = np.sort(ensemble[i])
        crps_i = 0.0
        for m, e in enumerate(ens):
            F_e    = (m + 1) / M
            H_e    = float(e >= y)
            width  = ens[m + 1] - e if m < M - 1 else 0.0
            crps_i += (F_e - H_e) ** 2 * width
        crps_vals.append(crps_i)
    return float(np.mean(crps_vals))


def compute_twcrps(
    obs: np.ndarray,
    ensemble: np.ndarray,
    threshold: float = 64.5,
    alpha: float = 1.0,
) -> float:
    """
    Threshold-weighted CRPS (twCRPS) — emphasises skill in the heavy-rain tail.

    Weight function: w(y) = (y / threshold)^alpha * 1(y >= threshold)
    Concentrates verification credit on values above the operational threshold.

    obs: (N,) observed values
    ensemble: (N, M) ensemble members
    threshold: weighting threshold in mm/day (default: heavy rain = 64.5 mm)
    alpha: exponent (higher = more emphasis on extreme tail)
    Returns mean twCRPS (lower is better).
    """
    N, M = ensemble.shape
    twcrps_vals = []

    def weight(y: float) -> float:
        return ((y / threshold) ** alpha) if y >= threshold else 0.0

    for i in range(N):
        y   = obs[i]
        ens = np.sort(ensemble[i])
        w_y = weight(y)
        twc = 0.0
        for m, e in enumerate(ens):
            F_e   = (m + 1) / M
            H_e   = float(e >= y)
            w_e   = weight(e)
            w_avg = 0.5 * (w_y + w_e)
            width = ens[m + 1] - e if m < M - 1 else 0.0
            twc  += (F_e - H_e) ** 2 * w_avg * width
        twcrps_vals.append(twc)

    return float(np.mean(twcrps_vals))


def compute_reliability(
    obs: np.ndarray,
    prob_fcst: np.ndarray,
    threshold: float = 64.5,
    n_bins: int = 10,
) -> dict:
    """
    Reliability / Attributes diagram data for a probabilistic forecast.

    obs: (N,) observed rainfall values (mm/day)
    prob_fcst: (N,) forecast exceedance probabilities P(R > threshold)
    threshold: exceedance threshold (mm/day)
    n_bins: number of probability bins

    Returns dict for plotting a reliability diagram, including BSS.
    """
    obs_bin   = (obs >= threshold).astype(float)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_cen   = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    obs_freq   = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    for k in range(n_bins):
        mask = (prob_fcst >= bin_edges[k]) & (prob_fcst < bin_edges[k + 1])
        bin_counts[k] = int(mask.sum())
        if bin_counts[k] > 0:
            obs_freq[k] = float(obs_bin[mask].mean())

    clim_freq = float(obs_bin.mean())
    bs        = float(np.mean((prob_fcst - obs_bin) ** 2))
    bs_ref    = float(clim_freq * (1.0 - clim_freq))
    bss       = 1.0 - bs / max(bs_ref, 1e-9)

    return {
        "threshold_mm":      threshold,
        "bin_centers":       bin_cen.tolist(),
        "observed_freq":     obs_freq.tolist(),
        "bin_counts":        bin_counts.tolist(),
        "climatology":       round(clim_freq, 5),
        "brier_score":       round(bs, 5),
        "brier_skill_score": round(bss, 4),
    }


# ---------------------------------------------------------------------------
# Full multi-model verification report builder
# ---------------------------------------------------------------------------

def build_verification_report(
    obs_fields: np.ndarray,
    nwp_fields: np.ndarray,
    qdm_fields: np.ndarray,
    varsh_fields: np.ndarray,
    regime_labels: np.ndarray,
    regime_names: list[str],
    thresholds: dict | None = None,
    fss_threshold: float = 64.5,
    label: str = "val",
) -> dict:
    """
    Produce a complete, publication-grade verification report comparing
    Raw NWP, QDM-corrected, and Varsh-AI forecasts against IMD observations.

    All contingency metrics are at GRIDPOINT level (not domain means).

    obs_fields, nwp_fields, qdm_fields, varsh_fields: (N, H, W)
    regime_labels: (N,) integer regime indices
    regime_names:  list of regime name strings
    thresholds:    IMD threshold dict
    fss_threshold: threshold for FSS computation (mm/day)
    """
    if thresholds is None:
        thresholds = IMD_THRESHOLDS

    N, H, W = obs_fields.shape
    report = {
        "label":        label,
        "n_days":       N,
        "grid_shape":   [H, W],
        "n_gridpoints": N * H * W,
        "data_source":  "IMD Gridded Daily Rainfall 0.25 deg (2021-2025)",
    }

    # Overall multi-threshold metrics across all days + gridpoints
    report["overall"] = {}
    for model_key, fcst_fields in [
        ("raw_nwp", nwp_fields),
        ("qdm",     qdm_fields),
        ("varsh",   varsh_fields),
    ]:
        report["overall"][model_key] = compute_metrics_multithreshold(
            obs_fields, fcst_fields, thresholds
        )

    # Multi-scale FSS at key thresholds
    report["fss"] = {}
    for thr_label, thr_mm in [("moderate", 15.6), ("heavy", 64.5), ("very_heavy", 115.6)]:
        report["fss"][thr_label] = {}
        for model_key, fcst_fields in [
            ("raw_nwp", nwp_fields),
            ("qdm",     qdm_fields),
            ("varsh",   varsh_fields),
        ]:
            report["fss"][thr_label][model_key] = compute_fss_multiscale(
                obs_fields, fcst_fields, thr_mm
            )

    # Regime-stratified metrics
    report["by_regime"] = {}
    for ridx, rname in enumerate(regime_names):
        mask = regime_labels == ridx
        if mask.sum() < 5:
            report["by_regime"][rname] = {"n_days": int(mask.sum()), "skipped": True}
            continue

        obs_r = obs_fields[mask]
        entry = {"n_days": int(mask.sum()), "models": {}, "fss": {}}

        for model_key, fcst_fields in [
            ("raw_nwp", nwp_fields[mask]),
            ("qdm",     qdm_fields[mask]),
            ("varsh",   varsh_fields[mask]),
        ]:
            entry["models"][model_key] = compute_metrics_multithreshold(
                obs_r, fcst_fields, thresholds
            )
            entry["fss"][model_key] = compute_fss_multiscale(
                obs_r, fcst_fields, fss_threshold
            )

        report["by_regime"][rname] = entry

    return report


# ---------------------------------------------------------------------------
# Quick self-test (run as script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(7)
    N, H, W = 40, 30, 40
    obs_fields   = np.clip(rng.lognormal(2.0, 1.2, (N, H, W)), 0, None).astype(np.float32)
    nwp_fields   = (obs_fields * rng.uniform(0.60, 0.85, (N, H, W))).astype(np.float32)
    qdm_fields   = (obs_fields * rng.uniform(0.85, 1.05, (N, H, W))).astype(np.float32)
    varsh_fields = (obs_fields * rng.uniform(0.88, 1.08, (N, H, W))).astype(np.float32)
    labels = rng.integers(0, 5, N)

    print("=== compute_metrics_multithreshold ===")
    m = compute_metrics_multithreshold(obs_fields, varsh_fields)
    for k, v in m.items():
        if isinstance(v, dict):
            print(f"  [{k:12s}] CSI={v.get('csi','—')}, POD={v.get('pod','—')}, "
                  f"hits={v.get('hits','—')}")

    print("\n=== FSS multi-scale (moderate 15.6 mm) ===")
    fss = compute_fss_multiscale(obs_fields, nwp_fields, threshold=15.6)
    for scale, vals in fss.items():
        print(f"  {scale}: FSS={vals['fss']:.3f}, useful={vals['useful']}")

    print("\n=== Reliability diagram ===")
    prob_fcst = np.clip(rng.uniform(0, 0.5, N * H * W), 0, 1)
    rel = compute_reliability(obs_fields.ravel(), prob_fcst, threshold=64.5)
    print(f"  BSS={rel['brier_skill_score']:.3f}, clim={rel['climatology']:.4f}")

    print("\n=== Full verification report ===")
    report = build_verification_report(
        obs_fields, nwp_fields, qdm_fields, varsh_fields, labels,
        ["active", "break", "depression", "orographic", "western_disturbance"],
    )
    print(f"  Keys: {list(report.keys())}")
    print(f"  NWP  RMSE: {report['overall']['raw_nwp']['continuous']['rmse']}")
    print(f"  Varsh RMSE: {report['overall']['varsh']['continuous']['rmse']}")
    nwp_csi  = report['overall']['raw_nwp']['heavy']['csi']
    varsh_csi= report['overall']['varsh']['heavy']['csi']
    print(f"  NWP  CSI@heavy: {nwp_csi}  (should be > 0)")
    print(f"  Varsh CSI@heavy: {varsh_csi}  (should be > 0)")
    print("OK qrf_probabilistic.py Phase 1")
