"""
models/qdm_correction.py

Quantile Delta Mapping (QDM) + Singularity Stochastic Removal (SSR)
All implemented in pure numpy.

QDM Formula:
  X*_sim,p(t) = F^{-1}_obs,h(τ) × [X_sim,p(t) / F^{-1}_sim,h(τ)]

where τ = F_sim,p(X_sim,p(t))  [the current forecast quantile in the sim CDF]

SSR pre-processing removes artificial drizzle before QDM by stochastically
replacing sub-threshold forecast values.
"""

from __future__ import annotations
import numpy as np

# IMD wet-day threshold (mm/day)
WET_THRESHOLD = 0.1


# ---------------------------------------------------------------------------
# SSR — Singularity Stochastic Removal
# ---------------------------------------------------------------------------

def apply_ssr(
    forecast: np.ndarray,
    threshold: float = WET_THRESHOLD,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Replace sub-threshold (drizzle) forecast values with random samples
    drawn uniformly from [0, threshold).

    forecast: any shape (mm/day)
    Returns: same shape, drizzle-free
    """
    if rng is None:
        rng = np.random.default_rng(42)
    out = forecast.copy()
    mask = (out > 0) & (out < threshold)
    out[mask] = rng.uniform(0, threshold, size=mask.sum())
    return out


# ---------------------------------------------------------------------------
# QDM — Quantile Delta Mapping
# ---------------------------------------------------------------------------

class QDMCorrector:
    """
    Fits QDM correction from historical NWP forecasts and observations.
    Then applies it to future (projection-period) forecasts.

    Usage:
        corrector = QDMCorrector()
        corrector.fit(hist_nwp, hist_obs)         # 1-D arrays (time,)
        corrected = corrector.transform(new_nwp)  # 1-D array
    """

    def __init__(self, n_quantiles: int = 100, threshold: float = WET_THRESHOLD):
        self.n_quantiles = n_quantiles
        self.threshold   = threshold
        self.quantile_levels_  = None
        self.nwp_quantiles_    = None
        self.obs_quantiles_    = None
        self.fitted_           = False

    def fit(self, hist_nwp: np.ndarray, hist_obs: np.ndarray):
        """
        hist_nwp: (T,) historical NWP forecast values
        hist_obs: (T,) historical observed IMD values
        """
        # SSR on historical NWP before fitting
        rng = np.random.default_rng(0)
        hist_nwp_ssr = apply_ssr(hist_nwp, self.threshold, rng)

        q_levels = np.linspace(0, 100, self.n_quantiles + 1)
        self.quantile_levels_ = q_levels
        self.nwp_quantiles_   = np.percentile(hist_nwp_ssr, q_levels)
        self.obs_quantiles_   = np.percentile(hist_obs,     q_levels)
        self.fitted_          = True
        return self

    def transform(self, new_nwp: np.ndarray) -> np.ndarray:
        """
        Apply QDM bias correction to a new (projection) NWP forecast.
        Preserves the relative change (delta) at each quantile.
        """
        if not self.fitted_:
            raise RuntimeError("QDMCorrector must be fitted first.")

        rng = np.random.default_rng(1)
        nwp_ssr = apply_ssr(new_nwp, self.threshold, rng)
        corrected = np.zeros_like(nwp_ssr)

        for i, val in enumerate(nwp_ssr.flat):
            # τ = F_sim,p(val)  — find quantile rank in historical NWP CDF
            tau_pct = float(np.searchsorted(self.nwp_quantiles_, val)) / self.n_quantiles * 100
            tau_pct = np.clip(tau_pct, 0, 100)

            # F^{-1}_sim,h(τ) — historical NWP quantile at this level
            nwp_hist_q = np.interp(tau_pct, self.quantile_levels_, self.nwp_quantiles_)

            # F^{-1}_obs,h(τ) — historical obs quantile at this level
            obs_hist_q = np.interp(tau_pct, self.quantile_levels_, self.obs_quantiles_)

            # Δ(t) = X_sim,p(t) / F^{-1}_sim,h(τ)
            if nwp_hist_q > 1e-6:
                delta = val / nwp_hist_q
            else:
                delta = 1.0

            corrected.flat[i] = obs_hist_q * delta

        return np.clip(corrected, 0, None)

    def save(self, path: str):
        np.savez(path,
                 quantile_levels=self.quantile_levels_,
                 nwp_quantiles=self.nwp_quantiles_,
                 obs_quantiles=self.obs_quantiles_)

    def load(self, path: str):
        d = np.load(path + ".npz")
        self.quantile_levels_ = d["quantile_levels"]
        self.nwp_quantiles_   = d["nwp_quantiles"]
        self.obs_quantiles_   = d["obs_quantiles"]
        self.fitted_          = True
        return self


# ---------------------------------------------------------------------------
# Spatial QDM — apply per-pixel (grid cell) correction
# ---------------------------------------------------------------------------

def fit_spatial_qdm(
    hist_nwp: np.ndarray,
    hist_obs: np.ndarray,
    n_quantiles: int = 50,
) -> list[list[QDMCorrector]]:
    """
    hist_nwp, hist_obs: (T, H, W)
    Returns: H×W grid of fitted QDMCorrector objects
    """
    T, H, W = hist_nwp.shape
    correctors = [[None] * W for _ in range(H)]
    for h in range(H):
        for w in range(W):
            c = QDMCorrector(n_quantiles=n_quantiles)
            c.fit(hist_nwp[:, h, w], hist_obs[:, h, w])
            correctors[h][w] = c
    return correctors


def apply_spatial_qdm(
    new_nwp: np.ndarray,
    correctors: list[list[QDMCorrector]],
) -> np.ndarray:
    """
    new_nwp: (T, H, W)
    correctors: H×W grid of fitted QDMCorrectors
    Returns: (T, H, W) corrected rainfall
    """
    T, H, W = new_nwp.shape
    corrected = np.zeros_like(new_nwp)
    for h in range(H):
        for w in range(W):
            corrected[:, h, w] = correctors[h][w].transform(new_nwp[:, h, w])
    return corrected


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(99)
    # Simulated historical data: biased NWP (0.7x obs) + noise
    T = 500
    obs_hist = rng.lognormal(mean=2.5, sigma=0.8, size=T)
    nwp_hist = obs_hist * 0.7 + rng.normal(0, 5, T)
    nwp_hist = np.clip(nwp_hist, 0, None)

    # New forecast
    nwp_new = rng.lognormal(mean=2.5, sigma=0.8, size=50) * 0.7

    corrector = QDMCorrector(n_quantiles=50)
    corrector.fit(nwp_hist, obs_hist)
    corrected = corrector.transform(nwp_new)

    print(f"NWP raw mean   : {nwp_new.mean():.2f} mm")
    print(f"QDM corrected  : {corrected.mean():.2f} mm")
    print(f"Obs hist mean  : {obs_hist.mean():.2f} mm")
    print("✅ QDM module OK")
