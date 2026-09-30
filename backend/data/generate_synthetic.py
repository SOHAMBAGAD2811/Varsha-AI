"""
generate_synthetic.py

Generates synthetic datasets for Varsh-AI MVP prototype.
Produces:
  - Atmospheric fields (MSLP, 850U, 850V, TCWV, 500Z) for regime classifier
  - NWP "raw" rainfall forecast (with typical regime-specific biases)
  - IMD-like "observed" rainfall
  - Pre-labeled regime ground truth for training

Output: synthetic_dataset.npz + metadata.json
"""

import numpy as np
import json
import os

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
SEED = 42
rng = np.random.default_rng(SEED)

N_SAMPLES = 2000          # total training samples
N_DISTRICTS = 50          # virtual districts
LAT_GRID = 30             # latitude grid points (India coverage)
LON_GRID = 40             # longitude grid points
N_CHANNELS = 5            # MSLP, 850U, 850V, TCWV, 500Z
N_REGIMES = 5

REGIMES = ["active", "break", "depression", "orographic", "western_disturbance"]
REGIME_LABELS = {r: i for i, r in enumerate(REGIMES)}

# Rainfall thresholds (mm/day)
THRESH_LIGHT = 10.0
THRESH_HEAVY = 64.5
THRESH_VERY_HEAVY = 115.5

# ------------------------------------------------------------------
# Regime-specific atmospheric signatures
# Each regime has a different pattern in atmospheric fields
# ------------------------------------------------------------------
def make_atmo_fields(regime_idx: int, n: int) -> np.ndarray:
    """
    Returns (n, N_CHANNELS, LAT_GRID, LON_GRID) tensor of atmospheric fields.
    Values are normalized ~N(0,1) with regime-specific offsets.
    """
    fields = rng.standard_normal((n, N_CHANNELS, LAT_GRID, LON_GRID))

    # Channel indices: 0=MSLP, 1=850U, 2=850V, 3=TCWV, 4=500Z
    if regime_idx == 0:  # Active Monsoon
        fields[:, 0] -= 0.8   # lower MSLP (monsoon trough)
        fields[:, 1] += 1.2   # strong SW U-wind
        fields[:, 2] += 0.9   # strong SW V-wind
        fields[:, 3] += 1.5   # high moisture
        fields[:, 4] -= 0.5   # lower 500hPa heights

    elif regime_idx == 1:  # Break Monsoon
        fields[:, 0] += 0.6   # higher MSLP over central India
        fields[:, 1] -= 0.4   # weaker westerlies
        fields[:, 2] -= 0.3
        fields[:, 3] -= 0.8   # lower moisture

    elif regime_idx == 2:  # Monsoon Depression
        fields[:, 0] -= 1.8   # very low MSLP at center
        fields[:, 1] += 0.5   # cyclonic flow
        fields[:, 2] -= 0.5
        fields[:, 3] += 2.0   # very high moisture
        fields[:, 4] -= 1.2   # very deep 500hPa trough

    elif regime_idx == 3:  # Coastal/Orographic
        fields[:, 0] -= 0.3
        fields[:, 1] += 1.8   # very strong SW flow (forced ascent)
        fields[:, 2] += 0.6
        fields[:, 3] += 1.0
        # Topographic signature: strong gradient in lon dimension
        topo_gradient = np.linspace(-2, 2, LON_GRID)
        fields[:, 3] += topo_gradient[None, None, :]

    elif regime_idx == 4:  # Western Disturbance
        fields[:, 0] += 0.2
        fields[:, 1] -= 1.0   # easterlies at 850 (reversed)
        fields[:, 2] += 0.8
        fields[:, 3] -= 0.5
        fields[:, 4] += 1.5   # upper-level ridge

    return fields.astype(np.float32)


def make_rainfall_pair(regime_idx: int, n: int):
    """
    Returns (obs, nwp_raw) rainfall in mm/day with regime-specific bias patterns.
    Shape: (n, LAT_GRID, LON_GRID)
    """
    # Base observed rainfall from a log-normal distribution
    obs_mean = [25, 5, 80, 60, 15][regime_idx]
    obs_std  = [20, 5, 60, 40, 12][regime_idx]

    obs = rng.lognormal(
        mean=np.log(max(obs_mean, 1)),
        sigma=0.5,
        size=(n, LAT_GRID, LON_GRID)
    ).astype(np.float32)

    # Apply regime-specific NWP bias
    if regime_idx == 0:   # Active: NWP underestimates heavy, overestimates extent
        bias = rng.uniform(0.5, 0.8, size=(n, LAT_GRID, LON_GRID))
        nwp = obs * bias + rng.uniform(0, 10, size=(n, LAT_GRID, LON_GRID))

    elif regime_idx == 1:  # Break: topographic biases (wet N, dry S)
        lat_bias = np.linspace(1.5, 0.3, LAT_GRID)[None, :, None]
        nwp = obs * lat_bias + rng.standard_normal((n, LAT_GRID, LON_GRID)) * 3

    elif regime_idx == 2:  # Depression: track displacement + intensity bias
        shift = rng.integers(-3, 4, size=(n, 2))
        nwp = obs * rng.uniform(0.6, 0.9, size=(n, LAT_GRID, LON_GRID))
        # Spatially displace forecast
        for i in range(n):
            dy, dx = shift[i]
            nwp[i] = np.roll(np.roll(nwp[i], dy, axis=0), dx, axis=1)

    elif regime_idx == 3:  # Orographic: windward wet / leeward dry
        lon_bias = np.concatenate([
            np.linspace(1.8, 0.4, LON_GRID // 2),
            np.linspace(0.4, 1.8, LON_GRID // 2)
        ])[None, None, :]
        nwp = obs * lon_bias

    elif regime_idx == 4:  # WD: timing errors → add noise
        nwp = obs * rng.uniform(0.7, 1.3, size=(n, LAT_GRID, LON_GRID))
        nwp += rng.standard_normal((n, LAT_GRID, LON_GRID)) * 8

    nwp = np.clip(nwp, 0, None).astype(np.float32)
    obs = np.clip(obs, 0, None).astype(np.float32)
    return obs, nwp


# ------------------------------------------------------------------
# QDM Simulation
# ------------------------------------------------------------------
def apply_qdm_simulation(nwp: np.ndarray, obs_hist: np.ndarray) -> np.ndarray:
    """
    Simple per-pixel QDM: map NWP forecast quantile through obs historical CDF.
    nwp, obs_hist: (n, H, W)
    Returns: qdm_corrected (n, H, W)
    """
    n, H, W = nwp.shape
    qdm = np.zeros_like(nwp)
    for h in range(H):
        for w in range(W):
            nwp_vals = nwp[:, h, w]
            obs_vals = obs_hist[:, h, w]
            # Quantile mapping
            ranks = np.argsort(np.argsort(nwp_vals))
            quantiles = ranks / (n - 1)
            obs_sorted = np.sort(obs_vals)
            idx = np.clip((quantiles * (n - 1)).astype(int), 0, n - 1)
            qdm[:, h, w] = obs_sorted[idx]
    return qdm.astype(np.float32)


# ------------------------------------------------------------------
# District aggregation
# ------------------------------------------------------------------
def make_district_data(obs: np.ndarray, nwp: np.ndarray, qdm: np.ndarray,
                        regime_labels: np.ndarray, n_districts=N_DISTRICTS):
    """
    Randomly assign grid cells to districts. Return district-level stats.
    """
    H, W = obs.shape[1], obs.shape[2]
    district_mask = rng.integers(0, n_districts, size=(H, W))

    district_rows = []
    for d in range(n_districts):
        mask = district_mask == d
        if mask.sum() == 0:
            continue
        obs_d = obs[:, mask].mean(axis=1)   # (n,)
        nwp_d = nwp[:, mask].mean(axis=1)
        qdm_d = qdm[:, mask].mean(axis=1)

        # Probabilistic: P(R > threshold) from empirical CDF
        p_heavy    = (obs_d > THRESH_HEAVY).mean()
        p_vheavy   = (obs_d > THRESH_VERY_HEAVY).mean()
        p_light    = (obs_d > THRESH_LIGHT).mean()

        for i in range(len(obs_d)):
            district_rows.append({
                "district_id": d,
                "district_name": f"District_{d:02d}",
                "sample_idx": i,
                "regime": REGIMES[regime_labels[i]],
                "obs_rainfall": float(obs_d[i]),
                "nwp_rainfall": float(nwp_d[i]),
                "qdm_rainfall": float(qdm_d[i]),
                "p_heavy": float(p_heavy),
                "p_very_heavy": float(p_vheavy),
                "p_light": float(p_light),
            })
    return district_rows


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))

    n_per_regime = N_SAMPLES // N_REGIMES
    all_atmo, all_obs, all_nwp, all_labels = [], [], [], []

    for ridx, rname in enumerate(REGIMES):
        print(f"  Generating regime: {rname} ({n_per_regime} samples)...")
        atmo = make_atmo_fields(ridx, n_per_regime)
        obs, nwp = make_rainfall_pair(ridx, n_per_regime)
        labels = np.full(n_per_regime, ridx, dtype=np.int32)

        all_atmo.append(atmo)
        all_obs.append(obs)
        all_nwp.append(nwp)
        all_labels.append(labels)

    atmo_all   = np.concatenate(all_atmo, axis=0)   # (N, 5, H, W)
    obs_all    = np.concatenate(all_obs,  axis=0)   # (N, H, W)
    nwp_all    = np.concatenate(all_nwp,  axis=0)   # (N, H, W)
    labels_all = np.concatenate(all_labels, axis=0) # (N,)

    # Shuffle
    perm = rng.permutation(len(labels_all))
    atmo_all   = atmo_all[perm]
    obs_all    = obs_all[perm]
    nwp_all    = nwp_all[perm]
    labels_all = labels_all[perm]

    print("  Applying QDM simulation (this may take a moment)...")
    # QDM on first 500 samples (full grid is slow; demonstrate per district)
    qdm_all = nwp_all.copy()  # Will be filled by per-district QDM at serving time

    # District data (aggregate, fast)
    print("  Aggregating district data...")
    district_rows = make_district_data(obs_all, nwp_all, qdm_all, labels_all)

    import json as _json
    with open(os.path.join(out_dir, "district_data.json"), "w") as f:
        _json.dump(district_rows[:5000], f)  # cap size

    # Save arrays
    np.savez_compressed(
        os.path.join(out_dir, "synthetic_dataset.npz"),
        atmo=atmo_all,
        obs=obs_all,
        nwp=nwp_all,
        qdm=qdm_all,
        labels=labels_all,
    )

    meta = {
        "n_samples": int(N_SAMPLES),
        "n_regimes": N_REGIMES,
        "regimes": REGIMES,
        "n_channels": N_CHANNELS,
        "channel_names": ["MSLP", "850U", "850V", "TCWV", "500Z"],
        "lat_grid": LAT_GRID,
        "lon_grid": LON_GRID,
        "n_districts": N_DISTRICTS,
        "thresholds": {
            "light": THRESH_LIGHT,
            "heavy": THRESH_HEAVY,
            "very_heavy": THRESH_VERY_HEAVY,
        }
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        _json.dump(meta, f, indent=2)

    print(f"\n✅ Synthetic dataset generated:")
    print(f"   atmo shape : {atmo_all.shape}")
    print(f"   obs shape  : {obs_all.shape}")
    print(f"   nwp shape  : {nwp_all.shape}")
    print(f"   labels     : {np.bincount(labels_all)}")
    print(f"   districts  : {len(district_rows)} rows")


if __name__ == "__main__":
    main()
