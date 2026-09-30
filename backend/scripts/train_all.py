"""
scripts/train_all.py

End-to-end training pipeline for Varsh-AI MVP:
  1. Regime Classifier (Gaussian NB)
  2. QDM Correctors (per-regime, fitted from historical data)
  3. QRF Probabilistic Forecaster (per-regime)
  4. Verification metrics (saved to JSON)

Run: python3 scripts/train_all.py
Outputs to: data/ directory
"""

import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from models.regime_classifier import train_classifier, extract_features, GaussianNBRegimeClassifier, REGIMES
from models.qdm_correction import QDMCorrector, apply_ssr
from models.qrf_probabilistic import QuantileRegressionForest, compute_metrics, compute_fss

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data")

def banner(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")

def main():
    t0 = time.time()
    rng = np.random.default_rng(42)

    # ---------------------------------------------------------------
    # Load synthetic dataset
    # ---------------------------------------------------------------
    banner("Loading Dataset")
    ds = np.load(os.path.join(DATA_DIR, "synthetic_dataset.npz"))
    atmo   = ds["atmo"]     # (2000, 5, 30, 40)
    obs    = ds["obs"]      # (2000, 30, 40)
    nwp    = ds["nwp"]      # (2000, 30, 40)
    labels = ds["labels"]   # (2000,)
    N = len(labels)
    print(f"  Loaded {N} samples | {atmo.shape[1]} channels | {atmo.shape[2]}×{atmo.shape[3]} grid")

    # Train/val split (80/20 by sample index, stratified-ish)
    idx = rng.permutation(N)
    train_idx = idx[:int(0.8 * N)]
    val_idx   = idx[int(0.8 * N):]
    print(f"  Train: {len(train_idx)} | Val: {len(val_idx)}")

    # ---------------------------------------------------------------
    # Stage 1: Regime Classifier
    # ---------------------------------------------------------------
    banner("Stage 1 — Regime Classifier")
    clf_path = os.path.join(DATA_DIR, "regime_classifier")
    result = train_classifier(
        os.path.join(DATA_DIR, "synthetic_dataset.npz"),
        clf_path
    )
    print(f"  Overall accuracy : {result['accuracy']:.4f}")
    for r, acc in result['per_regime'].items():
        print(f"  {r:25s}: {acc:.4f}")

    # Predict regimes on full dataset for downstream use
    clf = GaussianNBRegimeClassifier()
    clf.load(clf_path)
    feats = extract_features(atmo)
    pred_labels = clf.predict(feats)
    pred_proba  = clf.predict_proba(feats)
    print(f"  Regime distribution (pred): {np.bincount(pred_labels)}")

    # ---------------------------------------------------------------
    # Stage 2: QDM Correctors (one per regime)
    # ---------------------------------------------------------------
    banner("Stage 2 — QDM Correctors (per-regime)")
    qdm_correctors = {}
    qdm_corrected  = np.zeros_like(nwp)

    for ridx, regime in enumerate(REGIMES):
        mask = labels[train_idx] == ridx
        if mask.sum() < 10:
            continue

        # Flatten spatial: use domain-mean per sample for fitting
        nwp_hist = nwp[train_idx][mask].mean(axis=(1, 2))   # (n_regime,)
        obs_hist = obs[train_idx][mask].mean(axis=(1, 2))

        corrector = QDMCorrector(n_quantiles=100)
        corrector.fit(nwp_hist, obs_hist)
        corrector.save(os.path.join(DATA_DIR, f"qdm_{regime}"))
        qdm_correctors[regime] = corrector
        print(f"  {regime:25s}: fitted on {mask.sum()} samples")

    # Apply QDM to full dataset using predicted regime label
    for ridx, regime in enumerate(REGIMES):
        mask = pred_labels == ridx
        if mask.sum() == 0 or regime not in qdm_correctors:
            continue
        corr = qdm_correctors[regime]
        nwp_flat = nwp[mask].mean(axis=(1, 2))   # domain-mean per sample
        qdm_mean  = corr.transform(nwp_flat)       # corrected domain-mean

        # Scale the full 2D field by the correction ratio
        ratio = qdm_mean / (nwp[mask].mean(axis=(1, 2)) + 1e-6)
        qdm_corrected[mask] = nwp[mask] * ratio[:, None, None]

    qdm_corrected = np.clip(qdm_corrected, 0, None)
    print(f"\n  NWP mean  : {nwp.mean():.2f} mm")
    print(f"  QDM mean  : {qdm_corrected.mean():.2f} mm")
    print(f"  Obs mean  : {obs.mean():.2f} mm")

    # ---------------------------------------------------------------
    # Stage 3: QRF (one per regime, features = atmo stats)
    # ---------------------------------------------------------------
    banner("Stage 3 — Quantile Regression Forests (per-regime)")
    qrf_models = {}

    for ridx, regime in enumerate(REGIMES):
        train_mask = (labels[train_idx] == ridx)
        if train_mask.sum() < 20:
            print(f"  {regime}: skipped (too few samples)")
            continue

        X_train = feats[train_idx][train_mask]
        y_train = obs[train_idx][train_mask].mean(axis=(1, 2))  # domain-mean target

        qrf = QuantileRegressionForest(n_estimators=30, max_depth=5, seed=ridx)
        qrf.fit(X_train, y_train)
        qrf_models[regime] = qrf
        print(f"  {regime:25s}: trained on {train_mask.sum()} samples (n_est=30, depth=5)")

    # ---------------------------------------------------------------
    # Stage 4: Verification Metrics
    # ---------------------------------------------------------------
    banner("Stage 4 — Verification Metrics")

    # Prepare val arrays
    obs_val = obs[val_idx].mean(axis=(1, 2))
    nwp_val = nwp[val_idx].mean(axis=(1, 2))
    qdm_val = qdm_corrected[val_idx].mean(axis=(1, 2))
    lbl_val = labels[val_idx]

    # QRF varsh-ai predictions on val set
    varsh_val = np.zeros(len(val_idx))
    for ridx, regime in enumerate(REGIMES):
        mask = lbl_val == ridx
        if mask.sum() == 0 or regime not in qrf_models:
            varsh_val[mask] = qdm_val[mask]
            continue
        X_v = feats[val_idx][mask]
        varsh_val[mask] = qrf_models[regime].predict_median(X_v)

    # Exceedance probabilities
    thresholds = [10.0, 64.5, 115.5]
    p_exceed = np.zeros((len(val_idx), 3))
    for ridx, regime in enumerate(REGIMES):
        mask = lbl_val == ridx
        if mask.sum() == 0 or regime not in qrf_models:
            continue
        X_v = feats[val_idx][mask]
        p_exceed[mask] = qrf_models[regime].predict_exceedance(X_v, thresholds)

    # Overall metrics
    metrics_nwp   = compute_metrics(obs_val, nwp_val,   threshold=64.5)
    metrics_qdm   = compute_metrics(obs_val, qdm_val,   threshold=64.5)
    metrics_varsh = compute_metrics(obs_val, varsh_val, threshold=64.5)

    print(f"\n  {'Metric':12s}  {'Raw NWP':>10s}  {'QDM':>10s}  {'Varsh-AI':>10s}")
    print(f"  {'-'*50}")
    for k in ['rmse', 'mae', 'bias', 'pod', 'far', 'csi', 'ets']:
        print(f"  {k:12s}  {metrics_nwp[k]:>10.3f}  {metrics_qdm[k]:>10.3f}  {metrics_varsh[k]:>10.3f}")

    # FSS on a representative sample (2D spatial)
    sample_obs  = obs[val_idx[:50]]    # (50, 30, 40)
    sample_nwp  = nwp[val_idx[:50]]
    sample_varsh = qdm_corrected[val_idx[:50]]

    fss_results = {}
    for scale in [3, 5, 10]:
        fss_nwp   = np.mean([compute_fss(sample_obs[i], sample_nwp[i],   64.5, scale) for i in range(50)])
        fss_varsh = np.mean([compute_fss(sample_obs[i], sample_varsh[i], 64.5, scale) for i in range(50)])
        scale_km  = scale * 12  # ~12 km grid → scale in km
        fss_results[f"{scale_km}km"] = {"raw": round(fss_nwp, 3), "varsh": round(fss_varsh, 3)}
        print(f"  FSS @ {scale_km:3d}km   NWP={fss_nwp:.3f}  Varsh-AI={fss_varsh:.3f}")

    # Per-regime metrics
    regime_metrics = {}
    for ridx, regime in enumerate(REGIMES):
        mask = lbl_val == ridx
        if mask.sum() < 5:
            continue
        rm_nwp   = compute_metrics(obs_val[mask], nwp_val[mask],   threshold=64.5)
        rm_varsh = compute_metrics(obs_val[mask], varsh_val[mask], threshold=64.5)
        regime_metrics[regime] = {"raw": rm_nwp, "varsh": rm_varsh, "n_samples": int(mask.sum())}
        print(f"  {regime:25s}: RMSE {rm_nwp['rmse']:.2f} → {rm_varsh['rmse']:.2f}  "
              f"CSI {rm_nwp['csi']:.3f} → {rm_varsh['csi']:.3f}")

    # CRPS-style evaluation (using QRF quantiles as pseudo-ensemble)
    from models.qrf_probabilistic import compute_crps
    crps_vals = []
    for ridx, regime in enumerate(REGIMES):
        mask = lbl_val == ridx
        if mask.sum() == 0 or regime not in qrf_models:
            continue
        X_v = feats[val_idx][mask]
        q_preds = qrf_models[regime].predict_quantiles(X_v, [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9])
        c = compute_crps(obs_val[mask], q_preds)
        crps_vals.append(c)
    mean_crps = float(np.mean(crps_vals)) if crps_vals else 0.0
    print(f"\n  Mean CRPS (QRF): {mean_crps:.3f}")

    # ---------------------------------------------------------------
    # Save all results
    # ---------------------------------------------------------------
    banner("Saving Results")

    verification = {
        "overall": {
            "raw":   metrics_nwp,
            "qdm":   metrics_qdm,
            "varsh": metrics_varsh,
        },
        "by_regime": regime_metrics,
        "fss":       fss_results,
        "crps":      mean_crps,
        "n_val":     int(len(val_idx)),
    }

    with open(os.path.join(DATA_DIR, "verification_results.json"), "w") as f:
        json.dump(verification, f, indent=2)
    print(f"  Saved verification_results.json")

    # District data (re-generate with QDM applied)
    district_data = generate_district_data(obs, nwp, qdm_corrected, varsh_val, lbl_val,
                                            p_exceed, val_idx)
    with open(os.path.join(DATA_DIR, "district_data.json"), "w") as f:
        json.dump(district_data, f, indent=2)
    print(f"  Saved district_data.json ({len(district_data)} districts)")

    # Save QRF exceedance for val set
    np.savez_compressed(
        os.path.join(DATA_DIR, "val_predictions.npz"),
        obs=obs_val, nwp=nwp_val, qdm=qdm_val, varsh=varsh_val,
        p_exceed=p_exceed, labels=lbl_val
    )
    print(f"  Saved val_predictions.npz")

    elapsed = time.time() - t0
    print(f"\n✅ Training complete in {elapsed:.1f}s")


def generate_district_data(obs, nwp, qdm, varsh_val, lbl_val, p_exceed, val_idx):
    """Aggregate grid results into 50 virtual districts."""
    N_DISTRICTS = 50
    H, W = obs.shape[1], obs.shape[2]
    rng = np.random.default_rng(7)
    district_mask = rng.integers(0, N_DISTRICTS, size=(H, W))

    DISTRICT_NAMES = [
        "Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Ahmedabad",
        "Chennai", "Kolkata", "Pune", "Jaipur", "Surat",
        "Lucknow", "Kanpur", "Nagpur", "Indore", "Thane",
        "Bhopal", "Visakhapatnam", "Patna", "Vadodara", "Ghaziabad",
        "Ludhiana", "Agra", "Nashik", "Faridabad", "Meerut",
        "Rajkot", "Kalyan", "Vasai", "Varanasi", "Srinagar",
        "Aurangabad", "Dhanbad", "Amritsar", "Navi Mumbai", "Allahabad",
        "Ranchi", "Howrah", "Coimbatore", "Jabalpur", "Gwalior",
        "Vijayawada", "Jodhpur", "Madurai", "Raipur", "Kota",
        "Guwahati", "Chandigarh", "Solapur", "Hubballi", "Bhubaneswar",
    ]

    from models.regime_classifier import REGIMES
    STATES = {
        "Mumbai": "Maharashtra", "Delhi": "Delhi", "Bengaluru": "Karnataka",
        "Hyderabad": "Telangana", "Ahmedabad": "Gujarat", "Chennai": "Tamil Nadu",
        "Kolkata": "West Bengal", "Pune": "Maharashtra", "Jaipur": "Rajasthan",
        "Surat": "Gujarat", "Lucknow": "Uttar Pradesh", "Kanpur": "Uttar Pradesh",
        "Nagpur": "Maharashtra", "Indore": "Madhya Pradesh", "Thane": "Maharashtra",
        "Bhopal": "Madhya Pradesh", "Visakhapatnam": "Andhra Pradesh",
        "Patna": "Bihar", "Vadodara": "Gujarat", "Ghaziabad": "Uttar Pradesh",
        "Ludhiana": "Punjab", "Agra": "Uttar Pradesh", "Nashik": "Maharashtra",
        "Faridabad": "Haryana", "Meerut": "Uttar Pradesh", "Rajkot": "Gujarat",
        "Kalyan": "Maharashtra", "Vasai": "Maharashtra", "Varanasi": "Uttar Pradesh",
        "Srinagar": "J&K", "Aurangabad": "Maharashtra", "Dhanbad": "Jharkhand",
        "Amritsar": "Punjab", "Navi Mumbai": "Maharashtra", "Allahabad": "Uttar Pradesh",
        "Ranchi": "Jharkhand", "Howrah": "West Bengal", "Coimbatore": "Tamil Nadu",
        "Jabalpur": "Madhya Pradesh", "Gwalior": "Madhya Pradesh",
        "Vijayawada": "Andhra Pradesh", "Jodhpur": "Rajasthan",
        "Madurai": "Tamil Nadu", "Raipur": "Chhattisgarh", "Kota": "Rajasthan",
        "Guwahati": "Assam", "Chandigarh": "Punjab", "Solapur": "Maharashtra",
        "Hubballi": "Karnataka", "Bhubaneswar": "Odisha",
    }

    # Use validation samples to build district statistics
    obs_val  = obs[val_idx].mean(axis=(1, 2))
    nwp_val  = nwp[val_idx].mean(axis=(1, 2))
    qdm_val  = qdm[val_idx].mean(axis=(1, 2))

    result = []
    for d in range(N_DISTRICTS):
        d_mask = district_mask == d
        if d_mask.sum() == 0:
            continue
        name  = DISTRICT_NAMES[d] if d < len(DISTRICT_NAMES) else f"District_{d:02d}"
        state = STATES.get(name, "India")

        # Use the most common regime in this district's validation samples
        lbl_counts = np.bincount(lbl_val, minlength=5)
        regime_idx = int(lbl_counts.argmax())

        # Representative sample for this district (average a random subset)
        sample_idxs = rng.choice(len(obs_val), size=min(20, len(obs_val)), replace=False)

        result.append({
            "district_id":   d,
            "name":          name,
            "state":         state,
            "regime":        REGIMES[regime_idx],
            "nwp_mm":        round(float(nwp_val[sample_idxs].mean()), 1),
            "qdm_mm":        round(float(qdm_val[sample_idxs].mean()), 1),
            "varsh_mm":      round(float(varsh_val[sample_idxs].mean()), 1),
            "obs_mm":        round(float(obs_val[sample_idxs].mean()), 1),
            "p_light":       round(float(p_exceed[sample_idxs, 0].mean()), 3),
            "p_heavy":       round(float(p_exceed[sample_idxs, 1].mean()), 3),
            "p_very_heavy":  round(float(p_exceed[sample_idxs, 2].mean()), 3),
            "lat":           float(8  + (d // 10) * 2.5),
            "lon":           float(68 + (d %  10) * 2.8),
        })

    return result


if __name__ == "__main__":
    main()
