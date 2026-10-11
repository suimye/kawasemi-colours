#!/usr/bin/env python3
"""Train the interpretable FEATS_V1 -> colour model ON THE REAL DATA (torch-free).

The default train.py trains the numpy baseline on synthetic data and only uses
the real images to fine-tune the torch CNN. torch is unavailable in this
sandbox (no network for pip), so here we instead fit the interpretable ridge
regressor on synthetic + the real records, and honestly evaluate it on the real
records with k-fold cross-validation (leave-real-out), reporting peak-wavelength
MAE in nm. Saves model/baseline.npz for the exporter/simulator.

Run on the Mac too if you like; for the deep CNN use:  python train.py
"""
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from synth import generate_dataset                       # noqa
from features import extract_features                    # noqa
from colorutil import (VISIBLE_MIN_NM, VISIBLE_MAX_NM,    # noqa
                       srgb8_to_peak_sat_val)
import cv2                                                # noqa

DATASET = os.path.normpath(os.path.join(HERE, "..", "data", "dataset.jsonl"))
BASELINE = os.environ.get("KAWASEMI_BASELINE_OUT", os.path.join(HERE, "baseline.npz"))
SPAN = VISIBLE_MAX_NM - VISIBLE_MIN_NM


def enc(peak, sat, val):
    return np.array([(peak - VISIBLE_MIN_NM) / SPAN, sat, val], np.float64)


def calibrated_px_per_um(record):
    """Return a record's measured pixel scale, never the heuristic fallback."""
    scale = record.get("scale") or {}
    value = scale.get("px_per_um")
    if scale.get("calibrated") and value is not None and float(value) > 0:
        return float(value)
    return None


def load_real():
    base = os.path.dirname(DATASET); X, Y, meta = [], [], []
    for line in open(DATASET):
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        if not o.get("colour_training_eligible", True):
            continue
        p = o.get("image_file", "")
        p = os.path.join(base, p) if not os.path.isabs(p) else p
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        c = o.get("color", {}); srgb = c.get("srgb")
        if not srgb:
            continue
        p2, s2, v2 = srgb8_to_peak_sat_val(srgb)
        peak = float(c["peak_nm"]) if c.get("peak_nm") else p2
        # Absolute-length FEATS use the image-specific scale only where it was
        # measured from a scale bar.  Otherwise features.py records its explicit
        # heuristic fallback; callers must not interpret those lengths as measured.
        vec, _ = extract_features(img, px_per_um=calibrated_px_per_um(o))
        X.append(vec); Y.append(enc(peak, s2, v2))
        meta.append((o.get("common_name", ""), o.get("real_image", False), peak))
    return np.asarray(X), np.asarray(Y), meta


def fit_ridge(X, Y, lam=1.0):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    W = np.linalg.solve(Xs.T @ Xs + lam * np.eye(Xs.shape[1]), Xs.T @ Y)
    return W, mu, sd


def predict(X, W, mu, sd):
    Xs = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    return Xs @ Xs.T.T if False else Xs @ W


def main():
    print("Generating synthetic coverage set...")
    syn = generate_dataset(800, size=96, seed=0)
    Xs = np.array([extract_features(r["_image"])[0] for r in syn])
    Ys = np.array([enc(r["color"]["peak_nm"], r["color"]["saturation"], r["color"]["value"]) for r in syn])

    Xr, Yr, meta = load_real()
    nreal = len(Xr)
    real_mask = np.array([m[1] for m in meta])
    print(f"Loaded {nreal} eligible records ({int(real_mask.sum())} real-image, "
          f"{nreal - int(real_mask.sum())} synthetic-proxy).")

    # --- k-fold CV on REAL records (train on synth + other real folds) ---
    rng = np.random.default_rng(0)
    idx = rng.permutation(nreal); k = 5
    folds = np.array_split(idx, k)
    peak_err, sat_err = [], []
    for i in range(k):
        te = folds[i]; tr = np.concatenate([folds[j] for j in range(k) if j != i])
        Xtr = np.vstack([Xs, Xr[tr]]); Ytr = np.vstack([Ys, Yr[tr]])
        W, mu, sd = fit_ridge(Xtr, Ytr)
        pred = predict(Xr[te], W, mu, sd)
        peak_err.append(np.abs((pred[:, 0] - Yr[te, 0]) * SPAN))
        sat_err.append(np.abs(pred[:, 1] - Yr[te, 1]))
    peak_err = np.concatenate(peak_err); sat_err = np.concatenate(sat_err)
    print(f"\n[REAL 5-fold CV]  peak MAE = {peak_err.mean():.1f} nm  "
          f"(median {np.median(peak_err):.1f})   sat MAE = {sat_err.mean():.3f}")

    # baseline trained on synthetic ONLY, evaluated on real (for comparison)
    W0, mu0, sd0 = fit_ridge(Xs, Ys)
    p0 = predict(Xr, W0, mu0, sd0)
    e0 = np.abs((p0[:, 0] - Yr[:, 0]) * SPAN)
    print(f"[synthetic-only -> real]  peak MAE = {e0.mean():.1f} nm   "
          f"(training on real cuts error by {100*(1-peak_err.mean()/e0.mean()):.0f}%)")

    # --- final model: fit on ALL synthetic + ALL real, save ---
    Wf, muf, sdf = fit_ridge(np.vstack([Xs, Xr]), np.vstack([Ys, Yr]))
    np.savez(BASELINE, W=Wf, mu=muf, sd=sdf)
    print(f"\nSaved combined-trained model -> {BASELINE}")
    print("Next: python export_params.py && python ../scripts/adapt_model_for_ui.py")


if __name__ == "__main__":
    main()
