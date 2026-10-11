#!/usr/bin/env python3
"""Grouped cross-validation on the real dataset: leave-one-species-out (LOSO)
and leave-one-paper-out (LOPO), alongside the existing random 5-fold CV.

Random k-fold CV can leak information when multiple crops from the same
figure/species land in different folds (they share illumination, imaging
conditions and often near-duplicate structure). LOSO and LOPO remove an
entire species or an entire source paper from training in each fold, giving
a stricter, leakage-resistant estimate of generalisation.

Run: python validate_real_grouped.py
"""
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from synth import generate_dataset                       # noqa
from features import extract_features                    # noqa
from colorutil import VISIBLE_MIN_NM, VISIBLE_MAX_NM, srgb8_to_peak_sat_val  # noqa
import cv2                                                # noqa

DATASET = os.path.normpath(os.path.join(HERE, "..", "data", "dataset.jsonl"))
SPAN = VISIBLE_MAX_NM - VISIBLE_MIN_NM


def enc(peak, sat, val):
    return np.array([(peak - VISIBLE_MIN_NM) / SPAN, sat, val], np.float64)


def calibrated_px_per_um(record):
    """Return a measured pixel scale; leave uncalibrated records as heuristic."""
    scale = record.get("scale") or {}
    value = scale.get("px_per_um")
    if scale.get("calibrated") and value is not None and float(value) > 0:
        return float(value)
    return None


def load_real():
    base = os.path.dirname(DATASET)
    X, Y, species, pmcid, real_flag, calibrated = [], [], [], [], [], []
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
        vec, _ = extract_features(img, px_per_um=calibrated_px_per_um(o))
        X.append(vec); Y.append(enc(peak, s2, v2))
        species.append(o.get("species") or o.get("common_name", "unknown"))
        pmcid.append((o.get("source") or {}).get("pmcid", "unknown"))
        real_flag.append(bool(o.get("real_image", False)))
        calibrated.append(calibrated_px_per_um(o) is not None)
    return (np.asarray(X), np.asarray(Y), np.asarray(species),
            np.asarray(pmcid), np.asarray(real_flag), np.asarray(calibrated))


def fit_ridge(X, Y, lam=1.0):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    W = np.linalg.solve(Xs.T @ Xs + lam * np.eye(Xs.shape[1]), Xs.T @ Y)
    return W, mu, sd


def predict(X, W, mu, sd):
    Xs = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    return Xs @ W


def summarize_by_scale(errors, calibrated, label):
    """Report exploratory metrics separately for measured and heuristic scales."""
    for subset, mask in [("all real-image records", np.ones(len(errors), bool)),
                         ("measured-scale records", calibrated),
                         ("heuristic-scale records", ~calibrated)]:
        if not mask.any():
            continue
        e = errors[mask]
        print(f"  [{label}; {subset}] n={len(e)} MAE={e.mean():.1f} nm "
              f"median={np.median(e):.1f} nm")


def grouped_cv(Xs_syn, Ys_syn, Xr, Yr, groups, label, real_mask, calibrated):
    """Leave-one-group-out CV restricted to real_image==True records for
    evaluation (synthetic proxies are always available for training)."""
    uniq = np.unique(groups[real_mask])
    peak_err, eval_calibrated = [], []
    n_folds = 0
    for g in uniq:
        te = np.where((groups == g) & real_mask)[0]
        tr = np.where((groups != g))[0]
        if len(te) == 0 or len(tr) == 0:
            continue
        Xtr = np.vstack([Xs_syn, Xr[tr]]); Ytr = np.vstack([Ys_syn, Yr[tr]])
        W, mu, sd = fit_ridge(Xtr, Ytr)
        pred = predict(Xr[te], W, mu, sd)
        peak_err.append(np.abs((pred[:, 0] - Yr[te, 0]) * SPAN))
        eval_calibrated.append(calibrated[te])
        n_folds += 1
    peak_err = np.concatenate(peak_err)
    eval_calibrated = np.concatenate(eval_calibrated)
    print(f"[{label}]  groups={n_folds}  n={len(peak_err)}  "
          f"peak MAE = {peak_err.mean():.1f} nm  (median {np.median(peak_err):.1f}, "
          f"max {peak_err.max():.1f})")
    summarize_by_scale(peak_err, eval_calibrated, label)
    return peak_err, eval_calibrated


def main():
    print("Generating synthetic coverage set...")
    syn = generate_dataset(800, size=96, seed=0)
    Xs = np.array([extract_features(r["_image"])[0] for r in syn])
    Ys = np.array([enc(r["color"]["peak_nm"], r["color"]["saturation"], r["color"]["value"]) for r in syn])

    Xr, Yr, species, pmcid, real_mask, calibrated = load_real()
    n_real = int(real_mask.sum())
    print(f"Loaded {len(Xr)} records ({n_real} real-image, {len(Xr) - n_real} synthetic-proxy); "
          f"{len(np.unique(species[real_mask]))} species, {len(np.unique(pmcid[real_mask]))} source papers "
          f"among the real-image records.\n")

    # random 5-fold (existing baseline, for comparison) -- restricted to real_image records
    rng = np.random.default_rng(0)
    real_idx = np.where(real_mask)[0]
    idx = rng.permutation(real_idx); k = 5
    folds = np.array_split(idx, k)
    peak_err, eval_calibrated = [], []
    for i in range(k):
        te = folds[i]; tr = np.array([j for j in range(len(Xr)) if j not in set(folds[i])])
        Xtr = np.vstack([Xs, Xr[tr]]); Ytr = np.vstack([Ys, Yr[tr]])
        W, mu, sd = fit_ridge(Xtr, Ytr)
        pred = predict(Xr[te], W, mu, sd)
        peak_err.append(np.abs((pred[:, 0] - Yr[te, 0]) * SPAN))
        eval_calibrated.append(calibrated[te])
    peak_err = np.concatenate(peak_err)
    eval_calibrated = np.concatenate(eval_calibrated)
    print(f"[random 5-fold, real-image only]  n={len(peak_err)}  "
          f"peak MAE = {peak_err.mean():.1f} nm  (median {np.median(peak_err):.1f})")
    summarize_by_scale(peak_err, eval_calibrated, "random 5-fold")

    grouped_cv(Xs, Ys, Xr, Yr, species, "leave-one-species-out (real-image only)", real_mask, calibrated)
    grouped_cv(Xs, Ys, Xr, Yr, pmcid, "leave-one-paper-out (real-image only)", real_mask, calibrated)


if __name__ == "__main__":
    main()
