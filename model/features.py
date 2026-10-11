"""FEATS_V1 structural feature extractor (torch-free: opencv + numpy).

Extracts the 7-dim structural feature vector defined in docs/DATA_CONTRACT.md
from a grayscale SEM image:

    [ char_length_nm,     # dominant spatial period (nm) from radial FFT peak
      domain_std_nm,      # spread of the period (order/disorder)
      pore_fraction,      # air/void volume fraction (0..1) via Otsu threshold
      pore_diameter_nm,   # mean pore size
      wall_thickness_nm,  # mean solid wall thickness (distance transform)
      anisotropy,         # 0=isotropic .. 1=aligned (FFT angular spread)
      connectivity ]      # skeleton/euler-based network connectivity proxy

Physical scale: if px_per_um is known (from the scale bar) all *_nm values are
in true nanometres. If unknown we fall back to a default px_per_um so the
pipeline still runs; features remain internally consistent for training.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import cv2
import numpy as np

FEATS_V1 = [
    "char_length_nm",
    "domain_std_nm",
    "pore_fraction",
    "pore_diameter_nm",
    "wall_thickness_nm",
    "anisotropy",
    "connectivity",
]

# Fallback scale when the scale bar is unknown. Assumes a typical SEM crop
# where the field of view is ~ a couple microns. Chosen so a ~14px period maps
# to ~180nm (typical spongy keratin). px_per_um = px / um.
DEFAULT_PX_PER_UM = 78.0


def _px_to_nm(px, px_per_um):
    return float(px) / px_per_um * 1000.0


def _radial_power_spectrum(gray):
    """Return (radii_px_period, power) radial average of the FFT magnitude."""
    g = gray.astype(np.float32)
    g = g - g.mean()
    # window to reduce spectral leakage
    win = np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1]))
    g = g * win
    F = np.fft.fftshift(np.fft.fft2(g))
    mag = np.abs(F) ** 2

    h, w = mag.shape
    cy, cx = h / 2.0, w / 2.0
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r_int = r.astype(np.int32)
    rmax = int(min(cy, cx))
    radial = np.zeros(rmax)
    for k in range(1, rmax):  # skip DC (k=0)
        mask = r_int == k
        if mask.any():
            radial[k] = mag[mask].mean()
    return radial, mag, (cy, cx)


def _dominant_period(gray, px_per_um):
    """Characteristic period + spread from the radial power spectrum."""
    radial, _, _ = _radial_power_spectrum(gray)
    h = gray.shape[0]
    freqs = np.arange(len(radial)) / h  # cycles/pixel

    # search a plausible band: period 3..60 px
    lo = max(1, int(h / 60))
    hi = min(len(radial) - 1, int(h / 3))
    band = radial[lo:hi]
    if band.size == 0 or band.max() <= 0:
        return _px_to_nm(14.0, px_per_um), _px_to_nm(4.0, px_per_um)

    kpeak = lo + int(np.argmax(band))
    f_peak = freqs[kpeak]
    period_px = 1.0 / f_peak if f_peak > 0 else 14.0

    # spread: intensity-weighted std of period around the peak within band
    idx = np.arange(lo, hi)
    weights = radial[lo:hi].copy()
    weights = np.clip(weights, 0, None)
    periods = h / np.maximum(idx, 1)
    wsum = weights.sum() + 1e-9
    mean_p = float((periods * weights).sum() / wsum)
    var_p = float((weights * (periods - mean_p) ** 2).sum() / wsum)
    std_px = np.sqrt(max(var_p, 0.0))

    return _px_to_nm(period_px, px_per_um), _px_to_nm(std_px, px_per_um)


def _anisotropy(mag, center):
    """Angular spread of spectral energy. 0=isotropic ring, 1=oriented."""
    h, w = mag.shape
    cy, cx = center
    y, x = np.indices((h, w))
    dx = x - cx
    dy = y - cy
    r = np.sqrt(dx ** 2 + dy ** 2)
    theta = np.arctan2(dy, dx)  # -pi..pi
    # consider a mid-frequency annulus (exclude DC and very high freq)
    rmax = min(cy, cx)
    ann = (r > 0.08 * rmax) & (r < 0.6 * rmax)
    if not ann.any():
        return 0.0
    e = mag[ann]
    th = theta[ann]
    # fold to [0, pi) since spectrum is symmetric
    th = np.mod(th, np.pi)
    nb = 36
    hist, _ = np.histogram(th, bins=nb, range=(0, np.pi), weights=e)
    hist = hist / (hist.sum() + 1e-9)
    # anisotropy = 1 - normalized entropy (uniform -> 0, peaked -> 1)
    ent = -np.sum(hist * np.log(hist + 1e-12))
    ent_max = np.log(nb)
    return float(np.clip(1.0 - ent / ent_max, 0.0, 1.0))


def _otsu_pore(gray):
    """Otsu threshold -> pore mask (pores are the dark phase)."""
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # th==255 is bright (walls). pores = dark = 0.
    pore = (th == 0).astype(np.uint8)
    wall = (th == 255).astype(np.uint8)
    return pore, wall


def _mean_feature_size(mask, px_per_um):
    """Mean diameter of the True regions via distance transform (2x max EDT)."""
    if mask.sum() == 0:
        return 0.0
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    # peak local distances approximate radii; use the mean of the top values
    vals = dt[dt > 0]
    if vals.size == 0:
        return 0.0
    # ridge (skeleton) distances approximate the local radius
    thr = np.quantile(vals, 0.75)
    radius_px = float(vals[vals >= thr].mean())
    return _px_to_nm(2.0 * radius_px, px_per_um)


def _connectivity(wall):
    """Network connectivity proxy from the solid (wall) phase.

    Uses connected components: a well-connected single network -> value near 1;
    a fragmented set of many disconnected blobs -> value near 0.
    """
    if wall.sum() == 0:
        return 0.0
    num, labels = cv2.connectedComponents(wall)
    ncomp = num - 1  # exclude background
    if ncomp <= 0:
        return 0.0
    sizes = np.array([(labels == i).sum() for i in range(1, num)], dtype=np.float64)
    largest_frac = sizes.max() / (sizes.sum() + 1e-9)
    # combine: dominance of the largest component and fewness of components
    comp_penalty = 1.0 / (1.0 + 0.02 * ncomp)
    return float(np.clip(0.5 * largest_frac + 0.5 * comp_penalty, 0.0, 1.0))


def extract_features(image, px_per_um=None):
    """Extract the FEATS_V1 7-dim vector from a grayscale image (uint8 or path).

    Returns (vector: np.ndarray[7], dict).
    """
    if isinstance(image, str):
        gray = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(image)
    else:
        gray = image
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.uint8)

    ppu = DEFAULT_PX_PER_UM if px_per_um is None else float(px_per_um)

    char_length_nm, domain_std_nm = _dominant_period(gray, ppu)
    radial, mag, center = _radial_power_spectrum(gray)
    aniso = _anisotropy(mag, center)

    pore, wall = _otsu_pore(gray)
    pore_fraction = float(pore.mean())
    pore_diameter_nm = _mean_feature_size(pore, ppu)
    wall_thickness_nm = _mean_feature_size(wall, ppu)
    connectivity = _connectivity(wall)

    vec = np.array([
        char_length_nm,
        domain_std_nm,
        pore_fraction,
        pore_diameter_nm,
        wall_thickness_nm,
        aniso,
        connectivity,
    ], dtype=np.float32)

    d = {k: float(v) for k, v in zip(FEATS_V1, vec)}
    d["px_per_um"] = ppu
    return vec, d


def main():
    ap = argparse.ArgumentParser(description="FEATS_V1 feature extractor")
    ap.add_argument("path", help="image file or folder")
    ap.add_argument("--px-per-um", type=float, default=None,
                    help="pixels per micron (from scale bar); default heuristic")
    ap.add_argument("--out", default=None, help="write JSONL of features here")
    args = ap.parse_args()

    if os.path.isdir(args.path):
        files = sorted(glob.glob(os.path.join(args.path, "*.png")) +
                       glob.glob(os.path.join(args.path, "*.jpg")) +
                       glob.glob(os.path.join(args.path, "*.tif")))
    else:
        files = [args.path]

    out_f = open(args.out, "w") if args.out else None
    for f in files:
        vec, d = extract_features(f, px_per_um=args.px_per_um)
        rec = {"image_file": f, "feats_v1": [round(float(x), 4) for x in vec], "features": d}
        line = json.dumps(rec)
        if out_f:
            out_f.write(line + "\n")
        print(f"{os.path.basename(f)}: "
              + " ".join(f"{k}={d[k]:.1f}" if k.endswith('nm') else f"{k}={d[k]:.2f}"
                         for k in FEATS_V1))
    if out_f:
        out_f.close()
        print(f"wrote features -> {args.out}")


if __name__ == "__main__":
    main()
