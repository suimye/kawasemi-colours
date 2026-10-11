"""Synthetic spongy-structure SEM image generator + physics forward model.

Given structural parameters (characteristic period d, pore_fraction, disorder,
anisotropy) this produces:
  1. a realistic grayscale SEM-like image of a quasi-ordered spongy keratin
     network (pores = dark, walls = bright, or invertible), and
  2. the resulting color via a physics forward model built on the shared
     relation  peak ~= 2 * n_eff * d.

This gives model-labelled training data and supplies the internal synthetic target
for both the CNN (train.py) and the inverse model (inverse.py).

Torch-free (numpy + a little scipy-free filtering via cv2/numpy).
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np

from colorutil import (
    peak_from_structure,
    hsv_like_to_rgb,
    rgb01_to_srgb8,
    rgb_to_hex,
    srgb8_to_peak_sat_val,
    VISIBLE_MIN_NM,
    VISIBLE_MAX_NM,
)


# ---------------------------------------------------------------------------
# Physics forward model: structural params -> color
# ---------------------------------------------------------------------------
def forward_color(char_length_nm, pore_fraction, disorder, anisotropy,
                  return_components=False):
    """Structural params -> (peak_nm, saturation, value) and sRGB.

    - peak_nm  : shared relation peak = 2 * n_eff * d
    - saturation grows with ORDER (i.e. falls with disorder). Anisotropy also
      sharpens the reflectance a little.
    - value (brightness) grows with the amount of scattering interfaces, i.e.
      peaks near pore_fraction ~ 0.5 (maximal air/keratin contrast) and falls
      toward fully solid or fully empty.
    """
    peak = peak_from_structure(char_length_nm, pore_fraction)

    # saturation: high order (low disorder) -> saturated. disorder in ~[0,1].
    dis = float(np.clip(disorder, 0.0, 1.0))
    saturation = float(np.clip(1.0 - 0.9 * dis, 0.05, 1.0))
    # anisotropy mildly increases apparent saturation (directional coherence)
    saturation = float(np.clip(saturation * (1.0 + 0.15 * anisotropy), 0.05, 1.0))

    # brightness: contrast term maximized at pf~0.5
    pf = float(np.clip(pore_fraction, 0.0, 1.0))
    contrast = 4.0 * pf * (1.0 - pf)  # 0..1, peak at 0.5
    value = float(np.clip(0.45 + 0.5 * contrast, 0.1, 1.0))

    rgb01 = hsv_like_to_rgb(peak, saturation, value)
    srgb8 = rgb01_to_srgb8(rgb01)
    if return_components:
        return dict(peak_nm=peak, saturation=saturation, value=value,
                    srgb=srgb8, hex=rgb_to_hex(srgb8))
    return peak, saturation, value


# ---------------------------------------------------------------------------
# Synthetic SEM image generator
# ---------------------------------------------------------------------------
def _spectral_blob_field(size, period_px, disorder, anisotropy, rng):
    """Generate a random field whose power spectrum is a ring at 1/period_px.

    This yields a quasi-ordered isotropic (or anisotropic) blob pattern that
    looks like a spongy network. disorder widens the ring; anisotropy squashes
    it into an oriented pattern (channel-like).
    """
    fy = np.fft.fftfreq(size)[:, None]
    fx = np.fft.fftfreq(size)[None, :]

    # anisotropic frequency metric: stretch one axis
    ax = 1.0 + 3.0 * anisotropy
    fr = np.sqrt((fx * ax) ** 2 + (fy / ax) ** 2)

    f0 = 1.0 / max(period_px, 2.0)
    ring_width = f0 * (0.15 + 1.2 * float(np.clip(disorder, 0, 1)) + 0.05)
    ring = np.exp(-((fr - f0) ** 2) / (2.0 * ring_width ** 2))

    phase = rng.uniform(0, 2 * np.pi, size=(size, size))
    spec = ring * np.exp(1j * phase)
    field = np.fft.ifft2(spec).real
    field -= field.mean()
    std = field.std() + 1e-9
    field /= std
    return field


def generate_image(char_length_nm, pore_fraction, disorder, anisotropy,
                   px_per_um=None, size=192, seed=None):
    """Render a grayscale SEM-like image (uint8) of the spongy structure.

    px_per_um: if provided, the period is rendered at the true physical scale.
    Otherwise a default scale is chosen so the period is ~10-16 px (a typical
    SEM crop showing many repeats).
    """
    rng = np.random.default_rng(seed)

    if px_per_um is not None:
        period_px = (char_length_nm / 1000.0) * px_per_um
        period_px = float(np.clip(period_px, 3.0, size / 3.0))
    else:
        # default: render ~14 repeats across the frame
        period_px = float(np.clip(size / 14.0, 6.0, size / 4.0))

    field = _spectral_blob_field(size, period_px, disorder, anisotropy, rng)

    # threshold to set pore_fraction (pores dark). higher pore_fraction -> more dark area
    thresh = np.quantile(field, float(np.clip(pore_fraction, 0.02, 0.98)))
    pores = field < thresh  # boolean, True where pore

    img = np.where(pores, 0.15, 0.85).astype(np.float32)

    # soften edges (SEM has finite resolution / depth) and add grain
    blur_k = max(1, int(round(period_px * 0.15)) * 2 + 1)
    img = cv2.GaussianBlur(img, (blur_k, blur_k), 0)

    # subtle intensity gradient across walls (SEM edge brightening)
    edges = cv2.Laplacian(img, cv2.CV_32F, ksize=3)
    img = np.clip(img + 0.25 * np.abs(edges), 0, 1)

    # sensor noise
    noise = rng.normal(0, 0.04, size=img.shape).astype(np.float32)
    img = np.clip(img + noise, 0, 1)

    return (img * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Dataset sampling
# ---------------------------------------------------------------------------
def sample_params(rng):
    """Sample plausible structural params covering the visible range.

    We sample the peak roughly uniformly across the visible band then back out
    a consistent char_length so the training set spans blue->red evenly.
    """
    pore_fraction = float(rng.uniform(0.30, 0.70))
    peak_target = float(rng.uniform(VISIBLE_MIN_NM + 20, VISIBLE_MAX_NM - 50))
    from colorutil import d_from_peak
    char_length_nm = d_from_peak(peak_target, pore_fraction)
    char_length_nm = float(np.clip(char_length_nm, 90.0, 320.0))
    disorder = float(rng.uniform(0.05, 0.75))
    anisotropy = float(rng.uniform(0.0, 0.6))
    # Sample a physical scale (px per micron) so the period is rendered at its
    # true pixel size. This makes the absolute period learnable from the image
    # alone (matching how features.py assumes DEFAULT_PX_PER_UM). We keep the
    # scale in a band around that default so image period-in-px varies with d.
    from features import DEFAULT_PX_PER_UM
    px_per_um = float(rng.uniform(0.75, 1.35)) * DEFAULT_PX_PER_UM
    return dict(char_length_nm=char_length_nm, pore_fraction=pore_fraction,
                disorder=disorder, anisotropy=anisotropy, px_per_um=px_per_um)


def generate_dataset(n, out_dir=None, size=128, seed=0, save_images=False):
    """Yield / optionally save n synthetic (image, params, color) samples.

    Returns list of dicts. If out_dir + save_images, writes PNGs there.
    """
    rng = np.random.default_rng(seed)
    records = []
    if out_dir and save_images:
        os.makedirs(out_dir, exist_ok=True)
    for i in range(n):
        p = sample_params(rng)
        img = generate_image(size=size, seed=int(rng.integers(0, 2**31)), **p)
        # color depends only on structural params, not on imaging scale
        cp = {k: p[k] for k in ("char_length_nm", "pore_fraction",
                                "disorder", "anisotropy")}
        color = forward_color(return_components=True, **cp)
        rec = dict(index=i, params=p, color=color)
        if out_dir and save_images:
            path = os.path.join(out_dir, f"synth_{i:05d}.png")
            cv2.imwrite(path, img)
            rec["image_file"] = path
        else:
            rec["_image"] = img  # kept in-memory for training
        records.append(rec)
    return records


def main():
    ap = argparse.ArgumentParser(description="Synthetic spongy SEM generator")
    ap.add_argument("--n", type=int, default=8, help="number of samples")
    ap.add_argument("--out", default="synth_out", help="output dir")
    ap.add_argument("--size", type=int, default=192)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    recs = generate_dataset(args.n, out_dir=args.out, size=args.size,
                            seed=args.seed, save_images=True)
    meta = [{k: v for k, v in r.items() if k != "_image"} for r in recs]
    with open(os.path.join(args.out, "synth_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {len(recs)} synthetic samples to {args.out}/")
    for r in recs[:5]:
        c = r["color"]
        p = r["params"]
        print(f"  #{r['index']}: d={p['char_length_nm']:.0f}nm pf={p['pore_fraction']:.2f} "
              f"dis={p['disorder']:.2f} -> peak={c['peak_nm']:.0f}nm hex={c['hex']}")


if __name__ == "__main__":
    main()
