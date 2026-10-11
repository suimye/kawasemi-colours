"""Shared color / physics utilities for the kawasemi_colors model.

Torch-free. Implements the shared physics relation from docs/DATA_CONTRACT.md:

    peak_wavelength ~= 2 * n_eff * d
    n_eff = pore_fraction * n_air + (1 - pore_fraction) * n_keratin

and the color conventions (sRGB 8-bit, hex, dominant wavelength peak_nm).

These constants are the single source of truth mirrored into
model/model_params.json by export_params.py so the browser simulator can
reproduce the model without python.
"""
from __future__ import annotations

import math
import numpy as np

# ---------------------------------------------------------------------------
# Physics constants (shared prior). Any change here must be re-exported.
# ---------------------------------------------------------------------------
N_KERATIN = 1.56
N_AIR = 1.0

VISIBLE_MIN_NM = 380.0
VISIBLE_MAX_NM = 750.0


def n_eff(pore_fraction: float) -> float:
    """Volume-weighted effective refractive index."""
    pf = float(np.clip(pore_fraction, 0.0, 1.0))
    return pf * N_AIR + (1.0 - pf) * N_KERATIN


def peak_from_structure(char_length_nm: float, pore_fraction: float) -> float:
    """First-order coherent-scattering peak: peak ~= 2 * n_eff * d."""
    return 2.0 * n_eff(pore_fraction) * char_length_nm


def d_from_peak(peak_nm: float, pore_fraction: float) -> float:
    """Analytic inversion: d = peak / (2 * n_eff)."""
    return peak_nm / (2.0 * n_eff(pore_fraction))


# ---------------------------------------------------------------------------
# Wavelength <-> sRGB. Approximate CIE-based mapping (good enough for a
# structural-color demo and fully deterministic / torch-free).
# ---------------------------------------------------------------------------
def wavelength_to_rgb(wavelength_nm: float, gamma: float = 0.8) -> tuple:
    """Map a dominant wavelength (nm) to an approximate sRGB (0..1) triple.

    Classic piecewise approximation (Bruton). Returns fully saturated color;
    saturation/brightness are applied separately.
    """
    w = float(wavelength_nm)
    if w < 380:
        w = 380.0
    if w > 750:
        w = 750.0

    if 380 <= w < 440:
        r = -(w - 440) / (440 - 380)
        g = 0.0
        b = 1.0
    elif 440 <= w < 490:
        r = 0.0
        g = (w - 440) / (490 - 440)
        b = 1.0
    elif 490 <= w < 510:
        r = 0.0
        g = 1.0
        b = -(w - 510) / (510 - 490)
    elif 510 <= w < 580:
        r = (w - 510) / (580 - 510)
        g = 1.0
        b = 0.0
    elif 580 <= w < 645:
        r = 1.0
        g = -(w - 645) / (645 - 580)
        b = 0.0
    else:  # 645..750
        r = 1.0
        g = 0.0
        b = 0.0

    # intensity fall-off near the edges of the visible range
    if 380 <= w < 420:
        factor = 0.3 + 0.7 * (w - 380) / (420 - 380)
    elif 420 <= w < 645:
        factor = 1.0
    else:
        factor = 0.3 + 0.7 * (750 - w) / (750 - 645)

    def adj(c):
        return (c * factor) ** gamma if c > 0 else 0.0

    return (adj(r), adj(g), adj(b))


def hsv_like_to_rgb(peak_nm: float, saturation: float, value: float) -> tuple:
    """Combine a spectral hue with saturation and brightness into sRGB 0..1.

    saturation in [0,1]: 1 = pure spectral color, 0 = grey/white.
    value in [0,1]: overall brightness.
    """
    r, g, b = wavelength_to_rgb(peak_nm)
    s = float(np.clip(saturation, 0.0, 1.0))
    v = float(np.clip(value, 0.0, 1.0))
    # desaturate toward white
    r = (r * s + (1.0 - s)) * v
    g = (g * s + (1.0 - s)) * v
    b = (b * s + (1.0 - s)) * v
    return (float(np.clip(r, 0, 1)), float(np.clip(g, 0, 1)), float(np.clip(b, 0, 1)))


def rgb01_to_srgb8(rgb01) -> list:
    return [int(round(float(np.clip(c, 0, 1)) * 255)) for c in rgb01]


def srgb8_to_rgb01(srgb8) -> tuple:
    return tuple(float(c) / 255.0 for c in srgb8)


def rgb_to_hex(srgb8) -> str:
    r, g, b = [int(round(c)) for c in srgb8]
    return f"#{r:02x}{g:02x}{b:02x}"


def hex_to_srgb8(hexstr: str) -> list:
    h = hexstr.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)]


# ---------------------------------------------------------------------------
# color -> (peak_nm, saturation, value)   (needed by the inverse model)
# ---------------------------------------------------------------------------
def srgb8_to_peak_sat_val(srgb8) -> tuple:
    """Estimate dominant wavelength + saturation + brightness from sRGB.

    Dominant wavelength is found by matching the input hue against the spectral
    locus produced by wavelength_to_rgb (nearest hue in RGB-normalized space).
    Purples (magenta, no single dominant wavelength) are snapped to the closest
    spectral end. Good enough for the simulator's target-color input.
    """
    r, g, b = srgb8_to_rgb01(srgb8)
    mx = max(r, g, b)
    mn = min(r, g, b)
    value = mx
    saturation = 0.0 if mx <= 1e-6 else (mx - mn) / mx

    # Build spectral locus once and match by cosine/normalized distance.
    wls = np.linspace(VISIBLE_MIN_NM, VISIBLE_MAX_NM, 200)
    best_wl = 500.0
    best_err = 1e9
    tgt = np.array([r, g, b])
    tgt_n = tgt / (np.linalg.norm(tgt) + 1e-9)
    for wl in wls:
        cr, cg, cb = wavelength_to_rgb(float(wl))
        cvec = np.array([cr, cg, cb])
        nrm = np.linalg.norm(cvec) + 1e-9
        cvec_n = cvec / nrm
        err = float(np.sum((cvec_n - tgt_n) ** 2))
        if err < best_err:
            best_err = err
            best_wl = float(wl)
    return best_wl, float(np.clip(saturation, 0, 1)), float(np.clip(value, 0, 1))


# Convenience named targets used in tests / demos.
NAMED_COLORS = {
    "kingfisher blue": [31, 111, 176],
    "blue": [30, 90, 200],
    "green": [40, 170, 90],
    "red": [200, 50, 50],
}


if __name__ == "__main__":
    for name, srgb in NAMED_COLORS.items():
        peak, sat, val = srgb8_to_peak_sat_val(srgb)
        print(f"{name:16s} srgb={srgb} -> peak={peak:.0f}nm sat={sat:.2f} val={val:.2f} hex={rgb_to_hex(srgb)}")
    print("peak(d=180, pf=0.5) =", round(peak_from_structure(180, 0.5), 1), "nm")
    print("d(peak=470, pf=0.5) =", round(d_from_peak(470, 0.5), 1), "nm")
