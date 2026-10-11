"""Render nanostructures so a human EYE sees the structural colour.

Two views:
  1. colorize_structure(gray, srgb): paint the actual SEM/TEM nanostructure in
     the colour it physically produces (topography -> brightness, structural
     colour -> hue/saturation). You literally see the (colourless-looking)
     structure in its real colour.
  2. render_patch(srgb): a macroscopic material swatch (soft domain mottling +
     a specular highlight) — what a patch of that material looks like to the eye.

Plus iridescence: multilayer_angle_colors() sweeps the viewing angle through the
transfer-matrix model so you can SEE an iridescent stack shift colour with angle,
versus a spongy structure that stays the same colour.

Colour always comes from optics.py (wave optics -> CIE). Torch-free.
"""
from __future__ import annotations
import numpy as np
import cv2

import optics as o
import synth


def _srgb_to_hsv(srgb):
    b = np.uint8([[[srgb[2], srgb[1], srgb[0]]]])  # cv2 uses BGR
    h, s, v = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)[0, 0]
    return float(h), float(s), float(v)


def colorize_structure(gray, srgb, keep_contrast=0.85):
    """Paint a grayscale nanostructure image in its structural colour.

    Uses the structural colour's hue & saturation, and the image's own
    brightness as Value -> you see the structure lit in its true colour.
    """
    h, s, _ = _srgb_to_hsv(srgb)
    g = gray.astype(np.float32)
    g = (g - g.min()) / (np.ptp(g) + 1e-6)
    # compress contrast a bit so the colour reads well, keep some topography
    v = (0.25 + keep_contrast * g) * 255.0
    v = np.clip(v, 0, 255)
    hsv = np.zeros((*gray.shape, 3), np.uint8)
    hsv[..., 0] = int(h)
    hsv[..., 1] = int(np.clip(s, 60, 255))
    hsv[..., 2] = v.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def render_patch(srgb, size=220, gloss=0.6, seed=0):
    """A macroscopic material swatch showing the structural colour to the eye."""
    rng = np.random.default_rng(seed)
    col = np.array([srgb[2], srgb[1], srgb[0]], np.float32)  # BGR
    img = np.ones((size, size, 3), np.float32) * col
    # soft low-frequency domain mottling (brightness only)
    n = rng.normal(0, 1, (size // 16, size // 16)).astype(np.float32)
    n = cv2.resize(n, (size, size), interpolation=cv2.INTER_CUBIC)
    n = cv2.GaussianBlur(n, (0, 0), size / 24.0)
    img *= (1.0 + 0.14 * (n / (np.abs(n).max() + 1e-6)))[..., None]
    # specular highlight (glossy structural colour is often shiny)
    yy, xx = np.mgrid[0:size, 0:size]
    hx, hy = size * 0.34, size * 0.30
    d2 = ((xx - hx) ** 2 + (yy - hy) ** 2) / (size * 0.5) ** 2
    spec = np.exp(-d2 * 3.5) * (255.0 * gloss)
    img = np.clip(img + spec[..., None], 0, 255)
    # rounded-rect vignette so it reads as a physical sample
    mask = np.zeros((size, size), np.float32)
    cv2.rectangle(mask, (int(size*0.06), int(size*0.06)),
                  (int(size*0.94), int(size*0.94)), 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), size / 22.0)
    img = img * mask[..., None] + 245.0 * (1 - mask[..., None])
    return img.astype(np.uint8)


def structure_color(d_nm, pore_fraction=0.5, disorder=0.10, ppu=100.0, n=6, size=320):
    """Wave-optics structural colour (sRGB) for a spongy structure with spacing d."""
    ne = o.n_eff(pore_fraction); nmpp = 1000.0 / ppu
    Rs = []
    ref_img = None
    for s in range(n):
        img = synth.generate_image(d_nm, pore_fraction, disorder, 0.0,
                                    px_per_um=ppu, size=size, seed=100 + s)
        if ref_img is None:
            ref_img = img
        _, R = o.reflectance_from_image(img, nmpp, ne)
        Rs.append(R)
    R = np.mean(Rs, 0)
    lam = np.arange(360.0, 781.0, 2.0)
    srgb, hexv, peak = o.spectrum_to_srgb(lam, R)
    return srgb, hexv, peak, ref_img


def multilayer_angle_colors(d_layer=70.0, n_hi=2.0, n_lo=1.40, n_layers=8,
                            angles=(0, 15, 30, 45, 60)):
    """Iridescent multilayer (e.g. melanin/keratin): colour vs viewing angle.

    Returns list of (angle_deg, srgb, hex, peak_nm). Colour blue-shifts with
    angle -> that shift IS iridescence (contrast with a spongy structure, which
    returns a fixed colour at all angles)."""
    lam = np.arange(360.0, 781.0, 2.0)
    ns = ([n_hi, n_lo] * n_layers)[:2 * n_layers]
    ds = ([d_layer, d_layer] * n_layers)[:2 * n_layers]
    out = []
    for a in angles:
        R = o.tmm_reflectance(ns, ds, lam, n0=1.0, n_sub=1.5, theta_deg=a)
        srgb, hexv, peak = o.spectrum_to_srgb(lam, R)
        out.append((a, srgb, hexv, peak))
    return out
