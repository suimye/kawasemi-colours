"""3-D quasi-ordered nanostructure (keratin/air voxel volume) + 3-D wave-optics
colour. This removes the 2-D approximation: interference path lengths are now
computed over a real 3-D structure via its 3-D structure factor.

  generate_volume(d, pf, disorder, nm_per_vox, size) -> binary volume (1=keratin)
  reflectance_3d(volume, nm_per_vox, n_eff) -> (lam, R)   [coherent backscatter]
  colour comes from optics.spectrum_to_srgb (CIE).

Backscatter relation is the same in 3-D:  q_b = 4*pi*n_eff/lam, and the 3-D
structure factor peaks at |q*| = 2*pi/d  ->  lambda_peak = 2*n_eff*d.

numpy-only (no scipy/skimage). Torch-free.
"""
from __future__ import annotations
import numpy as np
import optics as o


def generate_volume(d_nm, pore_fraction=0.5, disorder=0.12,
                    nm_per_vox=20.0, size=72, n_waves=40, seed=0):
    """Isotropic quasi-ordered keratin/air volume with characteristic spacing d.

    Built as a sum of 3-D plane waves whose spatial frequency is ~1/d (disorder
    spreads it), then thresholded to the desired pore fraction. 1 = keratin, 0 = air.
    """
    rng = np.random.default_rng(seed)
    period_vox = max(3.0, d_nm / nm_per_vox)
    f0 = 1.0 / period_vox
    ax = np.arange(size)
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    field = np.zeros((size, size, size), np.float32)
    for _ in range(n_waves):
        # random direction on the unit sphere
        v = rng.normal(size=3); v /= np.linalg.norm(v) + 1e-9
        f = f0 * (1.0 + (rng.random() * 2 - 1) * disorder * 0.9)
        k = 2 * np.pi * f * v
        ph = rng.uniform(0, 2 * np.pi)
        field += np.cos(k[0] * X + k[1] * Y + k[2] * Z + ph)
    field -= field.mean(); field /= field.std() + 1e-9
    thr = np.quantile(field, np.clip(pore_fraction, 0.05, 0.95))
    keratin = (field >= thr).astype(np.uint8)   # bright/dense -> keratin
    return keratin


def structure_factor_3d(volume, nm_per_vox):
    """Radially-averaged 3-D structure factor. Returns (q [rad/nm], S(q))."""
    g = volume.astype(np.float32)
    g = g - g.mean()
    # 3-D Hann window
    w1 = np.hanning(g.shape[0])
    w = w1[:, None, None] * w1[None, :, None] * w1[None, None, :]
    F = np.fft.fftshift(np.fft.fftn(g * w))
    P = np.abs(F) ** 2
    N = g.shape[0]
    c = N // 2
    ax = np.arange(N) - c
    R = np.sqrt(ax[:, None, None] ** 2 + ax[None, :, None] ** 2 + ax[None, None, :] ** 2)
    ri = R.astype(int).ravel()
    S = np.bincount(ri, P.ravel()) / np.maximum(np.bincount(ri), 1)
    k = np.arange(len(S))
    f_cyc_per_vox = k / N
    q = 2 * np.pi * (f_cyc_per_vox / nm_per_vox)
    return q, S


def reflectance_3d(volume, nm_per_vox, n_eff_val,
                   lam=np.arange(360.0, 781.0, 2.0), melanin_absorption=True):
    q, S = structure_factor_3d(volume, nm_per_vox)
    S = S.copy(); S[:2] = 0.0
    qb = 4.0 * np.pi * n_eff_val / lam
    R = np.interp(qb, q, S, left=0.0, right=0.0)
    if melanin_absorption and R.max() > 0:
        R = np.clip(R - np.percentile(R[R > 0], 25), 0.0, None)
    if R.max() > 0:
        R = R / R.max()
    return lam, R


def colour_3d(volume, nm_per_vox, pore_fraction=0.5):
    ne = o.n_eff(pore_fraction)
    lam, R = reflectance_3d(volume, nm_per_vox, ne)
    srgb, hexv, peak = o.spectrum_to_srgb(lam, R)
    return srgb, hexv, peak, lam, R
