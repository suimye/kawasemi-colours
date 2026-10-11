"""Physics-based forward model: nanostructure image -> spectrum -> colour.

No machine learning or fitted target-colour parameters are used in this route.
It implements a coherent single-scattering approximation used for quasi-ordered
("amorphous photonic") structures, where back-scattered reflectance is taken as
proportional to the Fourier power (structure factor) S(q) evaluated at
q_b = 4*pi*n_eff / lambda (Prum & Torres; Noh et al. 2010; Dufresne et al.).

Display colour is derived from the spectrum with the CIE 1931 2-deg colour
matching functions (Wyman-Sloan-Shirley 2013 analytic approximation) under a
D65-like (6504 K Planckian) illuminant -> XYZ -> sRGB.

Also provides a 1-D transfer-matrix (TMM) reflectance for MULTILAYER
(melanosome / thin-film, i.e. iridescent) structures, the other structural-colour
regime.

Torch-free: numpy (+ cv2 only for reading images in the demo).
"""
from __future__ import annotations
import numpy as np

N_KERATIN, N_AIR = 1.56, 1.0
def n_eff(pore_fraction: float) -> float:
    pf = float(np.clip(pore_fraction, 0.0, 1.0))
    return pf * N_AIR + (1.0 - pf) * N_KERATIN


# ---------------------------------------------------------------------------
# CIE 1931 colour matching functions (Wyman, Sloan, Shirley 2013, JCGT)
# ---------------------------------------------------------------------------
def _g(x, mu, s1, s2):
    s = np.where(x < mu, s1, s2)
    return np.exp(-0.5 * ((x - mu) / s) ** 2)

def cie_xyz_bar(lam):
    xb = 1.056 * _g(lam, 599.8, 37.9, 31.0) + 0.362 * _g(lam, 442.0, 16.0, 26.7) \
        - 0.065 * _g(lam, 501.1, 20.4, 26.2)
    yb = 0.821 * _g(lam, 568.8, 46.9, 40.5) + 0.286 * _g(lam, 530.9, 16.3, 31.1)
    zb = 1.217 * _g(lam, 437.0, 11.8, 36.0) + 0.681 * _g(lam, 459.0, 26.0, 13.8)
    return xb, yb, zb

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

def _planck(lam_nm, T=6504.0):
    h, c, kb = 6.62607015e-34, 2.99792458e8, 1.380649e-23
    l = lam_nm * 1e-9
    return (1.0 / l ** 5) / (np.exp(h * c / (l * kb * T)) - 1.0)

def spectrum_to_srgb(lam, R, illuminant_T=6504.0):
    """Reflectance spectrum R(lam) (0..1) -> (sRGB uint8, hex, dominant_nm)."""
    lam = np.asarray(lam, float); R = np.asarray(R, float)
    I = _planck(lam, illuminant_T); I = I / I.max()
    xb, yb, zb = cie_xyz_bar(lam)
    k = 1.0 / _trapz(yb * I, lam)
    X = k * _trapz(R * I * xb, lam)
    Y = k * _trapz(R * I * yb, lam)
    Z = k * _trapz(R * I * zb, lam)
    M = np.array([[3.2406, -1.5372, -0.4986],
                  [-0.9689, 1.8758, 0.0415],
                  [0.0557, -0.2040, 1.0570]])
    rgb = M @ np.array([X, Y, Z])
    rgb = np.clip(rgb, 0.0, None)
    if rgb.max() > 1.0:
        rgb = rgb / rgb.max()
    a = 0.055
    srgb = np.where(rgb <= 0.0031308, 12.92 * rgb, (1 + a) * rgb ** (1 / 2.4) - a)
    srgb = np.clip(srgb, 0, 1)
    u8 = [int(round(v * 255)) for v in srgb]
    dom = float(lam[int(np.argmax(R))])
    return u8, "#{:02x}{:02x}{:02x}".format(*u8), dom


# ---------------------------------------------------------------------------
# Structure factor of an image  -> reflectance (quasi-ordered / spongy regime)
# ---------------------------------------------------------------------------
def radial_structure_factor(image, nm_per_px):
    """Return (q [rad/nm], S(q)) from the 2D power spectrum of an image.

    The grayscale image is treated as a refractive-index-contrast map. A Hann
    window suppresses FFT edge artefacts; the power spectrum is radially averaged.
    """
    g = np.asarray(image, float)
    g = g - g.mean()
    wy = np.hanning(g.shape[0])[:, None]
    wx = np.hanning(g.shape[1])[None, :]
    F = np.fft.fftshift(np.fft.fft2(g * (wy * wx)))
    P = np.abs(F) ** 2
    cy, cx = (np.array(P.shape) // 2)
    Y, X = np.indices(P.shape)
    r = np.hypot(X - cx, Y - cy).astype(int)
    tot = np.bincount(r.ravel(), P.ravel())
    cnt = np.bincount(r.ravel())
    S = tot / np.maximum(cnt, 1)
    N = g.shape[0]
    k_px = np.arange(len(S))
    f_cyc_per_px = k_px / N
    f_cyc_per_nm = f_cyc_per_px / nm_per_px
    q = 2.0 * np.pi * f_cyc_per_nm          # rad/nm
    return q, S

def reflectance_from_image(image, nm_per_px, n_eff_val,
                           lam=np.arange(360.0, 781.0, 2.0),
                           melanin_absorption=True):
    """Coherent single-scattering reflectance R(lam) from a structure image.

    R(lam) ~ S(q_b),  q_b = 4*pi*n_eff/lam  (normal-incidence backscatter).

    melanin_absorption: real feathers place melanin behind/among the structure
    which absorbs the broadband *incoherently* scattered light, leaving the
    coherent peak and thus saturating the colour (Noh et al. 2010). We model this
    by subtracting the diffuse baseline of S(q) over the visible band.
    """
    q, S = radial_structure_factor(image, nm_per_px)
    S = S.copy(); S[:3] = 0.0                      # drop DC / large-scale gradient
    qb = 4.0 * np.pi * n_eff_val / lam
    R = np.interp(qb, q, S, left=0.0, right=0.0)
    if melanin_absorption and R.max() > 0:
        floor = np.percentile(R[R > 0], 25)        # broadband diffuse ~ absorbed
        R = np.clip(R - floor, 0.0, None)
    if R.max() > 0:
        R = R / R.max()
    return lam, R

def bragg_peak_nm(char_length_nm, pore_fraction):
    """First-order backscatter peak  lambda = 2 * n_eff * d."""
    return 2.0 * n_eff(pore_fraction) * char_length_nm


# ---------------------------------------------------------------------------
# Transfer-matrix reflectance for a 1-D multilayer (iridescent regime)
# ---------------------------------------------------------------------------
def tmm_reflectance(n_layers, d_layers, lam, n0=1.0, n_sub=1.4, theta_deg=0.0):
    """Unpolarized reflectance of a 1-D stack (lists n_layers, d_layers[nm]).

    n0 = incident medium, n_sub = substrate. Uses the standard 2x2 characteristic
    matrix method; averages s- and p-polarizations.
    """
    lam = np.asarray(lam, float)
    th0 = np.deg2rad(theta_deg)
    def pol(pol_s):
        r = np.zeros_like(lam, dtype=complex)
        for i, L in enumerate(lam):
            # angles via Snell (complex-safe)
            n_seq = [n0] + list(n_layers) + [n_sub]
            sin0 = n0 * np.sin(th0)
            costh = [np.sqrt(1 - (sin0 / n) ** 2 + 0j) for n in n_seq]
            M = np.eye(2, dtype=complex)
            for j, (n, d) in enumerate(zip(n_layers, d_layers)):
                ct = costh[j + 1]
                delta = 2 * np.pi * n * d * ct / L
                eta = (n * ct) if pol_s else (n / ct)
                m = np.array([[np.cos(delta), 1j * np.sin(delta) / eta],
                              [1j * eta * np.sin(delta), np.cos(delta)]])
                M = M @ m
            eta0 = (n0 * costh[0]) if pol_s else (n0 / costh[0])
            etas = (n_sub * costh[-1]) if pol_s else (n_sub / costh[-1])
            B, C = M @ np.array([1.0, etas], dtype=complex)
            rr = (eta0 * B - C) / (eta0 * B + C)
            r[i] = rr
        return np.abs(r) ** 2
    return 0.5 * (pol(True) + pol(False))
