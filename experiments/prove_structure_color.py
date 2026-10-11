#!/usr/bin/env python3
"""IN-SILICO DEMONSTRATION: generate a nanostructure -> estimate reflectance
with a coherent single-scattering structure-factor model -> CIE colour. No
machine learning or fitted target-colour parameters are used. Shows:

  A) a drawn spongy structure with characteristic spacing d produces a colour
     whose dominant wavelength generally shifts from blue toward red as d grows;
  B) displayed bird SEM/TEM images provide context for synthetic, literature-
     informed keratin–air candidate structures; the optics is evaluated on the
     latter, not on the pixels of the displayed micrographs.

Outputs a figure and a comparison table.
  python experiments/prove_structure_color.py
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "model"))
import optics as o          # noqa
import synth                # noqa
from colorutil import srgb8_to_peak_sat_val  # noqa

LAM = np.arange(360.0, 781.0, 2.0)
OUT = os.environ.get("KAWASEMI_FIG_OUT", os.path.join(ROOT, "experiments", "proof_structure_color.png"))
os.makedirs(os.path.dirname(OUT), exist_ok=True)


def hexf(u8): return "#{:02x}{:02x}{:02x}".format(*u8)


def ensemble_reflectance(d_nm, pf, disorder, ppu, n=6, size=384):
    """Mean reflectance over several realisations of the SAME parameters."""
    ne = o.n_eff(pf); nmpp = 1000.0 / ppu
    Rs = []
    imgs = []
    for s in range(n):
        img = synth.generate_image(d_nm, pf, disorder, 0.0, px_per_um=ppu, size=size, seed=100 + s)
        _, R = o.reflectance_from_image(img, nmpp, ne, lam=LAM)
        Rs.append(R); imgs.append(img)
    R = np.mean(Rs, 0); R = R / max(R.max(), 1e-9)
    return imgs[0], R


def real_reflectance(img, nm_per_px, n_eff_val):
    _, R = o.reflectance_from_image(img, nm_per_px, n_eff_val, lam=LAM)
    return R


def dominant_period_px(img, band=(4.0, 26.0)):
    """Find the quasi-ordered RING: the strongest *local maximum* of S(q) within a
    plausible band (not the monotonic low-frequency pedestal).

    Returns (period_px, quality). quality = peak height above the local trend;
    a clean ring gives quality well above 1, a pedestal-only spectrum ~1.
    """
    _, S = o.radial_structure_factor(img, 1.0)
    S = S.copy(); S[:3] = 0.0
    S = np.convolve(S, np.ones(3) / 3.0, mode="same")     # light smoothing
    N = img.shape[0]
    kpx = np.arange(1, len(S) - 1)
    with np.errstate(divide="ignore"):
        period = N / kpx
    inband = (period >= band[0]) & (period <= band[1])
    localmax = (S[kpx] > S[kpx - 1]) & (S[kpx] > S[kpx + 1])
    cand = kpx[inband & localmax]
    if len(cand) == 0:
        return None, 0.0
    kk = cand[np.argmax(S[cand])]
    # quality: peak vs. the minimum of S between DC-ish and the peak (ring contrast)
    base = np.median(S[max(1, kk - 6):kk + 7])
    quality = float(S[kk] / (base + 1e-9))
    return N / kk, quality


def load_real(rid):
    for line in open(os.path.join(ROOT, "data", "dataset.jsonl")):
        o_ = json.loads(line)
        if o_["id"] == rid:
            p = os.path.join(ROOT, "data", o_["image_file"])
            return cv2.imread(p, cv2.IMREAD_GRAYSCALE), o_
    return None, None


def main():
    rows = []   # (title, image, R, predicted_srgb, subtitle, truth_srgb or None)

    # ---- A) drawn structures: known d, keratin/air, low disorder ----
    print("=== A) drawn structures (pure optics, no ML) ===")
    pf = 0.5
    for d in [140, 165, 190, 225, 270]:
        img, R = ensemble_reflectance(d, pf, 0.10, ppu=100.0)
        srgb, hx, dom = o.spectrum_to_srgb(LAM, R)
        bragg = o.bragg_peak_nm(d, pf)
        print(f"  d={d}nm  optical_peak={dom:.0f}nm  2*n_eff*d={bragg:.0f}nm  {hx}")
        rows.append((f"drawn d={d}nm", img, R, srgb,
                     f"peak {dom:.0f}nm  (2n·d={bragg:.0f})", None))

    A_rows = list(rows)  # keep drawn sweep separate

    # ---- B) literature-informed candidate structures: draw a keratin-air sponge
    #        at a declared representative spacing and compare its simulated colour
    #        with the record target.  The SEM is contextual only: this is not an
    #        optical prediction computed from its pixels. ----
    print("\n=== B) literature-informed candidate structure -> optics colour vs record target ===")
    # Representative spacings / pore fraction are declared illustrative priors,
    # not fitted reconstructions of the displayed image.
    # DISPLAY POLICY: only records with confirmed display permission are shown.
    # Prum et al. (2009) remains available to model development but is not reused
    # as a displayed image pending licence confirmation.
    birds = [
        ("PMC4685390_fig6c_jay",        178.0, 0.50, "Eurasian jay\nGarrulus glandarius"),
        ("BIORXIV_tersina_fig6c",       200.0, 0.50, "swallow tanager\nTersina viridis"),
        ("BIORXIV_tersina_fig6b_inset", 190.0, 0.50, "swallow tanager (inset)\nTersina viridis"),
    ]
    B = []
    for rid, d, pf, label in birds:
        real_img, rec = load_real(rid)
        drawn_img, R = ensemble_reflectance(d, pf, 0.10, ppu=100.0)
        srgb, hx, dom = o.spectrum_to_srgb(LAM, R)
        tsr = rec["color"]["srgb"]; tname = rec["color"]["name"]; tpk = rec["color"].get("peak_nm")
        err = abs(dom - tpk) if tpk else None
        print(f"  {label.splitlines()[0]:16} d={d:.0f}nm -> optics {dom:.0f}nm {hx}  "
              f"target {tname}({tpk}nm)" + (f"  |difference|={err:.0f}nm" if err else ""))
        B.append((label, real_img, drawn_img, R, srgb, tsr, f"d={d:.0f}nm → {dom:.0f}nm"))

    # ---------------- figure ----------------
    fig = plt.figure(figsize=(15.5, 12.4))
    gs = fig.add_gridspec(4, 5, height_ratios=[3, 2, 3, 2], hspace=0.45, wspace=0.18)

    # Section A: drawn sweep
    for i, (title, img, R, srgb, sub, _t) in enumerate(A_rows):
        col = np.array(srgb) / 255.0
        axi = fig.add_subplot(gs[0, i]); axi.imshow(img, cmap="gray"); axi.axis("off")
        axi.set_title(("A   " if i == 0 else "") + title, fontsize=9, loc="left")
        for sp in axi.spines.values():
            sp.set_visible(True); sp.set_color(col); sp.set_linewidth(7)
        axs = fig.add_subplot(gs[1, i])
        axs.plot(LAM, R, "k", lw=1); axs.fill_between(LAM, R, color=col, alpha=0.85)
        axs.set_xlim(360, 780); axs.set_ylim(0, 1.05); axs.set_yticks([]); axs.tick_params(labelsize=7)
        axs.set_title(sub, fontsize=8); axs.set_xlabel("nm", fontsize=7)

    # Section B: real vs drawn-at-published-spacing
    for i, (label, real_img, drawn_img, R, srgb, tsr, sub) in enumerate(B):
        col = np.array(srgb) / 255.0
        axr = fig.add_subplot(gs[2, i])
        if real_img is not None:
            axr.imshow(real_img, cmap="gray")
        axr.axis("off")
        axr.set_title(("B   " if i == 0 else "") + label, fontsize=8.5, loc="left")
        axs = fig.add_subplot(gs[3, i])
        axs.plot(LAM, R, "k", lw=1); axs.fill_between(LAM, R, color=col, alpha=0.85)
        axs.set_xlim(360, 780); axs.set_ylim(0, 1.05); axs.set_yticks([]); axs.tick_params(labelsize=7)
        axs.set_title(sub, fontsize=8); axs.set_xlabel("nm", fontsize=7)
        # Candidate-structure swatch and a non-quantitative display swatch.
        axs.add_patch(plt.Rectangle((0.06, 0.60), 0.26, 0.34, transform=axs.transAxes, color=col, ec="k"))
        axs.text(0.19, 0.955, "optics", transform=axs.transAxes, fontsize=6, ha="center", va="bottom")
        axs.add_patch(plt.Rectangle((0.68, 0.60), 0.26, 0.34, transform=axs.transAxes,
                                    color=np.array(tsr) / 255.0, ec="k"))
        axs.text(0.81, 0.955, "display swatch*", transform=axs.transAxes, fontsize=6, ha="center", va="bottom")
        axs.text(0.81, 0.52, "*not spectrally calibrated", transform=axs.transAxes, fontsize=5.2, ha="center", va="top")

    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    fig.savefig(OUT.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"\nsaved figure -> {OUT} (and .pdf)")


if __name__ == "__main__":
    main()
