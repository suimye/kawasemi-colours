#!/usr/bin/env python3
"""Answer two things:

(1) explainer.png  — an annotated diagram: WHERE the colour comes from
    (white light -> colourless keratin/air structure with spacing d ->
     interference selects lambda = 2*n_eff*d -> that colour reflects to the eye).

(2) emergent_color.png — NOT circular. The ONLY inputs are: geometry (grayscale,
    colourless), the refractive indices of keratin (1.56) and air (1.0) — both
    colourless — and WHITE light. No colour is ever painted in. We build a
    structure whose spacing d varies across the frame and let wave optics compute
    what each region reflects. A rainbow EMERGES from geometry alone.

    python experiments/explain_and_emergent.py
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "model"))
import optics as o        # noqa
import synth              # noqa

LAM = np.arange(360.0, 781.0, 2.0)
OUT_EXPL = os.path.join(ROOT, "experiments", "explainer.png")
OUT_EMERG = os.path.join(ROOT, "experiments", "emergent_color.png")


def color_of(d, pf=0.5, dis=0.10, ppu=100.0, n=6, size=320):
    """Colour reflected by a structure of spacing d — computed from GEOMETRY only.
    Inputs: the drawn geometry, keratin/air indices (in optics.n_eff), white light."""
    ne = o.n_eff(pf); nmpp = 1000.0 / ppu
    Rs, ref = [], None
    for s in range(n):
        img = synth.generate_image(d, pf, dis, 0.0, px_per_um=ppu, size=size, seed=100 + s)
        ref = img if ref is None else ref
        _, R = o.reflectance_from_image(img, nmpp, ne, lam=LAM)
        Rs.append(R)
    R = np.mean(Rs, 0)
    srgb, hexv, peak = o.spectrum_to_srgb(LAM, R)
    return srgb, R, peak, ref


# ---------------------------------------------------------------------------
def make_explainer():
    """Schematic: white light + colourless structure -> interference selects
    one wavelength -> that is the colour seen. Full explanation lives in the
    figure caption, not on the image."""
    INK = "#e6edf3"; MUT = "#aab6c2"; ACC = "#8fd3ff"; BG = "#0e1116"
    fig, ax = plt.subplots(figsize=(13.5, 6.0)); ax.axis("off")
    ax.set_xlim(0, 13.5); ax.set_ylim(0, 6.0); ax.set_facecolor(BG)

    d_demo = 185
    srgb, R, peak, ref = color_of(d_demo)
    col = np.array(srgb) / 255.0

    # white-light source (top-left)
    ax.add_patch(Circle((1.15, 4.7), 0.5, color="#fff3b0", ec="#caa63c", zorder=3))
    ax.text(1.15, 5.45, "white light", ha="center", fontsize=9.5, color=INK)
    for a in np.linspace(-0.45, 0.45, 5):
        ax.add_patch(FancyArrowPatch((1.65, 4.6), (3.75, 3.4 + a), arrowstyle="-|>",
                     mutation_scale=9, color="#e4d06a", lw=1.3, zorder=2))

    # colourless structure geometry (drawn directly in data coords)
    ax.imshow(ref, extent=[3.85, 6.35, 1.7, 4.6], cmap="gray", aspect="auto", zorder=1)
    ax.add_patch(Rectangle((3.85, 1.7), 2.5, 2.9, fill=False, ec="#3a444f", lw=1))
    ax.text(5.10, 4.75, "colourless structure", ha="center", fontsize=10, color=INK)
    ax.text(5.10, 1.48, "n$_{keratin}$=1.56, n$_{air}$=1.0", ha="center", fontsize=8.5, color=MUT)
    # spacing d annotation
    ax.annotate("", xy=(4.7, 2.85), xytext=(5.25, 2.85),
                arrowprops=dict(arrowstyle="<->", color="#ffe08a", lw=1.8))
    ax.text(4.97, 3.02, "d", fontsize=12, color="#ffe08a", ha="center", weight="bold")

    # interference relation
    ax.text(6.7, 4.55, "constructive interference:", fontsize=10.5, color=INK)
    ax.text(6.7, 3.95, r"$\lambda = 2\, n_{\mathrm{eff}}\, d$", fontsize=17, color=ACC)

    # outgoing coloured beam to the eye
    for a in np.linspace(-0.28, 0.28, 5):
        ax.add_patch(FancyArrowPatch((6.4, 3.2), (10.05, 4.7 + a), arrowstyle="-|>",
                     mutation_scale=13, color=col, lw=2.4, zorder=2))
    ax.add_patch(Circle((10.75, 4.85), 0.42, color="w", ec="k", zorder=3))
    ax.add_patch(Circle((10.75, 4.85), 0.17, color="#333", zorder=4))
    ax.text(10.75, 5.45, "observed colour", ha="center", fontsize=9.5, color=INK)

    # reflectance spectrum from the geometry (bottom-right)
    sp = fig.add_axes([0.60, 0.08, 0.345, 0.30]); sp.set_facecolor("#0b0f14")
    sp.plot(LAM, R, color=INK, lw=1); sp.fill_between(LAM, R, color=col, alpha=0.85)
    sp.set_xlim(360, 780); sp.set_ylim(0, 1.05); sp.set_yticks([])
    sp.set_xlabel("wavelength (nm)", fontsize=8, color=MUT)
    sp.set_title(f"reflectance, peak {peak:.0f} nm", fontsize=9, color=INK)
    sp.tick_params(labelsize=7, colors=MUT)
    for s in sp.spines.values(): s.set_color("#2a323d")

    fig.savefig(OUT_EXPL, dpi=150, facecolor=BG, bbox_inches="tight")
    fig.savefig(OUT_EXPL.replace(".png", ".pdf"), facecolor=BG, bbox_inches="tight")
    print("saved", OUT_EXPL, "(and .pdf)")


# ---------------------------------------------------------------------------
def make_emergent():
    """Spacing varies across the frame; colour is computed per-column from the
    geometry alone. A rainbow emerges without ever painting a colour."""
    ds = np.linspace(150, 285, 11).astype(int)
    strip_w = 70; H = 360
    gray_cols, color_cols, peaks = [], [], []
    for d in ds:
        srgb, R, peak, ref = color_of(int(d))
        g = cv2.resize(ref, (strip_w, H))
        gray_cols.append(g)
        color_cols.append(np.tile(np.array([srgb[2], srgb[1], srgb[0]], np.uint8), (H, strip_w, 1)))
        peaks.append(peak)
    gray = np.hstack(gray_cols)
    colr = cv2.cvtColor(np.hstack(color_cols), cv2.COLOR_BGR2RGB)

    fig = plt.figure(figsize=(13.5, 7.6))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 3, 2.4], hspace=0.35)

    ax0 = fig.add_subplot(gs[0]); ax0.imshow(gray, cmap="gray"); ax0.axis("off")
    ax0.set_title("Input: colourless structure (spacing d increases →)", fontsize=10, loc="left")

    ax1 = fig.add_subplot(gs[1]); ax1.imshow(colr, aspect="auto",
                                             extent=[0, gray.shape[1], 0, H]); ax1.axis("off")
    ax1.set_title("Output: computed colour (no colour was painted in)", fontsize=10, loc="left")
    for i, d in enumerate(ds):
        ax1.text((i + 0.5) * strip_w, -18, f"{d}nm\n{peaks[i]:.0f}nm", ha="center", va="top", fontsize=7.5)

    # a few reflectance spectra to show the peak sliding
    ax2 = fig.add_subplot(gs[2])
    for i in [0, 3, 6, 9]:
        srgb, R, peak, _ = color_of(int(ds[i]))
        ax2.plot(LAM, R, color=np.array(srgb) / 255.0, lw=2, label=f"d={ds[i]}nm → {peak:.0f}nm")
    ax2.set_xlim(360, 780); ax2.set_yticks([]); ax2.set_xlabel("wavelength (nm)", fontsize=9)
    ax2.legend(fontsize=8, ncol=4, loc="upper center"); ax2.set_title("reflectance spectra", fontsize=10, loc="left")
    fig.savefig(OUT_EMERG, dpi=150, bbox_inches="tight")
    fig.savefig(OUT_EMERG.replace(".png", ".pdf"), bbox_inches="tight")
    print("saved", OUT_EMERG, "(and .pdf)")


if __name__ == "__main__":
    make_explainer()
    make_emergent()
