#!/usr/bin/env python3
"""3-D in-silico demonstration using a generated keratin/air voxel structure.

The script computes a structure-factor spectrum without using a target colour
as input, then converts that spectrum to a display colour. This is a model
self-consistency demonstration, not independent experimental validation.

The 3-D structure is rendered in neutral grey (it is colourless); the colour is
shown only as the computed swatch + spectrum — nothing is painted onto the
structure.

  python experiments/proof_3d.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "model"))
import optics as o          # noqa
import structure3d as s3    # noqa

OUT = os.path.join(ROOT, "experiments", "proof_3d.png")
NMV = 20.0                  # nm per voxel
PF = 0.5


def surface_points(vol):
    """Voxels of the keratin phase that touch air (the visible surface)."""
    v = vol.astype(bool)
    nb = np.zeros_like(v)
    for ax in range(3):
        nb |= np.roll(v, 1, ax) & ~v
        nb |= np.roll(v, -1, ax) & ~v
    surf = v & (
        ~np.roll(v, 1, 0) | ~np.roll(v, -1, 0) |
        ~np.roll(v, 1, 1) | ~np.roll(v, -1, 1) |
        ~np.roll(v, 1, 2) | ~np.roll(v, -1, 2))
    return np.argwhere(surf)


def main():
    ds = [160, 200, 250]
    # colour from a well-sampled volume; render from a smaller one
    results = []
    for d in ds:
        vol_hi = s3.generate_volume(d, PF, 0.12, nm_per_vox=NMV, size=72, seed=1)
        srgb, hexv, peak, lam, R = s3.colour_3d(vol_hi, NMV, PF)
        vol_lo = s3.generate_volume(d, PF, 0.12, nm_per_vox=NMV, size=40, seed=1)
        results.append((d, srgb, peak, lam, R, vol_lo, vol_hi))
        print(f"  3D d={d}nm -> peak {peak:.0f}nm  2*n*d={o.bragg_peak_nm(d,PF):.0f}nm  {hexv}")

    fig = plt.figure(figsize=(13.5, 11))
    gs = fig.add_gridspec(3, 3, height_ratios=[3.2, 2.2, 2.0], hspace=0.33, wspace=0.22)

    for i, (d, srgb, peak, lam, R, vol_lo, vol_hi) in enumerate(results):
        col = np.array(srgb) / 255.0
        # 3-D structure (neutral grey; it is colourless)
        ax = fig.add_subplot(gs[0, i], projection="3d")
        pts = surface_points(vol_lo)
        depth = pts[:, 2]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=depth, cmap="Greys",
                   s=6, marker="s", alpha=0.5, linewidths=0)
        ax.set_title(f"d = {d} nm", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=22, azim=35)

        # mid slice (grey) + emergent swatch
        axs = fig.add_subplot(gs[1, i])
        sl = vol_hi[:, :, vol_hi.shape[2] // 2]
        axs.imshow(sl, cmap="gray"); axs.axis("off")
        axs.set_title("mid-plane slice", fontsize=8.5)
        axs.add_patch(plt.Rectangle((0.80, 0.80), 0.18, 0.18, transform=axs.transAxes,
                                    color=col, ec="k"))
        axs.text(0.79, 0.79, "colour", transform=axs.transAxes, fontsize=7, ha="right", va="top")

        # spectrum
        axr = fig.add_subplot(gs[2, i])
        axr.plot(lam, R, "k", lw=1); axr.fill_between(lam, R, color=col, alpha=0.85)
        axr.set_xlim(360, 780); axr.set_ylim(0, 1.05); axr.set_yticks([])
        axr.set_xlabel("nm", fontsize=8); axr.tick_params(labelsize=7)
        axr.set_title(f"reflectance, peak {peak:.0f} nm", fontsize=8.5)

    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    fig.savefig(OUT.replace(".png", ".pdf"), bbox_inches="tight")
    print("saved", OUT, "(and .pdf)")


if __name__ == "__main__":
    main()
