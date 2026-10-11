#!/usr/bin/env python3
"""SEE the structural colour: render the (colourless-looking) nanostructure in the
colour it physically produces, so a human eye sees it.

For a spacing sweep and for the real bird SEMs:
  [ grayscale structure ]  ->  [ same structure painted in its structural colour ]
  ->  [ a macroscopic material swatch of that colour ]

Colour is computed by wave optics (optics.py); rendering by render.py. No ML.
Also writes an iridescence strip (multilayer colour vs viewing angle).

  python experiments/see_the_color.py
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
import render as rd         # noqa

OUT = os.path.join(ROOT, "experiments", "see_the_color.png")
OUT2 = os.path.join(ROOT, "experiments", "see_the_color_iridescence.png")
def bgr2rgb(im): return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def load_real(rid):
    for line in open(os.path.join(ROOT, "data", "dataset.jsonl")):
        r = json.loads(line)
        if r["id"] == rid:
            return cv2.imread(os.path.join(ROOT, "data", r["image_file"]), cv2.IMREAD_GRAYSCALE), r
    return None, None


def main():
    # ---------- spacing sweep ----------
    sweep = [140, 170, 200, 235, 275]
    sweep_data = []
    for d in sweep:
        srgb, hexv, peak, ref = rd.structure_color(d, 0.5, 0.10)
        col = rd.colorize_structure(ref, srgb)
        patch = rd.render_patch(srgb)
        sweep_data.append((d, peak, ref, col, patch))
        print(f"  drawn d={d}nm -> {hexv} ({peak:.0f}nm)")

    # ---------- real birds ----------
    # DISPLAY POLICY: CC-BY images only (Methods §2.2).
    birds = [
        ("PMC4685390_fig6c_jay",        178, "Eurasian jay"),
        ("BIORXIV_tersina_fig6c",       200, "swallow tanager"),
        ("BIORXIV_tersina_fig6b_inset", 190, "swallow tanager (inset)"),
        ("PMC2706469_fig1a_channel",    185, "channel β-keratin (Prum)"),
        ("PMC2706469_fig1c_sphere",     180, "sphere β-keratin (Prum)"),
    ]
    bird_data = []
    for rid, d, name in birds:
        real, rec = load_real(rid)
        srgb, hexv, peak, _ = rd.structure_color(d, 0.5, 0.10)
        col = rd.colorize_structure(real, srgb) if real is not None else None
        patch = rd.render_patch(srgb)
        bird_data.append((name, d, real, col, patch, rec["color"]["srgb"]))
        print(f"  {name:16} d={d}nm -> {hexv} ({peak:.0f}nm)")

    # ---------- figure ----------
    ncol = 5
    fig = plt.figure(figsize=(15.5, 15.0))
    gs = fig.add_gridspec(5, ncol, hspace=0.28, wspace=0.12)

    # rows 0-1: sweep (colourised structure, patch)
    for i, (d, peak, ref, col, patch) in enumerate(sweep_data):
        ax = fig.add_subplot(gs[0, i]); ax.imshow(bgr2rgb(col)); ax.axis("off")
        ax.set_title(("structure, in colour\n" if i == 0 else "\n") + f"d={d}nm  ({peak:.0f}nm)", fontsize=9)
        ax = fig.add_subplot(gs[1, i]); ax.imshow(bgr2rgb(patch)); ax.axis("off")
        if i == 0: ax.set_title("material swatch", fontsize=9, loc="left")

    # rows 2-4: real SEM, colourised, patch
    for i, (name, d, real, col, patch, plumage) in enumerate(bird_data):
        ax = fig.add_subplot(gs[2, i])
        if real is not None: ax.imshow(real, cmap="gray")
        ax.axis("off"); ax.set_title(("SEM/TEM\n" if i == 0 else "\n") + f"{name}", fontsize=9)
        ax = fig.add_subplot(gs[3, i])
        if col is not None: ax.imshow(bgr2rgb(col))
        ax.axis("off")
        if i == 0: ax.set_title("structure, in colour", fontsize=9, loc="left")
        ax = fig.add_subplot(gs[4, i]); ax.imshow(bgr2rgb(patch)); ax.axis("off")
        if i == 0: ax.set_title("material swatch (□ actual)", fontsize=9, loc="left")
        ax.add_patch(plt.Rectangle((0.80, 0.03), 0.16, 0.16, transform=ax.transAxes,
                                   color=np.array(plumage) / 255.0, ec="k"))
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    fig.savefig(OUT.replace(".png", ".pdf"), bbox_inches="tight")
    print("saved", OUT, "(and .pdf)")

    # ---------- iridescence strip ----------
    ang = rd.multilayer_angle_colors()
    fig2, axes = plt.subplots(1, len(ang) + 1, figsize=(2.0 * (len(ang) + 1), 2.4))
    for j, (a, srgb, hexv, peak) in enumerate(ang):
        axes[j].imshow(bgr2rgb(rd.render_patch(srgb, size=160, gloss=0.75)))
        axes[j].set_title(f"{a}°  {peak:.0f}nm", fontsize=9); axes[j].axis("off")
    # spongy comparison (fixed colour at all angles)
    ssrgb, _, _, _ = rd.structure_color(185, 0.5, 0.10)
    axes[-1].imshow(bgr2rgb(rd.render_patch(ssrgb, size=160)))
    axes[-1].set_title("spongy\n(no shift)", fontsize=9); axes[-1].axis("off")
    fig2.tight_layout()
    fig2.savefig(OUT2, dpi=150)
    fig2.savefig(OUT2.replace(".png", ".pdf"))
    print("saved", OUT2, "(and .pdf)")


if __name__ == "__main__":
    main()
