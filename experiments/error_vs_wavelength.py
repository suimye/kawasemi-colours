#!/usr/bin/env python3
"""Quantify simulation variability as a function of wavelength/spacing and
species/panel-linked in-silico comparison, with error bars.

Two questions:
  (A) Species/panel-linked in-silico comparison: for each displayed record,
      compare a literature-informed candidate structure (same method as
      prove_structure_color.py) with its target peak label, and quantify the
      realization-to-realization variation at the declared parameters.
  (B) Simulation self-consistency across the spectrum: sweep the characteristic
      spacing d from violet-producing to red-producing values, and for each d
      generate N independent quasi-random realisations (different seed) at the
      SAME (d, pore_fraction, disorder). Report the spread (SD) of the resulting
      coherent peak wavelength across realisations, and the peak "quality"
      (peak height above the local baseline) -- both should degrade if the
      structure-factor route becomes less reliable at longer wavelengths/larger d.

Outputs experiments/error_vs_wavelength.png and prints a summary table.
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "model"))
import optics as o          # noqa
import synth                # noqa

LAM = np.arange(360.0, 781.0, 2.0)
OUT = os.environ.get("KAWASEMI_FIG_OUT", os.path.join(ROOT, "experiments", "error_vs_wavelength.png"))
N_REALIZATIONS = 20


def realizations(d_nm, pf, disorder, ppu=100.0, size=384, n=N_REALIZATIONS, seed0=100):
    """Return per-realisation dominant peak (nm) and peak quality for N
    independent quasi-random structures at the SAME nominal parameters, PLUS
    the peak of the ensemble-averaged reflectance (the statistic already used
    for the ensemble statistic used in the Figure 4B comparison)."""
    ne = o.n_eff(pf); nmpp = 1000.0 / ppu
    peaks, quals, Rs = [], [], []
    for s in range(n):
        img = synth.generate_image(d_nm, pf, disorder, 0.0, px_per_um=ppu, size=size, seed=seed0 + s)
        _, R = o.reflectance_from_image(img, nmpp, ne, lam=LAM)
        if R.max() <= 0:
            continue
        Rn = R / R.max()
        _, _, peak = o.spectrum_to_srgb(LAM, Rn)
        peaks.append(peak)
        floor = np.percentile(Rn, 25)
        quals.append(float(Rn.max() - floor))
        Rs.append(R)
    Rmean = np.mean(Rs, 0); Rmean = Rmean / max(Rmean.max(), 1e-9)
    _, _, ensemble_peak = o.spectrum_to_srgb(LAM, Rmean)
    return np.array(peaks), np.array(quals), ensemble_peak


def load_real(rid):
    for line in open(os.path.join(ROOT, "data", "dataset.jsonl")):
        rec = json.loads(line)
        if rec["id"] == rid:
            return rec
    return None


def part_A_record_comparison():
    """Candidate structures compared with record target labels, with ensemble-SD
    error bars from independent disorder realisations."""
    birds = [
        ("PMC4685390_fig6c_jay",        178.0, 0.50, "Eurasian jay"),
        ("BIORXIV_tersina_fig6c",       200.0, 0.50, "swallow tanager"),
        ("BIORXIV_tersina_fig6b_inset", 190.0, 0.50, "swallow tanager (inset)"),
    ]
    rows = []
    print("=== (A) species/panel-linked in-silico comparison: ensemble-averaged")
    print("    synthetic peak versus record target; error bar =")
    print("    SD of individual-realisation peaks (realisation-to-realisation spread) ===")
    for rid, d, pf, label in birds:
        rec = load_real(rid)
        if rec is None:
            print(f"  [skip] {rid} not found"); continue
        actual = rec["color"].get("peak_nm")
        peaks, quals, ens_peak = realizations(d, pf, 0.10)
        pred_sd = peaks.std()
        err = abs(ens_peak - actual) if actual else None
        print(f"  {label:24s} d={d:5.0f}nm  ensemble-peak={ens_peak:6.1f}nm  "
              f"realisation-SD=±{pred_sd:4.1f}nm  target={actual}nm  "
              + (f"|difference|={err:.1f}nm" if err else "no target peak_nm"))
        rows.append((label, actual, ens_peak, pred_sd, err))
    return rows


def part_B_spectrum_sweep():
    """Sweep d from violet- to red-producing values; report ensemble SD and
    peak quality (coherent-peak contrast) at each."""
    pf, disorder = 0.5, 0.10
    ds = [130, 150, 170, 190, 210, 230, 250, 270, 290]
    print("\n=== (B) simulation self-consistency across the spectrum (pf=0.5, disorder=0.10) ===")
    rows = []
    for d in ds:
        bragg = o.bragg_peak_nm(d, pf)
        peaks, quals, ens_peak = realizations(d, pf, disorder, n=N_REALIZATIONS)
        print(f"  d={d:4.0f}nm  target(2n_eff d)={bragg:6.1f}nm  "
              f"ensemble-peak={ens_peak:6.1f}nm  realisation-SD=±{peaks.std():5.1f}nm  "
              f"quality={quals.mean():.3f}±{quals.std():.3f}  n={len(peaks)}")
        rows.append((d, bragg, ens_peak, peaks.std(), quals.mean(), quals.std()))
    return rows


def main():
    rowsA = part_A_record_comparison()
    rowsB = part_B_spectrum_sweep()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

    # Panel A: candidate-structure peak versus record target, with seed variation.
    ax = axes[0]
    actuals = [r[1] for r in rowsA]
    preds = [r[2] for r in rowsA]
    sds = [r[3] for r in rowsA]
    labels = [r[0] for r in rowsA]
    colors = [o.spectrum_to_srgb(LAM, np.exp(-0.5 * ((LAM - a) / 25) ** 2))[0] for a in actuals]
    colors = [np.array(c) / 255.0 for c in colors]
    ax.plot([380, 650], [380, 650], "k--", lw=1, alpha=0.5, label="exact match")
    label_offsets = {
        "Eurasian jay": (6, -12),
        "swallow tanager": (6, 8),
        "swallow tanager (inset)": (6, -14),
    }
    for a, p, s, c, lbl in zip(actuals, preds, sds, colors, labels):
        ax.errorbar(a, p, yerr=s, fmt="o", color=c, ecolor="0.4", elinewidth=1.5,
                     capsize=4, markersize=9, markeredgecolor="k", markeredgewidth=0.8)
        ax.annotate(lbl, (a, p), fontsize=7, xytext=label_offsets.get(lbl, (4, 4)),
                    textcoords="offset points")
    ax.set_xlabel("record target peak (nm)")
    ax.set_ylabel("candidate-structure peak (nm)")
    ax.set_title("A", fontsize=13, weight="bold", loc="left")
    ax.set_xlim(380, 650); ax.set_ylim(380, 650)
    ax.legend(fontsize=8, loc="upper left")
    ax.axvspan(600, 650, color="red", alpha=0.05)
    ax.text(625, 410, "no real\ndata here", fontsize=7, color="firebrick",
            ha="center", style="italic")

    # Panel B: single message -- does that noise grow toward red?
    ax2 = axes[1]
    expected = [r[1] for r in rowsB]
    means = [r[2] for r in rowsB]
    sds2 = [r[3] for r in rowsB]
    quals = [r[4] for r in rowsB]
    qual_sds = [r[5] for r in rowsB]
    colors2 = [np.array(o.spectrum_to_srgb(LAM, np.exp(-0.5 * ((LAM - t) / 25) ** 2))[0]) / 255.0 for t in expected]
    ax2.errorbar(expected, means, yerr=sds2, fmt="o-", color="0.3", ecolor="0.5",
                 elinewidth=1.5, capsize=4, markersize=0, lw=1, zorder=1)
    for t, m, c in zip(expected, means, colors2):
        ax2.scatter([t], [m], color=c, s=90, edgecolor="k", zorder=2)
    ax2.plot([350, 750], [350, 750], "k--", lw=1, alpha=0.4, label="exact match")
    ax2.set_xlabel("expected colour from spacing d (λ=2·n_eff·d, nm)")
    ax2.set_ylabel("simulated colour, mean ± SD (nm)")
    ax2.set_title("B", fontsize=13, weight="bold", loc="left")
    ax2.legend(fontsize=8, loc="upper left")

    ax2b = ax2.twinx()
    ax2b.errorbar(expected, quals, yerr=qual_sds, fmt="s--", color="firebrick",
                  elinewidth=1, capsize=3, markersize=5, alpha=0.7, label="peak sharpness")
    ax2b.set_ylabel("peak sharpness (0-1)", color="firebrick")
    ax2b.tick_params(axis="y", colors="firebrick")

    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    fig.savefig(OUT.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"\nsaved figure -> {OUT} (and .pdf)")


if __name__ == "__main__":
    main()
