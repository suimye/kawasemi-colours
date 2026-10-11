"""Export a compact JSON the web simulator can use WITHOUT python/torch.

Writes model/model_params.json containing:
  - the physics constants (n_keratin, n_air) and the peak = 2*n_eff*d relation,
  - the saturation-vs-disorder and brightness-vs-pore_fraction curves,
  - the interpretable FEATS_V1 -> color regressor weights (if baseline.npz),
  - a sampled color -> params LOOKUP GRID over the visible range, so the browser
    can do inverse (color -> structure) by nearest-neighbor without optimizing.

This is the single artifact the simulator loads. Kept small (<200KB).
"""
from __future__ import annotations

import json
import os

import numpy as np

from colorutil import (
    N_KERATIN, N_AIR, VISIBLE_MIN_NM, VISIBLE_MAX_NM,
    n_eff, rgb_to_hex,
)
from synth import forward_color
from inverse import invert_color

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "model_params.json")
BASELINE = os.path.join(HERE, "baseline.npz")

from features import FEATS_V1, DEFAULT_PX_PER_UM


def build_lookup_grid(n_peak=48, pore_fractions=(0.35, 0.45, 0.55, 0.65)):
    """color -> structural params lookup, sampled across the visible range.

    For each (peak, pore_fraction) we store the forward color and the analytic
    structure so the browser can nearest-match a target color to a structure.
    """
    grid = []
    peaks = np.linspace(VISIBLE_MIN_NM + 20, VISIBLE_MAX_NM - 40, n_peak)
    disorders = (0.1, 0.4)  # ordered vs partly disordered
    for peak in peaks:
        for pf in pore_fractions:
            for dis in disorders:
                d = peak / (2.0 * n_eff(pf))
                c = forward_color(char_length_nm=d, pore_fraction=pf,
                                  disorder=dis, anisotropy=0.05,
                                  return_components=True)
                grid.append({
                    "srgb": c["srgb"],
                    "hex": c["hex"],
                    "peak_nm": round(float(peak), 1),
                    "period_nm": round(float(d), 1),
                    "pore_fraction": round(float(pf), 3),
                    "disorder": round(float(dis), 3),
                    "sat": round(c["saturation"], 3),
                    "val": round(c["value"], 3),
                })
    return grid


def load_baseline():
    if not os.path.exists(BASELINE):
        return None
    z = np.load(BASELINE)
    return {
        "feats_order": FEATS_V1,
        "W": z["W"].round(6).tolist(),      # (F+1, 3), last row = bias
        "mu": z["mu"].round(6).tolist(),
        "sd": z["sd"].round(6).tolist(),
        "target": ["peak_norm", "saturation", "value"],
        "note": "peak_nm = peak_norm*(vis_max-vis_min)+vis_min; input FEATS_V1 standardized (x-mu)/sd then append 1 for bias",
    }


def main():
    params = {
        "version": "FEATS_V1",
        "physics": {
            "n_keratin": N_KERATIN,
            "n_air": N_AIR,
            "relation": "peak_nm = 2 * n_eff * d",
            "n_eff": "pore_fraction*n_air + (1-pore_fraction)*n_keratin",
            "visible_range_nm": [VISIBLE_MIN_NM, VISIBLE_MAX_NM],
        },
        "forward_curves": {
            "saturation_vs_disorder": "sat = clip(1 - 0.9*disorder, 0.05, 1) * (1 + 0.15*anisotropy)",
            "value_vs_pore_fraction": "val = clip(0.45 + 0.5*(4*pf*(1-pf)), 0.1, 1)",
        },
        "feature_scale": {
            "default_px_per_um": DEFAULT_PX_PER_UM,
            "feats_order": FEATS_V1,
        },
        "baseline_regressor": load_baseline(),
        "lookup_grid": build_lookup_grid(),
        "named_examples": {},
    }

    # a few worked inverse examples for UI display and consistency checks
    for name, srgb in {
        "kingfisher blue": [31, 111, 176],
        "green": [40, 170, 90],
        "red": [200, 50, 50],
    }.items():
        r = invert_color(srgb, refine_iters=40)
        params["named_examples"][name] = {
            "target_srgb": srgb,
            "period_nm": r["structure"]["period_nm"],
            "pore_fraction": r["structure"]["pore_fraction"],
            "disorder": r["structure"]["disorder"],
            "reproduced_peak_nm": r["reproduced"]["peak_nm"],
            "reproduced_hex": r["reproduced"]["hex"],
        }

    with open(OUT, "w") as f:
        json.dump(params, f, separators=(",", ":"))
    size = os.path.getsize(OUT)
    # check the model-internal round trip
    with open(OUT) as f:
        json.load(f)
    print(f"wrote {OUT} ({size/1024:.1f} KB, {len(params['lookup_grid'])} grid entries)")
    if size > 200 * 1024:
        print("WARNING: exceeds 200KB budget")
    for name, ex in params["named_examples"].items():
        print(f"  {name}: period={ex['period_nm']}nm pf={ex['pore_fraction']} "
              f"-> peak={ex['reproduced_peak_nm']}nm {ex['reproduced_hex']}")


if __name__ == "__main__":
    main()
