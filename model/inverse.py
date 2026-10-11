"""Inverse model: TARGET COLOR -> structural parameters.

Given a target color (sRGB / hex / name) produce the structural parameters that
reproduce it under the physics forward model:

    period_nm (char_length), pore_fraction, pore_diameter, order/disorder,
    anisotropy

Strategy:
  (a) analytic inversion of the shared relation  d = peak / (2 * n_eff)
      plus closed-form guesses for disorder (from saturation) and
      pore_fraction (from brightness), then
  (b) refinement by coordinate/gradient-free optimization against
      synth.forward_color so the produced sRGB matches the target.

Torch-free so the browser simulator's fallback path can mirror this logic.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from colorutil import (
    d_from_peak,
    n_eff,
    srgb8_to_peak_sat_val,
    hex_to_srgb8,
    rgb01_to_srgb8,
    rgb_to_hex,
    srgb8_to_rgb01,
    NAMED_COLORS,
)
from synth import forward_color


def _analytic_seed(peak_nm, saturation, value):
    """Closed-form first guess for the structural params."""
    # brightness -> pore_fraction. Brightness peaks at pf~0.5 (max contrast).
    # We can't uniquely invert the parabola, so prefer pf in the physical
    # spongy range and nudge with brightness.
    pore_fraction = float(np.clip(0.5 + 0.0 * value, 0.35, 0.6))

    # disorder from saturation: sat = 1 - 0.9*disorder  =>  disorder=(1-sat)/0.9
    disorder = float(np.clip((1.0 - saturation) / 0.9, 0.0, 1.0))

    # anisotropy: default isotropic sponge (kingfisher spongy is isotropic)
    anisotropy = 0.05

    char_length_nm = float(d_from_peak(peak_nm, pore_fraction))
    return dict(char_length_nm=char_length_nm, pore_fraction=pore_fraction,
                disorder=disorder, anisotropy=anisotropy)


def _color_loss(params, target_rgb01):
    c = forward_color(return_components=True, **params)
    got = np.array(srgb8_to_rgb01(c["srgb"]))
    err = float(np.sum((got - np.array(target_rgb01)) ** 2))
    # mild prior: spongy structures are isotropic, so penalize anisotropy so
    # the optimizer doesn't invent alignment to chase small color residuals.
    err += 0.05 * float(params.get("anisotropy", 0.0)) ** 2
    return err, c


def refine(params, target_rgb01, iters=60):
    """Gradient-free coordinate refinement against the forward model."""
    p = dict(params)
    bounds = dict(char_length_nm=(90.0, 320.0), pore_fraction=(0.30, 0.70),
                  disorder=(0.0, 1.0), anisotropy=(0.0, 0.8))
    steps = dict(char_length_nm=20.0, pore_fraction=0.06,
                 disorder=0.12, anisotropy=0.1)
    best_loss, _ = _color_loss(p, target_rgb01)
    for _ in range(iters):
        improved = False
        for k in ["char_length_nm", "pore_fraction", "disorder", "anisotropy"]:
            for sign in (+1, -1):
                trial = dict(p)
                trial[k] = float(np.clip(p[k] + sign * steps[k], *bounds[k]))
                loss, _ = _color_loss(trial, target_rgb01)
                if loss < best_loss - 1e-9:
                    best_loss = loss
                    p = trial
                    improved = True
        if not improved:
            for k in steps:
                steps[k] *= 0.6
            if all(steps[k] < 1e-3 for k in steps):
                break
    return p, best_loss


def invert_color(srgb8, refine_iters=60):
    """Main entry: sRGB target -> structural params + reproduced color."""
    peak_nm, saturation, value = srgb8_to_peak_sat_val(srgb8)
    seed = _analytic_seed(peak_nm, saturation, value)
    refined, loss = refine(seed, srgb8_to_rgb01(srgb8), iters=refine_iters)

    c = forward_color(return_components=True, **refined)

    # derived structural descriptors for the report
    d = refined["char_length_nm"]
    pf = refined["pore_fraction"]
    result = {
        "target": {
            "srgb": list(srgb8),
            "hex": rgb_to_hex(srgb8),
            "peak_nm": round(peak_nm, 1),
            "saturation": round(saturation, 3),
            "value": round(value, 3),
        },
        "structure": {
            "period_nm": round(d, 1),
            "pore_fraction": round(pf, 3),
            "pore_diameter_nm": round(d * pf, 1),          # ~ period * pore fraction
            "wall_thickness_nm": round(d * (1.0 - pf), 1),  # complement
            "disorder": round(refined["disorder"], 3),
            "order": round(1.0 - refined["disorder"], 3),
            "anisotropy": round(refined["anisotropy"], 3),
            "n_eff": round(n_eff(pf), 4),
        },
        "reproduced": {
            "srgb": c["srgb"],
            "hex": c["hex"],
            "peak_nm": round(c["peak_nm"], 1),
            "saturation": round(c["saturation"], 3),
            "value": round(c["value"], 3),
        },
        "color_rmse_srgb": round(float(np.sqrt(loss / 3.0)) * 255, 2),
    }
    return result


def parse_color_arg(s):
    s = s.strip()
    if s.lower() in NAMED_COLORS:
        return NAMED_COLORS[s.lower()]
    if s.startswith("#"):
        return hex_to_srgb8(s)
    if "," in s:
        return [int(float(x)) for x in s.split(",")]
    raise ValueError(f"cannot parse color: {s!r}")


def main():
    ap = argparse.ArgumentParser(description="Target color -> structural params")
    ap.add_argument("color", help="name ('kingfisher blue'), #hex, or 'r,g,b'")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--json", action="store_true", help="print raw JSON")
    args = ap.parse_args()

    srgb8 = parse_color_arg(args.color)
    res = invert_color(srgb8, refine_iters=args.iters)
    if args.json:
        print(json.dumps(res, indent=2))
        return
    t = res["target"]
    s = res["structure"]
    r = res["reproduced"]
    print(f"TARGET: {t['hex']} srgb={t['srgb']} peak~{t['peak_nm']}nm "
          f"sat={t['saturation']} val={t['value']}")
    print("STRUCTURE (to reproduce it):")
    print(f"  period (char_length) = {s['period_nm']} nm")
    print(f"  pore_fraction        = {s['pore_fraction']}")
    print(f"  pore_diameter        = {s['pore_diameter_nm']} nm")
    print(f"  wall_thickness       = {s['wall_thickness_nm']} nm")
    print(f"  disorder / order     = {s['disorder']} / {s['order']}")
    print(f"  anisotropy           = {s['anisotropy']}")
    print(f"  n_eff                = {s['n_eff']}")
    print(f"REPRODUCED: {r['hex']} peak={r['peak_nm']}nm  "
          f"(sRGB RMSE {res['color_rmse_srgb']})")


if __name__ == "__main__":
    main()
