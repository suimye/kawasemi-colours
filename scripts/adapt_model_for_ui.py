#!/usr/bin/env python3
"""Integration adapter: make the model's rich model_params.json consumable by
the web simulator (which expects {type:"lookup", table:[...]}). Adds `type`
and `table` keys (keeping all rich fields) and writes copies to both
model/ and simulator/ so index.html finds a compatible file either way.

Run from project root:  python3 scripts/adapt_model_for_ui.py
"""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "model", "model_params.json")
N_KERATIN, N_AIR = 1.56, 1.0

def n_eff(pf):
    return pf * N_AIR + (1 - pf) * N_KERATIN

def build_table(grid):
    table = []
    for e in grid:
        pf = e["pore_fraction"]
        period = e["period_nm"]
        pore_d = round(period * pf, 1)               # matches inverse.py convention
        table.append({
            "peak_nm": e["peak_nm"],
            "d_nm": period,
            "n_eff": round(n_eff(pf), 4),
            "pore_fraction": pf,
            "pore_diameter_nm": pore_d,
            "domain_std_nm": round(e.get("disorder", 0.1) * period, 1),
        })
    table.sort(key=lambda r: r["peak_nm"])
    return table

def main():
    with open(SRC) as f:
        d = json.load(f)
    d["type"] = "lookup"
    d["table"] = build_table(d["lookup_grid"])
    for dst in [SRC, os.path.join(ROOT, "simulator", "model_params.json")]:
        with open(dst, "w") as f:
            json.dump(d, f, separators=(",", ":"))
        print("wrote", dst, os.path.getsize(dst), "bytes,", len(d["table"]), "table rows")

if __name__ == "__main__":
    main()
