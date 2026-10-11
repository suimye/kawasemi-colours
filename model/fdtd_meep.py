#!/usr/bin/env python3
"""Preliminary FDTD reflectance cross-check for a nanostructure image, with MEEP.

This complements the analytic structure-factor model in optics.py by solving
Maxwell's equations for a 2-D computational representation and returning a
reflectance spectrum -> colour. It is an in-silico cross-check, not independent
experimental validation; convergence tests, a positive control, and comparison
with independent measured spectra remain to be completed. Run it on your Mac
(MEEP does not install in the sandbox here).

Install (Apple Silicon):
    conda create -n meep -c conda-forge pymeep python=3.11
    conda activate meep
    pip install opencv-python-headless numpy   # for the helpers below

Run:
    python fdtd_meep.py --id STAVENGA2011_king_fig3F --period-nm 185
    python fdtd_meep.py --image path/to/sem.png --nm-per-px 3.0

Method: standard MEEP reflectance recipe (Meep tutorial "Transmittance spectrum
of a waveguide"/"mode decomposition" analogue for a slab):
  1) normalization run with vacuum to record the incident flux,
  2) run with the structure as a MaterialGrid (air<->keratin) to record the
     reflected flux, then R(f) = -reflected/incident.
Colour via optics.spectrum_to_srgb (CIE).
"""
import argparse, os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)


def load_design(args):
    """Return (design[0..1] HxW, nm_per_px). 1 = keratin, 0 = air."""
    import cv2
    if args.id:
        base = os.path.normpath(os.path.join(HERE, "..", "data"))
        for line in open(os.path.join(base, "dataset.jsonl")):
            r = json.loads(line)
            if r["id"] == args.id:
                img = cv2.imread(os.path.join(base, r["image_file"]), cv2.IMREAD_GRAYSCALE)
                break
        else:
            raise SystemExit("id not found")
    else:
        img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (args.grid, args.grid))
    design = (img.astype(np.float32) / 255.0)          # bright walls -> keratin
    # infer nm/px: prefer explicit; else from a target period
    if args.nm_per_px:
        nmpp = args.nm_per_px
    elif args.period_nm:
        # scale so the image's dominant period == period_nm
        sys.path.insert(0, HERE)
        import optics as o
        _, S = o.radial_structure_factor(img, 1.0); S = S.copy(); S[:3] = 0
        k = np.arange(1, len(S) - 1)
        per = img.shape[0] / k
        m = (per >= 4) & (per <= 26) & (S[k] > S[k - 1]) & (S[k] > S[k + 1])
        kk = k[m][np.argmax(S[k][m])] if m.any() else int(img.shape[0] / 12)
        nmpp = args.period_nm / (img.shape[0] / kk)
    else:
        nmpp = 3.0
    return design, nmpp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id"); ap.add_argument("--image")
    ap.add_argument("--nm-per-px", type=float, default=None)
    ap.add_argument("--period-nm", type=float, default=None)
    ap.add_argument("--grid", type=int, default=160, help="structure grid (px)")
    ap.add_argument("--res", type=int, default=40, help="MEEP resolution (px per a)")
    ap.add_argument("--n-keratin", type=float, default=1.56)
    args = ap.parse_args()

    import meep as mp

    design, nmpp = load_design(args)
    Lnm = design.shape[0] * nmpp
    a_nm = 1000.0                       # 1 MEEP unit = 1 micron = 1000 nm
    L = Lnm / a_nm                      # structure size in MEEP units
    print(f"structure {design.shape} at {nmpp:.2f} nm/px -> {Lnm:.0f} nm = {L:.3f} a")

    dpml = 1.0; pad = 1.0
    sx = L; sy = L + 2 * (dpml + pad)
    cell = mp.Vector3(sx, sy)
    pml = [mp.PML(dpml, direction=mp.Y)]

    # broadband planewave covering the visible band
    lam_min, lam_max = 0.36, 0.78       # microns
    fmin, fmax = 1 / lam_max, 1 / lam_min
    fcen, df = 0.5 * (fmin + fmax), (fmax - fmin)
    src_y = -0.5 * sy + dpml + 0.3
    sources = [mp.Source(mp.GaussianSource(fcen, fwidth=df), component=mp.Ez,
                         center=mp.Vector3(0, src_y), size=mp.Vector3(sx, 0))]
    refl_y = src_y + 0.4
    refl_fr = mp.FluxRegion(center=mp.Vector3(0, refl_y), size=mp.Vector3(sx, 0))
    nfreq = 120

    # ---- 1) normalization (empty) ----
    sim = mp.Simulation(cell_size=cell, boundary_layers=pml, sources=sources,
                        resolution=args.res, k_point=mp.Vector3())
    refl = sim.add_flux(fcen, df, nfreq, refl_fr)
    sim.run(until_after_sources=mp.stop_when_fields_decayed(
        20, mp.Ez, mp.Vector3(0, refl_y), 1e-3))
    incident = mp.get_fluxes(refl)
    incident_data = sim.get_flux_data(refl)
    freqs = np.array(mp.get_flux_freqs(refl))
    sim.reset_meep()

    # ---- 2) with structure (MaterialGrid air<->keratin) ----
    keratin = mp.Medium(index=args.n_keratin)
    mg = mp.MaterialGrid(mp.Vector3(design.shape[1], design.shape[0]),
                         mp.air, keratin, design_parameters=design.flatten(),
                         grid_type="U_MEAN")
    geometry = [mp.Block(center=mp.Vector3(0, 0), size=mp.Vector3(L, L, mp.inf),
                         material=mg)]
    sim = mp.Simulation(cell_size=cell, boundary_layers=pml, sources=sources,
                        geometry=geometry, resolution=args.res, k_point=mp.Vector3())
    refl = sim.add_flux(fcen, df, nfreq, refl_fr)
    sim.load_minus_flux_data(refl, incident_data)   # subtract incident field
    sim.run(until_after_sources=mp.stop_when_fields_decayed(
        20, mp.Ez, mp.Vector3(0, refl_y), 1e-3))
    reflected = mp.get_fluxes(refl)

    R = -np.array(reflected) / np.array(incident)
    lam_nm = (1.0 / freqs) * a_nm
    order = np.argsort(lam_nm)
    lam_nm, R = lam_nm[order], np.clip(R[order], 0, None)

    import optics as o
    srgb, hexv, peak = o.spectrum_to_srgb(lam_nm, R / max(R.max(), 1e-9))
    print(f"\nFDTD reflectance peak ~ {peak:.0f} nm   colour {hexv}  sRGB{tuple(srgb)}")
    # save spectrum
    out = os.path.join(HERE, "..", "experiments", "fdtd_spectrum.csv")
    np.savetxt(out, np.c_[lam_nm, R], delimiter=",", header="lambda_nm,R", comments="")
    print("saved spectrum ->", os.path.normpath(out))


if __name__ == "__main__":
    main()
