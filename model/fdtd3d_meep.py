#!/usr/bin/env python3
"""Preliminary 3-D FDTD reflectance cross-check for a generated structure.

This is a preliminary full-Maxwell route: build a 3-D voxel structure
(structure3d.generate_volume), shine broadband WHITE light on it, and let MEEP
solve the discretized 3-D Maxwell equations. The resulting spectrum is converted
to a display colour and compared with the analytic 3-D structure-factor model in
structure3d.py. Current runs are not numerically converged; see Supplementary S5.

MEEP does not install in the sandbox here -> run on your Mac:
    conda create -n meep -c conda-forge pymeep python=3.11
    conda activate meep
    pip install numpy
Run (start SMALL -- 3-D FDTD is heavy):
    python fdtd3d_meep.py --d 185 --vox 24 --nm-per-vox 22 --res 24

Method: standard MEEP reflectance recipe in 3-D:
  1) empty normalization run  -> incident flux,
  2) run with the structure (MaterialGrid air<->keratin) -> reflected flux,
     R(f) = -reflected/incident.
Colour via optics.spectrum_to_srgb.
"""
import argparse, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=float, default=185.0, help="characteristic spacing (nm)")
    ap.add_argument("--pf", type=float, default=0.5, help="pore fraction")
    ap.add_argument("--disorder", type=float, default=0.12)
    ap.add_argument("--vox", type=int, default=24, help="structure voxels per side (keep small!)")
    ap.add_argument("--nm-per-vox", type=float, default=22.0)
    ap.add_argument("--res", type=int, default=24, help="MEEP resolution (pixels per a=1um)")
    ap.add_argument("--n-keratin", type=float, default=1.56)
    ap.add_argument("--seed", type=int, default=1, help="disorder-realisation seed")
    ap.add_argument("--subpixel-smoothing", action="store_true",
                    help="enable MaterialGrid subpixel averaging for a sensitivity sweep")
    ap.add_argument("--out", default=None, help="CSV spectrum output path")
    args = ap.parse_args()

    import meep as mp
    import structure3d as s3
    import optics as o

    vol = s3.generate_volume(args.d, args.pf, args.disorder,
                             nm_per_vox=args.nm_per_vox, size=args.vox, seed=args.seed)
    a_nm = 1000.0
    L = args.vox * args.nm_per_vox / a_nm          # structure side in MEEP units (um)
    print(f"3-D structure {vol.shape} at {args.nm_per_vox} nm/vox -> {L*1000:.0f} nm = {L:.3f} a")

    dpml, pad = 1.0, 0.6
    sx = sy = L
    sz = L + 2 * (dpml + pad)
    cell = mp.Vector3(sx, sy, sz)
    pml = [mp.PML(dpml, direction=mp.Z)]

    lam_min, lam_max = 0.36, 0.78
    fmin, fmax = 1 / lam_max, 1 / lam_min
    fcen, df = 0.5 * (fmin + fmax), (fmax - fmin)
    nfreq = 100
    src_z = -0.5 * sz + dpml + 0.2
    sources = [mp.Source(mp.GaussianSource(fcen, fwidth=df), component=mp.Ex,
                         center=mp.Vector3(0, 0, src_z), size=mp.Vector3(sx, sy, 0))]
    refl_z = src_z + 0.25
    refl_fr = mp.FluxRegion(center=mp.Vector3(0, 0, refl_z), size=mp.Vector3(sx, sy, 0))
    # periodic in x,y (infinite slab of the structure)
    kpt = mp.Vector3()

    # 1) normalization (empty)
    sim = mp.Simulation(cell_size=cell, boundary_layers=pml, sources=sources,
                        resolution=args.res, k_point=kpt,
                        default_material=mp.air, dimensions=3)
    refl = sim.add_flux(fcen, df, nfreq, refl_fr)
    sim.run(until_after_sources=mp.stop_when_fields_decayed(
        20, mp.Ex, mp.Vector3(0, 0, refl_z), 1e-3))
    incident = np.array(mp.get_fluxes(refl))
    incident_data = sim.get_flux_data(refl)
    freqs = np.array(mp.get_flux_freqs(refl))
    sim.reset_meep()

    # 2) with the 3-D structure as a MaterialGrid
    keratin = mp.Medium(index=args.n_keratin)
    design = vol.astype(np.float64).flatten()      # 1 keratin, 0 air
    mg = mp.MaterialGrid(mp.Vector3(*vol.shape), mp.air, keratin,
                         weights=design, do_averaging=args.subpixel_smoothing)
    geometry = [mp.Block(center=mp.Vector3(0, 0, 0),
                         size=mp.Vector3(L, L, L), material=mg)]
    sim = mp.Simulation(cell_size=cell, boundary_layers=pml, sources=sources,
                        geometry=geometry, resolution=args.res, k_point=kpt, dimensions=3)
    refl = sim.add_flux(fcen, df, nfreq, refl_fr)
    sim.load_minus_flux_data(refl, incident_data)
    sim.run(until_after_sources=mp.stop_when_fields_decayed(
        20, mp.Ex, mp.Vector3(0, 0, refl_z), 1e-3))
    reflected = np.array(mp.get_fluxes(refl))

    R = -reflected / incident
    lam_nm = (1.0 / freqs) * a_nm
    order = np.argsort(lam_nm)
    lam_nm, R = lam_nm[order], np.clip(R[order], 0, None)
    srgb, hexv, peak = o.spectrum_to_srgb(lam_nm, R / max(R.max(), 1e-9))
    print(f"\n3-D FDTD reflectance peak ~ {peak:.0f} nm   colour {hexv}  sRGB{tuple(srgb)}")
    out = args.out or os.path.join(HERE, "..", "experiments", "fdtd3d_spectrum.csv")
    np.savetxt(out, np.c_[lam_nm, R], delimiter=",", header="lambda_nm,R", comments="")
    print("saved spectrum ->", os.path.normpath(out))
    print("compare with the analytic 3-D structure factor: "
          "python -c \"import sys;sys.path.insert(0,'.');import structure3d as s,optics as o;"
          "v=s.generate_volume(%.0f,%.2f,%.2f,nm_per_vox=%.0f,size=48);"
          "print(s.colour_3d(v,%.0f,%.2f)[1:3])\"" % (args.d, args.pf, args.disorder,
                                                       args.nm_per_vox, args.nm_per_vox, args.pf))


if __name__ == "__main__":
    main()
