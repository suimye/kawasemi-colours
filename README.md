# Kawasemi Colours — public reproducibility package

This repository is the public software and derived-output companion to the
Kawasemi Colours study. It contains code, configuration, record-level metadata,
browser tools, supplementary tables, and figures that can be redistributed.

It deliberately does **not** contain source electron micrographs copied from
third-party publications. Paths such as `images/...` in the metadata preserve
record provenance but do not imply that those image files are distributed here.

## Included

- `model/` — feature extraction, synthetic structures, forward optics, inverse
  design, grouped evaluation, and preliminary MEEP/FDTD scripts.
- `experiments/` — figure-generating scripts and redistribution-safe outputs.
- `simulator/` — self-contained forward and inverse browser tools.
- `data/` — source, licence, provenance, fold-membership, and derived record
  metadata; no source micrograph pixels.
- `figures/` — manuscript-associated figures containing only software
  screenshots, calculated plots, or generated structures.
- `supplementary/` — data-licence, provenance, validation, and FDTD tables.

## Scientific scope

The coherent-scattering calculations and synthetic structures are in-silico
models. The MEEP/FDTD material is retained as a preliminary full-Maxwell
cross-check. It is not a converged or independent experimental validation;
resolution convergence, a positive control, and comparison with independent
measured spectra remain future work.

Real-image grouped analyses require lawfully obtained source images and cannot
be rerun from this public repository alone. Their deterministic membership,
derived metadata, protocol, and reported summary tables are included so that
the evaluated population and limitations remain inspectable.

## Quick start

Python 3.10 or later is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# Target colour -> candidate structural parameters
python model/inverse.py "#2a78b8"

# Synthetic, redistribution-safe figures
python experiments/explain_and_emergent.py
python experiments/proof_3d.py
python experiments/error_vs_wavelength.py
```

CNN training is optional:

```bash
python -m pip install -r requirements-ml.txt
python model/train.py --n 400 --no-cnn
```

The static browser tools can be opened directly:

- `simulator/index.html` — inverse design, colour to candidate structure.
- `simulator/appearance.html` — forward calculation, structure to colour.

Screenshot-based instructions for both browser tools are available in the
[Japanese web app user guide](docs/USER_GUIDE.ja.md).

MEEP is installed separately with conda; see `environment-meep.yml` and
`supplementary/supplementary_S5_fdtd_reproducibility.md`.

## Figures and source-image exclusions

See `figures/README.md`. Manuscript Figures 1, 2, 5, and 10 are omitted from
this repository because their composites contain third-party micrographs.

## Licence and citation

Source code is released under the MIT License. Original documentation, derived
figures, and repository-authored tabular metadata are released under CC BY 4.0.
Third-party works are not relicensed and are not distributed here. See
`LICENSES.md`, `data/sources.csv`, and `supplementary/supplementary_S1_data_licenses.csv`.

Citation metadata are provided in `CITATION.cff`. A version-specific Zenodo DOI
can be added after the first archived GitHub release.
