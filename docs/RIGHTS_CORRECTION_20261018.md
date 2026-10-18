# Image-rights and methods correction — 18 October 2026

This correction changes provenance and explanatory text, not numerical results,
training populations, fold assignments, model parameters or executable calculations.

## Image-level rights

The CC BY 4.0 licence of Jeon et al. (2021),
https://doi.org/10.1186/s42649-021-00063-w, was incorrectly propagated to
some third-party images reproduced in that review. Individual figure credits
must be checked independently of the article licence.

- `PMC8505553_fig6d_{sphere,roundish,mixed,ellipsoid,elong}`:
  Kohri et al. (2019), https://doi.org/10.1021/acs.langmuir.9b00400;
  the review credits reproduction with ACS permission.
- `PMC8505553_fig4_stellersjay`: Shawkey and Hill (2006),
  https://doi.org/10.1242/jeb.02115; Company of Biologists permission credit.
- `PMC8505553_fig5a_parotia`, `PMC8505553_fig5c_starling`, and
  `PMC8505553_fig5d_racket`: third-party permission credits; separate
  redistribution permission is not documented here.
- Rock-dove and teal records retain original-source CC BY credits; version 4.0
  is not inferred from the review licence.

The nine permission-credit records are not asserted to be CC BY. This is a
conservative redistribution assessment, not a determination that every use is
prohibited. Original sources and applicable permissions must be checked.

Source electron micrographs and composite Figures 1, 2, 5 and 10 were already
excluded from this public repository. No source-image deletion was necessary.
They remain excluded, including the revised manuscript Figure 2. Metadata-only
records are retained to preserve analysis provenance. Dataset CSV/JSONL,
provenance tables and Supplementary S1 are corrected together. `data/sources.csv`
records article-level licences, not blanket licences for reproduced images.
Regenerated metadata must be checked against the corrected image-level table.

## Methods and terminology

The Fourier-method reference is Prum and Torres (2003), *A Fourier tool for the
analysis of coherent light scattering by bio-optical nanostructures*,
https://doi.org/10.1093/icb/43.4.591. The backscattering approximation relates to
Noh et al. (2010), https://doi.org/10.1002/adma.200903699, equations 2–3.
Percentile baseline subtraction is an implementation heuristic, not a quantitative
melanin-absorption model established by that paper.

Figure 6 “peak sharpness” is the maximum of the peak-normalized spectrum minus
its 25th percentile: a dimensionless contrast statistic, not linewidth.
The revised manuscript Supplementary Figure S1 plots first-order analytically
calculated wavelengths for adjacent independent texture regions, not peaks
extracted from the textures. `experiments/emergent_color.*` is a separate calculation.
FDTD/Meep remains a preliminary in-silico cross-check. Supplementary S5's reference
to the 2-D analysis has been corrected to Section 3.4, Figure 6B.

## Archived versions

This correction updates the default branch only. Existing Git history, releases,
Zenodo records and bioRxiv versions have not been replaced or withdrawn.
Consult this correction when using earlier snapshots; archive updates are separate.
