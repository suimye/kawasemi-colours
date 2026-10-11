# Supplementary Table S3. Scale-stratified validation of FEATS_V1 prediction

Length-valued FEATS_V1 descriptors were calculated with the record-specific
`scale.px_per_um` only when the record was marked `calibrated: true`. All other
records retain the extractor's explicit heuristic conversion and are reported as
the **heuristic-scale** subset. The latter is not an absolute nanometre
measurement. Models were trained as described in `model/validate_real_grouped.py`:
800 synthetic structures plus the available non-held-out records. Evaluation is
restricted to 28 `real_image=true` records eligible for colour-model training.
The three swallow-tanager records are excluded because their retained sRGB values
are illustrative display swatches, not spectrophotometrically calibrated colour
targets. Results are exploratory because the measured-scale subset contains only
six records and still spans different optical morphologies.

| Validation split | All eligible real-image records (n=28) | Measured-scale records (n=6) | Heuristic-scale records (n=22) |
|---|---:|---:|---:|
| Random 5-fold MAE, median (nm) | 58.9, 50.1 | 90.1, 85.3 | 50.4, 37.7 |
| Leave-one-species-out MAE, median (nm) | 55.7, 40.1 | 87.9, 73.4 | 47.0, 33.5 |
| Leave-one-paper-out MAE, median (nm) | 57.0, 41.1 | 87.9, 72.5 | 48.6, 36.7 |

The poorer measured-scale-subset values do not establish that calibrated scales
reduce performance. They show instead that the present data are too small and too
heterogeneous to make a reliable absolute-scale performance claim. In particular,
the six calibrated records include several melanosome-array examples outside the
quasi-ordered keratin–air mechanism targeted by the primary simulator. Future
evaluation should use an independently acquired, scale-calibrated, non-iridescent
keratin–air test set with reflectance spectra measured under recorded geometry.
