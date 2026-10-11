# Public metadata package

This directory contains record-level metadata and derived numerical values. It
does not contain third-party electron-micrograph pixels.

- `dataset.jsonl` and `dataset.csv`: record metadata and derived fields.
- `sources.csv`: paper-level provenance and source licence information.
- `provenance_table.csv`: measurement and calibration provenance.
- `validation_folds_seed0.csv`: fixed random-five-fold membership.
- `candidate_sources.csv`: candidate literature sources.

The `image_file` values in `dataset.*` refer to files held in the private local
analysis copy or obtainable from the cited source under its own terms. They are
retained so that record identities and provenance remain stable. They are not
download links and do not imply permission to redistribute the images.
