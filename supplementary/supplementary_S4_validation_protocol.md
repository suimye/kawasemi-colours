# Supplementary Table S4. Validation populations, randomization, and synthetic training protocol

All results below use the eligible record set after excluding the three
swallow-tanager records with non-calibrated illustrative sRGB display swatches.
The code and fixed seeds are stated so that the distinct evaluation populations
are not conflated.

| Analysis | Evaluated records | Training data in each fold | Split and seed | Current result |
|---|---|---|---|---|
| `train_real.py` random 5-fold | 39 eligible records: 28 `real_image=true` and 11 synthetic-proxy metadata records; mixed optical morphologies | 800 on-the-fly synthetic structures plus the non-held-out eligible records | `np.random.default_rng(0)`, five contiguous `array_split` folds after permutation | Peak MAE 57.2 nm; median 37.0 nm; saturation MAE 0.189 |
| `validate_real_grouped.py` random 5-fold | 28 eligible `real_image=true` records; mixed optical morphologies | 800 on-the-fly synthetic structures, all 11 proxies, and non-held-out real images | `np.random.default_rng(0)`, five contiguous `array_split` folds after permutation | Peak MAE 58.9 nm; median 50.1 nm |
| `validate_real_grouped.py` LOSO | Same 28 real images, 16 species groups | 800 on-the-fly synthetic structures, all proxies, and records outside the held-out species | one held-out species at a time; no stochastic split beyond synthetic seed 0 | Peak MAE 55.7 nm; median 40.1 nm |
| `validate_real_grouped.py` LOPO | Same 28 real images, 7 source-paper groups | 800 on-the-fly synthetic structures, all proxies, and records outside the held-out source paper | one held-out source paper at a time; no stochastic split beyond synthetic seed 0 | Peak MAE 57.0 nm; median 41.1 nm |

The exact real-image random-five-fold assignment is archived in
`data/validation_folds_seed0.csv`. These are exploratory pooled results: the
evaluated set includes quasi-ordered keratin–air structures, melanosome arrays,
and biomimetic controls. It is not a single-mechanism accuracy estimate.

## On-the-fly synthetic training structures

The 11 synthetic-proxy metadata records above are not the synthetic training
set. Each run generates 800 synthetic 96 × 96-pixel images in memory via
`model/synth.py`, with generator seed 0. For each image, pore fraction is
sampled uniformly from 0.30–0.70, disorder from 0.05–0.75, anisotropy from
0–0.6, and a target peak is sampled across the visible band before converting it
to a characteristic spacing clipped to 90–320 nm. Pixel scale is independently
sampled at 0.75–1.35 times `DEFAULT_PX_PER_UM`.

## Interpretation of Figure 5

The 20-run dispersion is **synthetic-realisation variability**: only the random
seed changes while the generator, spacing, pore fraction, disorder, image size,
and optical model remain fixed. It does not include uncertainty from imaging,
biological structural heterogeneity, target-label provenance, reflectance
measurement, or model misspecification. It is therefore neither an experimental
uncertainty interval nor a prediction interval for a biological specimen.
