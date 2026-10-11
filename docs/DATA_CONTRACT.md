# Kawasemi Colors — Shared Data Contract (v1)

All components (crawler, model, simulator) MUST conform to this schema.

## Directory layout
```
kawasemi_colors/
  data/
    raw_figures/        # full figure images downloaded from papers
    images/             # cropped SEM panels used for training (PNG, grayscale ok)
    dataset.jsonl       # one JSON object per line (see record schema)
    dataset.csv         # flattened convenience copy
    sources.csv         # paper-level provenance / license log
  crawler/              # OWNER: crawler agent
  model/                # OWNER: model agent
  simulator/            # OWNER: ui agent
  docs/
```

## Record schema (one per training image) — dataset.jsonl
```json
{
  "id": "PMC1234567_fig3b",
  "image_file": "images/PMC1234567_fig3b.png",
  "source": {
    "pmcid": "PMC1234567",
    "doi": "10.xxxx/xxxx",
    "title": "Paper title",
    "license": "CC BY 4.0",
    "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1234567/",
    "figure": "Figure 3B"
  },
  "species": "Alcedo atthis",
  "common_name": "common kingfisher",
  "structure_type": "spongy",          // one of: spongy | channel | quasi_ordered_sphere | photonic_crystal | melanosome_array | unknown
  "color": {
    "name": "blue",
    "hex": "#1f6fb0",
    "srgb": [31, 111, 176],
    "peak_nm": 470                       // dominant reflectance peak; null if unknown
  },
  "scale": { "bar_um": 1.0, "px_per_um": null },
  "notes": "spongy medullary keratin, non-iridescent structural blue"
}
```

## Structural feature vector (produced by model/features.py from each image)
Fixed order, all floats. This is the bridge between images and the color model.
```
[ char_length_nm,        # dominant spatial period (nm) from radial FFT peak
  domain_std_nm,         # spread of the period (order/disorder)
  pore_fraction,         # air/void volume fraction (0..1) via threshold
  pore_diameter_nm,      # mean pore size
  wall_thickness_nm,     # mean solid wall thickness
  anisotropy,            # 0=isotropic(sponge) .. 1=aligned(channels/iridescent)
  connectivity ]         # Euler/skeleton-based network connectivity proxy
```
Feature spec version: FEATS_V1 (7 dims). Any change bumps the version and all agents update.

## Color convention
- Colors stored as sRGB 8-bit + hex + optional dominant wavelength (peak_nm).
- Simulator target color -> convert to (peak_nm, saturation, brightness) before inverse model.

## Physics note (shared prior)
Structural color from a quasi-ordered spongy keratin network is dominated by
coherent scattering at short range. First-order approximation:
  peak_wavelength ≈ 2 * n_eff * d
where d = characteristic center-to-center spacing (char_length_nm) and
n_eff = volume-weighted refractive index of keratin (~1.56) and air (1.0):
  n_eff = pore_fraction*1.0 + (1-pore_fraction)*1.56
Saturation grows with order (smaller domain_std_nm). Agents may refine this,
but the simulator's fallback inverse must use this relation so it works
without a trained checkpoint.
