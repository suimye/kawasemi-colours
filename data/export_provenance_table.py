#!/usr/bin/env python3
"""Export an auditable provenance table from dataset.jsonl.

The dataset intentionally retains records whose colour labels come from different
sources.  This export makes that heterogeneity visible rather than treating every
``peak_nm`` as a directly measured reflectance maximum.
"""
from __future__ import annotations

import csv
import json
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
INPUT = HERE / "dataset.jsonl"
OUTPUT = HERE / "provenance_table.csv"

FIELDS = [
    "id", "real_image", "species", "structure_type", "source_figure",
    "colour_source", "peak_source", "measurement_conditions",
    "value_status", "peak_nm", "reported_period_nm", "reported_period_source",
    "scale_px_per_um", "scale_status", "scale_provenance", "notes",
]

SOURCE_AUDIT = {
    "PMC4685390": {
        "measurement_conditions": "Direct feather reflectance: Ocean Optics USB2000+ with deuterium-tungsten-halogen source; normal-incidence illumination; three scans averaged (Parnell et al. 2015 Methods).",
        "peak_source": "The article reports spectra but the record's single 480-nm value is not a directly transcribed spectral maximum; treat it as author-derived until replaced by a digitized source spectrum.",
    },
    "PMC2706469": {
        "measurement_conditions": "TEM developmental-morphology study; no panel-specific reflectance measurement is recorded for the cropped channel/sphere panels.",
        "peak_source": "author-derived colour label; no panel-specific source peak identified",
    },
    "PMC8505553": {
        "measurement_conditions": "Review-derived figure or swatch; measurement geometry is not available from the record.",
        "peak_source": "author-derived from a review figure/swatch unless a primary measurement source is linked",
    },
}


def provenance(record: dict) -> dict:
    notes = record.get("notes", "")
    colour_marker = notes.partition("color_source=")[2]
    colour_marker = colour_marker.split(";", 1)[0].strip()
    peak = (record.get("color") or {}).get("peak_nm")
    scale = record.get("scale") or {}
    source = record.get("source") or {}
    source_key = source.get("pmcid") or ""
    audit = SOURCE_AUDIT.get(source_key, {})
    if "biorxiv.org/content/10.1101/2020.07.28.225862" in source.get("url", ""):
        audit = {
            "measurement_conditions": "Swallow tanager plumage reflectance was measured in the primary study; the published article reports a reflectance peak around 550 nm for both back and belly patches (Bazzano et al. 2021).",
            "peak_source": "Bazzano et al. (2021) abstract: reflectance peak around 550 nm for both back and belly patches; this approximate literature value is used here.",
        }
    calibrated = bool(scale.get("calibrated"))

    colour_metadata = (record.get("color") or {}).get("srgb_source")
    if colour_metadata:
        colour_source = colour_metadata
        value_status = "display-only author-derived swatch; not a spectral measurement"
    elif colour_marker:
        colour_source = colour_marker
        value_status = "author-derived from cited source description"
    elif peak is None:
        colour_source = "not recorded"
        value_status = "no peak value"
    else:
        colour_source = "record label; source method not yet structured"
        value_status = "manual source verification required"

    if audit.get("peak_source"):
        peak_source = audit["peak_source"]
    elif peak is None:
        peak_source = "not available"
    elif colour_marker:
        peak_source = "same cited text/caption source as colour label"
    else:
        peak_source = "record peak_nm; direct measurement status not yet structured"

    if calibrated:
        scale_status = "measured calibration"
        scale_provenance = "scale bar converted to px_per_um; see record notes"
    else:
        scale_status = "uncalibrated"
        scale_provenance = "no reliable in-panel scale; absolute lengths are heuristic unless a reported period is supplied"

    return {
        "id": record.get("id"),
        "real_image": record.get("real_image", False),
        "species": record.get("species"),
        "structure_type": record.get("structure_type"),
        "source_figure": (record.get("source") or {}).get("figure"),
        "colour_source": colour_source,
        "peak_source": peak_source,
        "measurement_conditions": audit.get("measurement_conditions", "not recorded in the current record; retrieve from the cited source before treating as a spectral measurement"),
        "value_status": value_status,
        "peak_nm": peak,
        "reported_period_nm": record.get("reported_period_nm"),
        "reported_period_source": record.get("reported_period_source"),
        "scale_px_per_um": scale.get("px_per_um"),
        "scale_status": scale_status,
        "scale_provenance": scale_provenance,
        "notes": notes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    records = [json.loads(line) for line in INPUT.read_text().splitlines() if line.strip()]
    rows = [provenance(record) for record in records]
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} provenance rows -> {args.output}")


if __name__ == "__main__":
    main()
