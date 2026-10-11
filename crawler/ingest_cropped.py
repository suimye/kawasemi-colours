#!/usr/bin/env python3
"""Crop real figure panels (per crawler/crops.py) and ingest them as REAL pixels.

- Bird records: crop the mapped SEM/TEM panel, replace the synthetic proxy in
  data/images/, re-extract FEATS_V1 from the real pixels, flip real_image:true.
- Fig6 (biomimetic melanin films): add NEW records — real SEM + sampled real
  colour swatch — as controlled structure->colour training pairs.

No manual renaming needed: mapping is by original figure filename in crops.py.
Synthetic originals are backed up to data/images/_synthetic_backup/.
Torch-free. Safe to re-run.
"""
import os, sys, json, csv, shutil
import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "model"))
sys.path.insert(0, os.path.join(ROOT, "crawler"))
from features import extract_features                       # noqa: E402
import colorutil as cu                                      # noqa: E402
from crops import (BIRD_CROPS, PX_PER_UM, FIG6_SRC, FIG6_SEM_Y, FIG6_SEM_H,   # noqa: E402
                   FIG6_SWATCH_Y, FIG6_SWATCH_H, FIG6_COLS, FIG6_IDS)

DATA = os.path.join(ROOT, "data")
DROP = os.path.join(DATA, "real_images")
IMAGES = os.path.join(DATA, "images")
BACKUP = os.path.join(IMAGES, "_synthetic_backup")
JSONL = os.path.join(DATA, "dataset.jsonl")

# scale-bar physical length + measured pixel length (measured from the figures)
# px_per_um lets features.py output absolute nm. Rough but better than nothing.
SCALE_CAL = {
    "20um": None,   # blue-tit SEM bars present but bar-px hard to fix; left uncalibrated
    "300nm": None,
    "200nm": None,
}


def load_records():
    with open(JSONL) as f:
        return [json.loads(l) for l in f if l.strip()]


def save_records(recs):
    with open(JSONL, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    cols = ["id", "image_file", "real_image", "pmcid", "doi", "title", "license",
            "url", "figure", "species", "common_name", "structure_type",
            "color_name", "color_hex", "color_srgb", "peak_nm", "char_length_nm",
            "pore_fraction", "scale_bar_um", "notes"]
    with open(os.path.join(DATA, "dataset.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in recs:
            s, c = r["source"], r["color"]
            w.writerow([r["id"], r["image_file"], r.get("real_image", False),
                        s.get("pmcid", ""), s.get("doi", ""), s.get("title", ""),
                        s.get("license", ""), s.get("url", ""), s.get("figure", ""),
                        r.get("species", ""), r.get("common_name", ""), r.get("structure_type", ""),
                        c.get("name", ""), c.get("hex", ""), "|".join(map(str, c.get("srgb", []))),
                        c.get("peak_nm", ""), r.get("features", {}).get("char_length_nm", ""),
                        r.get("features", {}).get("pore_fraction", ""),
                        r.get("scale", {}).get("bar_um", ""), r.get("notes", "")])


def ingest_birds(recs):
    by_id = {r["id"]: r for r in recs}
    done = []
    for rid, c in BIRD_CROPS.items():
        rec = by_id.get(rid)
        if rec is None:
            print("  ! no record for", rid); continue
        src = os.path.join(DROP, c["src"])
        im = cv2.imread(src)
        if im is None:
            print("  ! missing file", c["src"], "for", rid); continue
        x, y, w, h = c["box"]
        crop = im[y:y+h, x:x+w]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        base = os.path.splitext(os.path.basename(rec["image_file"]))[0] + ".png"
        old = os.path.join(IMAGES, base)
        if os.path.exists(old) and not rec.get("real_image"):
            shutil.copy2(old, os.path.join(BACKUP, base))
        cv2.imwrite(os.path.join(IMAGES, base), gray)
        rec["image_file"] = "images/" + base
        px_per_um = PX_PER_UM.get(rid)
        _, feat = extract_features(gray, px_per_um=px_per_um)
        rec["features"] = feat
        rec["real_image"] = True
        rec.setdefault("scale", {})["px_per_um"] = px_per_um
        rec["scale"]["calibrated"] = px_per_um is not None
        if px_per_um:
            cal = f"scale calibrated {px_per_um}px/um (bar {c.get('scale')})"
        else:
            cal = f"scale UNCALIBRATED (bar {c.get('scale')}; low-mag/under-resolved -> absolute nm relative)"
        base_note = rec.get("notes", "").replace("synthetic_proxy", "").strip(" ;")
        # drop any prior ingest note to avoid duplication on re-run
        base_note = base_note.split("real_image cropped from")[0].strip(" ;")
        rec["notes"] = f"real_image cropped from {c['src']} box={c['box']}; {cal}. {base_note}".strip()
        done.append((rid, feat.get("char_length_nm"), feat.get("anisotropy"), px_per_um))
    return done


def ingest_fig6(recs):
    src = os.path.join(DROP, FIG6_SRC)
    im = cv2.imread(src)
    if im is None:
        print("  ! Fig6 file missing:", FIG6_SRC); return []
    os.makedirs(BACKUP, exist_ok=True)
    existing = {r["id"] for r in recs}
    added = []
    for (x, w), name in zip(FIG6_COLS, FIG6_IDS):
        rid = f"PMC8505553_fig6d_{name}"
        sem = im[FIG6_SEM_Y:FIG6_SEM_Y+FIG6_SEM_H, x:x+w]
        sw = im[FIG6_SWATCH_Y:FIG6_SWATCH_Y+FIG6_SWATCH_H, x:x+w]
        bgr = sw.reshape(-1, 3).mean(0)
        srgb = [int(bgr[2]), int(bgr[1]), int(bgr[0])]
        peak, sat, val = cu.srgb8_to_peak_sat_val(srgb)
        gray = cv2.cvtColor(sem, cv2.COLOR_BGR2GRAY)
        base = rid + ".png"
        cv2.imwrite(os.path.join(IMAGES, base), gray)
        _, feat = extract_features(gray, px_per_um=None)
        rec = {
            "id": rid, "image_file": "images/" + base, "real_image": True,
            "source": {"pmcid": "PMC8505553",
                       "doi": "10.1186/s42649-021-00063-w",
                       "title": "A review of structural colour in bird feathers "
                                "(biomimetic melanin supraball films, Fig 6d)",
                       "license": "CC BY 4.0",
                       "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8505553/",
                       "figure": f"Figure 6d ({name})"},
            "species": "biomimetic (synthetic melanin nanoparticles)",
            "common_name": "artificial structural-colour film",
            "structure_type": "quasi_ordered_sphere",
            "color": {"name": "sampled", "hex": cu.rgb_to_hex(srgb),
                      "srgb": srgb, "peak_nm": round(peak, 1)},
            "scale": {"bar_um": None, "px_per_um": None},
            "features": feat,
            "notes": f"real_image cropped from {FIG6_SRC}; colour sampled from the "
                     f"paper's swatch under panel; pigment-free printed structural colour "
                     f"(melanin supraballs, sphere->ellipsoid series). peak~{peak:.0f}nm",
        }
        if rid in existing:
            for i, r in enumerate(recs):
                if r["id"] == rid:
                    recs[i] = rec
        else:
            recs.append(rec)
        added.append((rid, srgb, round(peak)))
    return added


def main():
    os.makedirs(BACKUP, exist_ok=True)
    recs = load_records()
    print("Ingesting real bird panels:")
    birds = ingest_birds(recs)
    for rid, cl, an, ppu in birds:
        tag = f"cal {ppu}px/um" if ppu else "uncalibrated"
        print(f"  {rid:34} char_length≈{cl:7.0f}nm  anisotropy={an:.2f}  [{tag}]")
    print("Adding Fig6 biomimetic structure->colour pairs:")
    fig6 = ingest_fig6(recs)
    for rid, srgb, pk in fig6:
        print(f"  {rid:34} rgb={tuple(srgb)}  peak≈{pk}nm")
    save_records(recs)
    n_real = sum(1 for r in recs if r.get("real_image"))
    print(f"\nDataset now: {n_real}/{len(recs)} records backed by REAL images "
          f"(+{len(fig6)} new biomimetic pairs).")
    print("Next: cd model && python train.py && python export_params.py && "
          "python ../scripts/adapt_model_for_ui.py")


if __name__ == "__main__":
    main()
