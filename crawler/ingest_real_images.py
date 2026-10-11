#!/usr/bin/env python3
"""Ingest REAL figure images you downloaded by hand, replacing synthetic proxies.

Workflow
--------
1. Download the real figure image from its paper URL (see data/real_images/README.md).
2. Crop the relevant SEM/TEM panel and save it into  data/real_images/
   naming it EXACTLY after the target record's image filename, e.g.
     PMC8599086_fig1c_bluevane.png
   (run this script with --list to print every valid target name.)
   A full uncropped figure named <PMCID>_<figure>.png is also accepted but you
   should crop to the nanostructure panel for best feature extraction.
3. (optional) Add scale calibration in data/real_images/scale.csv:
      filename,px_per_um
   so absolute nm values (period, pore size) are physically correct.
4. Run:  python3 crawler/ingest_real_images.py
   -> real images are copied into data/images/, features re-extracted from the
      REAL pixels, records flagged real_image:true, dataset.jsonl/csv rewritten.
      The old synthetic images are backed up to data/images/_synthetic_backup/.

Torch-free. Safe to re-run.
"""
import os, sys, json, csv, shutil, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "model"))
import cv2  # noqa: E402
from features import extract_features, FEATS_V1  # noqa: E402

DATA = os.path.join(ROOT, "data")
DROP = os.path.join(DATA, "real_images")
IMAGES = os.path.join(DATA, "images")
BACKUP = os.path.join(IMAGES, "_synthetic_backup")
JSONL = os.path.join(DATA, "dataset.jsonl")
CSV = os.path.join(DATA, "dataset.csv")
EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")


def load_records():
    recs = []
    with open(JSONL) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def load_scale():
    p = os.path.join(DROP, "scale.csv")
    out = {}
    if os.path.exists(p):
        with open(p) as f:
            for r in csv.DictReader(f):
                try:
                    out[r["filename"].strip()] = float(r["px_per_um"])
                except (KeyError, ValueError):
                    pass
    return out


def basename_noext(name):
    return os.path.splitext(os.path.basename(name))[0].lower()


def match_record(fname, recs):
    """Match a dropped file to a record. Exact basename match first, then prefix."""
    stem = basename_noext(fname)
    by_base = {basename_noext(r["image_file"]): r for r in recs}
    if stem in by_base:
        return by_base[stem]
    # prefix match: dropped file starts with a record's basename (allows _crop suffix)
    for base, r in by_base.items():
        if stem.startswith(base) or base.startswith(stem):
            return r
    return None


def write_dataset(recs):
    with open(JSONL, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # CSV: flatten, keep original header if present
    cols = ["id", "image_file", "real_image", "pmcid", "doi", "title", "license",
            "url", "figure", "species", "common_name", "structure_type",
            "color_name", "color_hex", "color_srgb", "peak_nm", "char_length_nm",
            "pore_fraction", "scale_bar_um", "notes"]
    with open(CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in recs:
            s, c = r["source"], r["color"]
            w.writerow([r["id"], r["image_file"], r.get("real_image", False),
                        s.get("pmcid", ""), s.get("doi", ""), s.get("title", ""),
                        s.get("license", ""), s.get("url", ""), s.get("figure", ""),
                        r.get("species", ""), r.get("common_name", ""),
                        r.get("structure_type", ""), c.get("name", ""),
                        c.get("hex", ""), "|".join(map(str, c.get("srgb", []))),
                        c.get("peak_nm", ""), r.get("features", {}).get("char_length_nm", ""),
                        r.get("features", {}).get("pore_fraction", ""),
                        r.get("scale", {}).get("bar_um", ""), r.get("notes", "")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="print valid target filenames and exit")
    args = ap.parse_args()

    recs = load_records()
    if args.list:
        print("Drop real images into data/real_images/ named after these targets:\n")
        for r in recs:
            print(f"  {os.path.basename(r['image_file']):40}  {r['common_name']} / {r['color']['name']}  [{r['source'].get('pmcid','')}]")
        return

    os.makedirs(BACKUP, exist_ok=True)
    scale = load_scale()
    dropped = [f for f in os.listdir(DROP)
               if f.lower().endswith(EXTS) and os.path.isfile(os.path.join(DROP, f))]
    if not dropped:
        print("No images found in data/real_images/. Add files then re-run. (--list shows target names)")
        return

    updated, skipped = [], []
    for fname in sorted(dropped):
        rec = match_record(fname, recs)
        if rec is None:
            skipped.append((fname, "no matching record name"))
            continue
        src = os.path.join(DROP, fname)
        img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
        if img is None:
            skipped.append((fname, "could not read image"))
            continue
        dst_base = os.path.basename(rec["image_file"])
        # backup synthetic
        old = os.path.join(IMAGES, dst_base)
        if os.path.exists(old) and not rec.get("real_image", False):
            shutil.copy2(old, os.path.join(BACKUP, dst_base))
        dst = os.path.join(IMAGES, os.path.splitext(dst_base)[0] + ".png")
        cv2.imwrite(dst, img)
        rec["image_file"] = "images/" + os.path.splitext(dst_base)[0] + ".png"

        px_per_um = scale.get(fname) or scale.get(dst_base)
        if px_per_um is None and rec.get("scale", {}).get("px_per_um"):
            px_per_um = rec["scale"]["px_per_um"]
        _, feat = extract_features(img, px_per_um=px_per_um)
        rec["features"] = feat
        rec["real_image"] = True
        note = rec.get("notes", "")
        note = note.replace("synthetic_proxy", "real_image").strip()
        cal = f" (px_per_um={px_per_um})" if px_per_um else " (uncalibrated scale — add scale.csv for accurate nm)"
        rec["notes"] = ("real_image ingested from " + fname + cal + "; " + note).strip("; ")
        updated.append((fname, dst_base, feat.get("char_length_nm")))

    write_dataset(recs)

    print(f"\nIngested {len(updated)} real image(s):")
    for fname, base, cl in updated:
        print(f"  {fname:34} -> {base:34} char_length≈{cl:.0f}nm" if cl else f"  {fname} -> {base}")
    if skipped:
        print(f"\nSkipped {len(skipped)}:")
        for fname, why in skipped:
            print(f"  {fname:34} ({why})")
    n_real = sum(1 for r in recs if r.get("real_image"))
    print(f"\nDataset now: {n_real}/{len(recs)} records backed by REAL images.")
    print("Next: cd model && python train.py   (re-trains/fine-tunes on the real data)")
    print("      python export_params.py && python ../scripts/adapt_model_for_ui.py")


if __name__ == "__main__":
    main()
