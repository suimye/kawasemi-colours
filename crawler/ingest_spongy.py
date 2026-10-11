#!/usr/bin/env python3
"""Ingest the newly added SPONGY / channel / sphere structural-colour figures as
NEW real records (kingfisher, swallow tanager, Prum channel/sphere). These fill
the quasi-ordered spongy gap that the earlier melanosome-heavy set lacked.

Crops verified visually. Scale: most panels ingested uncalibrated (absolute nm
relative) because bars were ambiguous in-crop; scale-free features (order,
anisotropy, pore fraction) remain valid. Where a bar value is known and
measurable it is applied.
Torch-free, safe to re-run.
"""
import os, sys, json, csv, cv2
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "model"))
sys.path.insert(0, os.path.join(ROOT, "crawler"))
from features import extract_features        # noqa: E402
import colorutil as cu                        # noqa: E402

DATA = os.path.join(ROOT, "data")
DROP = os.path.join(DATA, "real_images")
IMAGES = os.path.join(DATA, "images")
JSONL = os.path.join(DATA, "dataset.jsonl")

# id, src, box(x,y,w,h), px_per_um(None=uncal), species, common, structure, color_name, srgb, peak, license, url, figure, notes
NEW = [
 ("STAVENGA2011_king_fig3F","kingfisher_stavenga.png",(883,1450,300,235),None,
  "Alcedo atthis","common kingfisher","spongy","cyan-blue",[36,150,178],485,
  "not CC (green-OA)","https://pure.rug.nl/ws/files/6770003/2011JExpBiolStavenga.pdf",
  "Fig 3F (tail barb spongy, 2um)","resolved spongy keratin mesh producing cyan/blue; project namesake"),
 ("BIORXIV_tersina_fig6c","swtanager_fig6.png",(240,780,300,300),None,
  "Tersina viridis","swallow tanager","spongy","greenish-blue",[70,150,150],550,
  "CC BY","https://www.biorxiv.org/content/10.1101/2020.07.28.225862v1",
  "Fig 6c (SEM spongy matrix)","greenish-blue back feather spongy keratin; author-assigned illustrative display swatch (not spectrophotometrically calibrated); peak_nm=550 is an approximate primary-study reflectance maximum"),
 ("BIORXIV_tersina_fig6b_inset","swtanager_fig6.png",(1152,112,166,150),None,
  "Tersina viridis","swallow tanager","quasi_ordered_sphere","greenish-blue",[70,150,150],550,
  "CC BY","https://www.biorxiv.org/content/10.1101/2020.07.28.225862v1",
  "Fig 6b inset (high-mag sphere-type air array)","quasi-ordered air spheres in keratin; author-assigned illustrative display swatch (not spectrophotometrically calibrated); peak_nm=550 is an approximate primary-study reflectance maximum"),
 ("BIORXIV_tersina_fig6d_inset","swtanager_fig6.png",(1168,715,152,150),None,
  "Tersina viridis","swallow tanager","quasi_ordered_sphere","greenish-blue",[70,150,150],550,
  "CC BY","https://www.biorxiv.org/content/10.1101/2020.07.28.225862v1",
  "Fig 6d inset (high-mag sphere-type array)","quasi-ordered air spheres in keratin; author-assigned illustrative display swatch (not spectrophotometrically calibrated); peak_nm=550 is an approximate primary-study reflectance maximum"),
 ("PMC2706469_fig1e_channel","PMC2706469_fig1.jpg",(8,435,205,198),None,
  "Aves (Prum 2009, channel-type)","structural-blue barb (channel-type)","channel","blue",[40,90,180],470,
  "CC BY 2.0","https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2706469/",
  "Fig 1e (high-mag channel beta-keratin, 120nm bar)","channel-type quasi-ordered keratin; per-panel species not confirmed from caption"),
 ("PMC2706469_fig1a_channel","PMC2706469_fig1.jpg",(5,5,205,205),None,
  "Aves (Prum 2009, channel-type)","structural-blue barb (channel-type)","channel","blue",[40,90,180],470,
  "CC BY 2.0","https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2706469/",
  "Fig 1a (channel beta-keratin)","channel-type; species per-panel not confirmed"),
 ("PMC2706469_fig1c_sphere","PMC2706469_fig1.jpg",(220,5,205,205),None,
  "Aves (Prum 2009, sphere-type)","structural-blue barb (sphere-type)","quasi_ordered_sphere","blue",[45,95,175],460,
  "CC BY 2.0","https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2706469/",
  "Fig 1c (sphere-type beta-keratin)","sphere-type; species per-panel not confirmed"),
]


def load(): return [json.loads(l) for l in open(JSONL) if l.strip()]


def save(recs):
    with open(JSONL, "w") as f:
        for r in recs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    cols = ["id","image_file","real_image","pmcid","doi","title","license","url","figure",
            "species","common_name","structure_type","color_name","color_hex","color_srgb",
            "peak_nm","char_length_nm","pore_fraction","scale_bar_um","notes"]
    with open(os.path.join(DATA, "dataset.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in recs:
            s, c = r["source"], r["color"]
            w.writerow([r["id"], r["image_file"], r.get("real_image", False), s.get("pmcid",""),
                        s.get("doi",""), s.get("title",""), s.get("license",""), s.get("url",""),
                        s.get("figure",""), r.get("species",""), r.get("common_name",""),
                        r.get("structure_type",""), c.get("name",""), c.get("hex",""),
                        "|".join(map(str, c.get("srgb", []))), c.get("peak_nm",""),
                        r.get("features",{}).get("char_length_nm",""),
                        r.get("features",{}).get("pore_fraction",""),
                        r.get("scale",{}).get("bar_um",""), r.get("notes","")])


def main():
    recs = load(); by_id = {r["id"]: i for i, r in enumerate(recs)}
    added = 0
    for (rid, src, box, ppu, sp, cn, st, coln, srgb, peak, lic, url, fig, note) in NEW:
        im = cv2.imread(os.path.join(DROP, src))
        if im is None:
            print("  ! missing", src); continue
        x, y, w, h = box
        gray = cv2.cvtColor(im[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        base = rid + ".png"; cv2.imwrite(os.path.join(IMAGES, base), gray)
        _, feat = extract_features(gray, px_per_um=ppu)
        rec = {"id": rid, "image_file": "images/"+base, "real_image": True,
               "source": {"pmcid": rid.split("_")[0] if rid.startswith("PMC") else "",
                          "doi": "", "title": fig.split("(")[0].strip(), "license": lic,
                          "url": url, "figure": fig},
               "species": sp, "common_name": cn, "structure_type": st,
               "color": {"name": coln, "hex": cu.rgb_to_hex(srgb), "srgb": srgb, "peak_nm": peak},
               "scale": {"bar_um": None, "px_per_um": ppu, "calibrated": ppu is not None},
               "features": feat,
               "notes": f"real_image cropped from {src} box={box}; "
                        f"{'calibrated '+str(ppu)+'px/um' if ppu else 'scale uncalibrated (bar ambiguous in-crop; absolute nm relative)'}. {note}"}
        if rid.startswith("BIORXIV_tersina"):
            # A peak maximum alone does not determine CIE/sRGB coordinates.  The
            # retained swatch is display context only, so these records cannot
            # supervise the model's saturation/value targets.
            rec["colour_training_eligible"] = False
            rec["color"]["srgb_source"] = (
                "author-assigned illustrative display swatch; not spectrophotometrically calibrated"
            )
            rec["color"]["spectrally_calibrated"] = False
        if rid in by_id: recs[by_id[rid]] = rec
        else: recs.append(rec); added += 1
        print(f"  {rid:32} char_length≈{feat['char_length_nm']:6.0f}nm order={1-feat['domain_std_nm']/max(feat['char_length_nm'],1):.2f} aniso={feat['anisotropy']:.2f}")
    save(recs)
    real = sum(1 for r in recs if r.get("real_image"))
    spongy = sum(1 for r in recs if r.get("real_image") and r["structure_type"] in ("spongy","channel","quasi_ordered_sphere"))
    print(f"\nAdded {added} new real spongy/channel/sphere records.")
    print(f"Dataset: {len(recs)} total | {real} real images | {spongy} real quasi-ordered(spongy-type)")


if __name__ == "__main__":
    main()
