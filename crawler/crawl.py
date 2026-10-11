#!/usr/bin/env python3
"""
kawasemi_colors crawler
=======================

Discovers OPEN-ACCESS PubMed Central (PMC) papers about avian *structural
colour* produced by spongy / quasi-ordered keratin nanostructures (and the
related melanosome-array iridescence), pulls their figure captions, and
keyword-filters those captions for BOTH an electron-microscopy mention
(SEM/TEM) AND a colour / structural-colour mention. Papers and matching
figures are logged with full provenance (PMCID, DOI, title, licence, URL).

The script is deliberately conservative and re-runnable:

  * It only queries public NCBI E-utilities + the NCBI BioC PMC-OA API.
  * It writes a paper-level log (sources.csv) and a candidate figure log
    (data/figure_candidates.csv) so a human can curate the final dataset.
  * It caches every network response under crawler/.cache/ so re-runs are
    cheap and reproducible, and so a curator can re-parse without re-hitting
    the API.
  * It NEVER downloads figure pixels blindly. PMC gates figure binaries
    behind a browser check (reCAPTCHA); this script records the canonical
    figure-binary URL for each candidate so images can be fetched manually
    (or in an environment where the binaries are reachable) but does not
    pretend to have pixels it cannot get.

------------------------------------------------------------------------------
SANDBOX NOTE (important, read crawler/README.md):
The environment in which this dataset was originally assembled had NO direct
outbound network from the Python process (only a separate fetch tool could
reach the web). In that environment this script cannot run end-to-end; the
committed data/ files were assembled from the same API responses via that
fetch tool. In any NORMAL machine with internet access, `python crawl.py`
runs the full pipeline unaided.
------------------------------------------------------------------------------

Usage
-----
    python crawl.py                 # run discovery + caption filter
    python crawl.py --max 40        # cap papers examined
    python crawl.py --no-network    # only re-parse the local cache

Outputs
-------
    ../data/sources.csv                 paper-level provenance log
    ../data/figure_candidates.csv       every SEM+colour figure that passed the filter
    crawler/.cache/*.json               raw cached API responses
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import urllib.parse
import urllib.request

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
CACHE = os.path.join(HERE, ".cache")

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
BIOC = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json"
PMC_ARTICLE = "https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
PMC_FIG_BIN = "https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/bin/{fname}"

# Search terms tuned for avian structural-colour + electron microscopy.
# Each is run against PMC restricted to the open-access subset.
QUERIES = [
    'spongy keratin feather structural color "open access"[filter]',
    '(structural colour OR structural color) AND feather AND keratin '
    'AND (SEM OR TEM OR "electron microscopy") AND "open access"[filter]',
    'non-iridescent structural color bird feather quasi-ordered medullary '
    'keratin "open access"[filter]',
    'feather barb spongy keratin coherent scattering colour '
    '"open access"[filter]',
    'melanosome array feather iridescent structural colour electron '
    'microscopy "open access"[filter]',
]

# A caption must contain at least one term from EACH group to be a candidate.
EM_TERMS = [
    "sem", "tem", "electron micrograph", "electron microscop",
    "scanning electron", "transmission electron", "micrograph",
]
COLOR_TERMS = [
    "blue", "green", "violet", "purple", "cyan", "ultraviolet", "uv ",
    "iridescent", "structural colo", "reflectance", "hue", "spongy",
    "nanostructure", "melanosome", "photonic",
]

# Structure-type keywords -> DATA_CONTRACT structure_type enum.
STRUCTURE_KEYWORDS = [
    ("spongy", "spongy"),
    ("medullary", "spongy"),
    ("quasi-ordered", "spongy"),
    ("channel", "channel"),
    ("photonic crystal", "photonic_crystal"),
    ("melanosome", "melanosome_array"),
    ("melanin stack", "melanosome_array"),
    ("multilayer", "melanosome_array"),
]

NM_RE = re.compile(r"(\d{2,4})\s*nm")
SCALE_RE = re.compile(r"(?:scale\s*bar[s]?[^.\d]{0,20})?(\d+(?:\.\d+)?)\s*(?:μm|um|µm)")


# --------------------------------------------------------------------------- #
# Networking with on-disk cache
# --------------------------------------------------------------------------- #

def _cache_path(key: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", key)[:180]
    return os.path.join(CACHE, safe + ".json")


def fetch(url: str, key: str, use_network: bool) -> str | None:
    """Fetch `url`, caching by `key`. Returns text or None on failure."""
    cp = _cache_path(key)
    if os.path.exists(cp):
        with open(cp, encoding="utf-8") as fh:
            return fh.read()
    if not use_network:
        return None
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "kawasemi-colors-crawler/1.0"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            text = resp.read().decode("utf-8", "replace")
        os.makedirs(CACHE, exist_ok=True)
        with open(cp, "w", encoding="utf-8") as fh:
            fh.write(text)
        time.sleep(0.4)  # be polite to NCBI
        return text
    except Exception as exc:  # noqa: BLE001
        print(f"  ! fetch failed for {url}: {exc}")
        return None


# --------------------------------------------------------------------------- #
# PMC discovery
# --------------------------------------------------------------------------- #

def esearch(term: str, retmax: int, use_network: bool) -> list[str]:
    url = (
        f"{EUTILS}/esearch.fcgi?db=pmc&retmode=json&retmax={retmax}"
        f"&term={urllib.parse.quote(term)}"
    )
    txt = fetch(url, f"esearch_{term}", use_network)
    if not txt:
        return []
    try:
        return json.loads(txt)["esearchresult"].get("idlist", [])
    except Exception:  # noqa: BLE001
        return []


def esummary(uid: str, use_network: bool) -> dict:
    url = f"{EUTILS}/esummary.fcgi?db=pmc&retmode=json&id={uid}"
    txt = fetch(url, f"esummary_{uid}", use_network)
    if not txt:
        return {}
    try:
        return json.loads(txt)["result"][uid]
    except Exception:  # noqa: BLE001
        return {}


def article_meta(uid: str, use_network: bool) -> dict:
    """Return normalised paper metadata from an esummary record."""
    s = esummary(uid, use_network)
    if not s:
        return {}
    ids = {a["idtype"]: a["value"] for a in s.get("articleids", [])}
    pmcid = ids.get("pmcid", f"PMC{uid}")
    return {
        "uid": uid,
        "pmcid": pmcid,
        "doi": ids.get("doi", ""),
        "title": s.get("title", "").rstrip("."),
        "journal": s.get("fulljournalname", ""),
        "year": (s.get("pubdate", "") or "")[:4],
        "url": PMC_ARTICLE.format(pmcid=pmcid),
    }


# --------------------------------------------------------------------------- #
# BioC full text -> figures
# --------------------------------------------------------------------------- #

def bioc_figures(pmcid: str, use_network: bool) -> tuple[str, list[dict]]:
    """Return (licence, [figure dicts]) for a PMC article via the BioC API."""
    url = f"{BIOC}/{pmcid}/unicode"
    txt = fetch(url, f"bioc_{pmcid}", use_network)
    if not txt:
        return "", []
    # Strip any web-fetch header prefix and locate the JSON array.
    idx = txt.find("[{")
    if idx > 0:
        txt = txt[idx:]
    try:
        data = json.loads(txt)
        doc = data[0]["documents"][0]
        licence = doc.get("infons", {}).get("license", "")
        # BioC splits a figure into a short "fig" (title) passage and a longer
        # "fig_caption" passage. Merge by file, keeping the longest text.
        by_file: dict[str, dict] = {}
        order: list[str] = []
        for p in doc.get("passages", []):
            inf = p.get("infons", {})
            if inf.get("type") not in ("fig", "fig_caption"):
                continue
            key = inf.get("file") or inf.get("id") or f"_{len(order)}"
            text = p.get("text", "")
            if key not in by_file:
                by_file[key] = {"file": inf.get("file", ""),
                                "id": inf.get("id", ""), "caption": text}
                order.append(key)
            elif len(text) > len(by_file[key]["caption"]):
                by_file[key]["caption"] = text
        return licence, [by_file[k] for k in order]
    except Exception:  # noqa: BLE001
        # Truncated / malformed JSON (common when a fetch was cut off in the
        # reference section). Salvage figure passages + licence with regex.
        return _salvage_figs(txt)


def _salvage_figs(txt: str) -> tuple[str, list[dict]]:
    """Best-effort recovery of licence + figure captions from partial JSON."""
    lic_m = re.search(r'"license"\s*:\s*"([^"]{0,120})"', txt)
    licence = lic_m.group(1) if lic_m else ""
    by_file: dict[str, dict] = {}
    order: list[str] = []
    # Match a fig / fig_caption infons block followed by its "text".
    for m in re.finditer(
        r'"infons"\s*:\s*\{([^{}]*?"type"\s*:\s*"fig(?:_caption)?"[^{}]*?)\}'
        r'\s*,\s*"text"\s*:\s*"((?:[^"\\]|\\.)*)"', txt
    ):
        infons, caption = m.group(1), m.group(2)
        fm = re.search(r'"file"\s*:\s*"([^"]*)"', infons)
        im = re.search(r'"id"\s*:\s*"([^"]*)"', infons)
        try:
            caption = caption.encode().decode("unicode_escape", "replace")
        except Exception:  # noqa: BLE001
            pass
        key = (fm.group(1) if fm else "") or (im.group(1) if im else "") \
            or f"_{len(order)}"
        if key not in by_file:
            by_file[key] = {"file": fm.group(1) if fm else "",
                            "id": im.group(1) if im else "",
                            "caption": caption}
            order.append(key)
        elif len(caption) > len(by_file[key]["caption"]):
            by_file[key]["caption"] = caption
    return licence, [by_file[k] for k in order]


def classify_structure(caption: str) -> str:
    low = caption.lower()
    for kw, st in STRUCTURE_KEYWORDS:
        if kw in low:
            return st
    return "unknown"


def caption_passes(caption: str) -> bool:
    low = caption.lower()
    has_em = any(t in low for t in EM_TERMS)
    has_color = any(t in low for t in COLOR_TERMS)
    return has_em and has_color


def extract_numbers(caption: str) -> dict:
    nms = [int(m) for m in NM_RE.findall(caption)]
    scales = [float(m) for m in SCALE_RE.findall(caption)]
    return {
        "nm_values": ";".join(str(n) for n in nms),
        "scale_um_values": ";".join(str(s) for s in scales),
    }


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max", type=int, default=60,
                    help="max papers to examine across all queries")
    ap.add_argument("--retmax", type=int, default=40,
                    help="results per query")
    ap.add_argument("--no-network", action="store_true",
                    help="use only cached responses")
    args = ap.parse_args()
    use_net = not args.no_network

    os.makedirs(DATA, exist_ok=True)
    os.makedirs(CACHE, exist_ok=True)

    # 1. discover UIDs across all queries (dedup, keep order)
    seen, uids = set(), []
    for q in QUERIES:
        print(f"[esearch] {q}")
        for uid in esearch(q, args.retmax, use_net):
            if uid not in seen:
                seen.add(uid)
                uids.append(uid)
        if len(uids) >= args.max:
            break
    uids = uids[: args.max]
    print(f"  -> {len(uids)} unique candidate papers")

    # 2. for each paper: metadata + figure captions, keyword filter
    source_rows, fig_rows = [], []
    for uid in uids:
        meta = article_meta(uid, use_net)
        if not meta:
            continue
        licence, figs = bioc_figures(meta["pmcid"], use_net)
        matches = [f for f in figs if caption_passes(f["caption"])]
        if not matches:
            continue  # skip papers with no SEM+colour figure
        print(f"[hit] {meta['pmcid']}  {len(matches)} fig(s)  {meta['title'][:60]}")
        source_rows.append(
            {
                "pmcid": meta["pmcid"],
                "doi": meta["doi"],
                "title": meta["title"],
                "license": licence,
                "url": meta["url"],
                "why_selected": (
                    f"{len(matches)} figure caption(s) mention both electron "
                    f"microscopy and colour/structural-colour; journal "
                    f"{meta['journal']} ({meta['year']})."
                ),
            }
        )
        for f in matches:
            nums = extract_numbers(f["caption"])
            fig_rows.append(
                {
                    "pmcid": meta["pmcid"],
                    "figure_file": f["file"],
                    "figure_id": f["id"],
                    "structure_type": classify_structure(f["caption"]),
                    "nm_values": nums["nm_values"],
                    "scale_um_values": nums["scale_um_values"],
                    "figure_binary_url": PMC_FIG_BIN.format(
                        pmcid=meta["pmcid"], fname=f["file"]
                    )
                    if f["file"]
                    else "",
                    "caption": f["caption"].replace("\n", " ").strip(),
                }
            )

    # 3. write logs
    with open(os.path.join(DATA, "sources.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["pmcid", "doi", "title", "license", "url",
                        "why_selected"],
        )
        w.writeheader()
        w.writerows(source_rows)

    with open(os.path.join(DATA, "figure_candidates.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["pmcid", "figure_file", "figure_id", "structure_type",
                        "nm_values", "scale_um_values", "figure_binary_url",
                        "caption"],
        )
        w.writeheader()
        w.writerows(fig_rows)

    print(f"\nDone. {len(source_rows)} papers, {len(fig_rows)} candidate figures.")
    print("Curate figure_candidates.csv into dataset.jsonl / dataset.csv "
          "(see build_dataset.py for how the committed dataset was produced).")


if __name__ == "__main__":
    main()
