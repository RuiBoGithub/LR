#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, unicodedata, sys
from typing import Optional, List, Tuple
import pandas as pd
import requests
from lxml import etree as ET
from _credentials import keys

# ---------------------------
# Config
# ---------------------------
CSV_PATH   = "csv_output/300_manually_filtered/_verification-list.csv"
OUT_ROOT   = "elsevier_out"
ACCEPT     = "xml"          # or "json" (we'll try to discover embedded XML)
TIMEOUT    = 30
TARGET_ONLY = True          # Keep only Methods/Results/(Results and Discussion)/Abstract/Conclusion

API_KEY   = keys.get("els-apikey")
INST_TOKEN = keys.get("els-insttoken", "")

RAW_DIR   = os.path.join(OUT_ROOT, "papers_raw")
EXTR_DIR  = os.path.join(OUT_ROOT, "papers_extracted")
FTRR_DIR  = os.path.join(OUT_ROOT, "papers_ftrr")
for d in (RAW_DIR, EXTR_DIR, FTRR_DIR):
    os.makedirs(d, exist_ok=True)

# ---------------------------
# Fetcher
# ---------------------------
def fetch_article(doi: str):
    url = f"https://api.elsevier.com/content/article/doi/{doi}"
    headers = {"X-ELS-APIKey": API_KEY,
               "Accept": "application/json" if ACCEPT == "json" else "application/xml"}
    if INST_TOKEN:
        headers["X-ELS-Insttoken"] = INST_TOKEN
    params = {"view": "FULL"}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=TIMEOUT)
    except requests.RequestException as e:
        return 0, b"", f"Request error: {e}", None
    if r.status_code == 200 and r.content:
        return 200, r.content, None, r.headers.get("Content-Type", "")
    return r.status_code, r.content or b"", f"HTTP {r.status_code}: {r.text[:300]}", r.headers.get("Content-Type", "")

# ---------------------------
# Label helpers
# ---------------------------
def norm_title_match(s: str) -> str:
    if not s: return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-z0-9\s&\-]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()

METHODS_PATTERNS = [
    r"\bmethods?\b", r"\bmethodolog(y|ies)\b", r"\bmaterials?\s+and\s+methods?\b",
    r"\bexperimental(\s+(setup|procedure|procedures|design))?\b",
    r"\bapproach\b", r"\bdata\s*collection\b", r"\bdata\s*(and|&)\s*methods\b",
    r"\bstudy\s*design\b", r"\bprotocol\b", r"\bimplementation\b", r"\bprocedures?\b",
    r"\b(framework|conceptual\s+framework|system\s+framework|methodological\s+framework)\b",
    r"\b(system|model|architecture)\s+(overview|design|description)\b",
]
RESULTS_PATTERNS = [
    r"\bresults?\b", r"\bresults?\s*(and|&)\s*analysis\b", r"\bfindings?\b",
    r"\bevaluation\b", r"\bempirical\s+results?\b", r"\boutcomes?\b",
    r"\bperformance\b", r"\bexperiments?\b", r"\bresults?\s*(and|&)\s*(experiments?|evaluation)\b",
]
CONCLUSION_PATTERNS = [
    r"\bconclusions?\b", r"\bconcluding\s+remarks?\b", r"\bsummary\b",
    r"\bconclusion(s|)\s+and\s+future\s+work\b", r"\bfinal\s+remarks?\b",
]
MIXED_RESULTS_DISCUSSION = [r"\bresults?\s*(and|&)\s*discussion\b", r"\bdiscussion\s*(and|&)\s*results?\b"]
HARD_STOP_PATTERNS = [r"\breferences?\b", r"\bbibliograph(y|ies)\b", r"\bappendix\b", r"\bsupplement(ary|al)?\b"]

def label_title(title_norm: str) -> Optional[str]:
    if any(re.search(p, title_norm) for p in MIXED_RESULTS_DISCUSSION):
        return "Results and Discussion"
    if any(re.search(p, title_norm) for p in METHODS_PATTERNS):
        return "Methods"
    if any(re.search(p, title_norm) for p in RESULTS_PATTERNS):
        return "Results"
    if any(re.search(p, title_norm) for p in CONCLUSION_PATTERNS):
        return "Conclusion"
    return None  # not target

def should_hard_stop(title_norm: str) -> bool:
    return any(re.search(p, title_norm) for p in HARD_STOP_PATTERNS)

# ---------------------------
# XML parsing
# ---------------------------
NS = {
    "ce": "http://www.elsevier.com/xml/common/dtd",
    "ja": "http://www.elsevier.com/xml/ja/dtd",
    "dc": "http://purl.org/dc/elements/1.1/",
}

def elem_text(e) -> str:
    if e is None: return ""
    return re.sub(r"\s+", " ", " ".join(e.itertext())).strip()

def section_title(sec) -> str:
    st = sec.find("ce:section-title", NS)
    return elem_text(st) if st is not None else ""

def walk_sections_in_order(root):
    def recurse(sec, path):
        t = section_title(sec)
        yield (t, sec)
        for sub in sec.findall("ce:section", NS):
            yield from recurse(sub, path + [t])
    top = root.findall(".//ce:sections/ce:section", NS) or root.findall(".//ce:section", NS)
    for s in top:
        p = s.getparent()
        if p is None or not str(p.tag).endswith("section"):
            yield from recurse(s, [])

def find_article_root(root):
    for tag in ["ja:article", "article"]:
        found = root.find(f".//{tag}", NS)
        if found is not None:
            return found
    return root

def extract_sections_from_xml(xml_bytes: bytes) -> List[Tuple[str, str]]:
    """
    Returns [(Heading, Body)] for ALL sections (with normalized headings for target ones),
    stopping at back-matter.
    """
    root = ET.fromstring(xml_bytes)
    art = find_article_root(root)
    sections: List[Tuple[str, str]] = []

    # Include abstract first (unchanged label)
    for abs_ in art.findall(".//ce:abstract", NS):
        txt = elem_text(abs_)
        if txt: sections.append(("Abstract", txt))

    stopped = False
    for title, sec in walk_sections_in_order(art):
        t_norm = norm_title_match(title)
        if not stopped and should_hard_stop(t_norm):
            stopped = True
        if stopped:
            continue
        body = elem_text(sec)
        if not body:
            continue

        # normalize heading if it's a target, else keep the original as-is
        normalized = label_title(t_norm) or (title.strip() or "Section")
        sections.append((normalized, body))

    if not sections:
        whole = elem_text(art)
        if whole: sections = [("Body", whole)]
    return sections

def filter_target_sections(sections: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Keep only Methods/Results/Results and Discussion/Abstract/Conclusion, preserving order."""
    kept = []
    for head, body in sections:
        hnorm = norm_title_match(head)
        # Check if it's a target section or abstract
        if head == "Abstract" or label_title(hnorm) in {"Methods", "Results", "Results and Discussion", "Conclusion"}:
            # Use the normalized canonical label for target sections, keep 'Abstract' as-is
            canon = label_title(hnorm) if head != "Abstract" else "Abstract"
            kept.append((canon, body))
    return kept

def stringify_sections(sections: List[Tuple[str, str]]) -> str:
    # One clean heading line, then body, blank line between sections
    parts = []
    for head, body in sections:
        parts.append(f"{head}\n{body}\n")
    return "\n".join(parts).strip() + "\n"

# ---------------------------
# Main
# ---------------------------
def get_doi_column(df: pd.DataFrame) -> str:
    for cand in ["DOI", "doi", "Doi"]:
        if cand in df.columns:
            return cand
    raise KeyError("No DOI column found (expected DOI/doi/Doi)")

def main():

    if not API_KEY:
        print("ERROR: Missing API key in _credentials.py", file=sys.stderr)
        sys.exit(2)

    df = pd.read_csv(CSV_PATH)
    doi_col = get_doi_column(df)
    df = df.drop_duplicates(subset=[doi_col]).reset_index(drop=True)

    no_mr_rows = []
    ok_count = 0

    for i, row in df.iterrows():
        doi = str(row[doi_col]).strip()
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", doi)
        print(f"[{i+1}/{len(df)}] {doi}")

        status, raw_bytes, err, ctype = fetch_article(doi)
        raw_ext = ".json" if (ACCEPT == "json" or (ctype and "json" in ctype.lower())) else ".xml"
        raw_path = os.path.join(RAW_DIR, f"{safe}{raw_ext}")
        if status == 200 and raw_bytes:
            with open(raw_path, "wb") as f: f.write(raw_bytes)

        normalized_text = None
        error_msg = err
        has_target = False

        if status == 200 and raw_bytes and raw_ext == ".xml":
            try:
                all_sections = extract_sections_from_xml(raw_bytes)
                if TARGET_ONLY:
                    target_sections = filter_target_sections(all_sections)
                    has_target = len(target_sections) > 0
                    normalized_text = stringify_sections(target_sections)
                else:
                    normalized_text = stringify_sections(all_sections)

                if TARGET_ONLY and not has_target:
                    no_mr_rows.append({"DOI": doi, "HTTP_status": status, "Error": error_msg, "Raw_File": raw_path})
            except Exception as e:
                error_msg = f"XML parse error: {e}"
        else:
            no_mr_rows.append({"DOI": doi, "HTTP_status": status, "Error": error_msg, "Raw_File": raw_path})

        if normalized_text is None:
            normalized_text = ""  # keep the JSON shape even if empty

        # Write normalized JSON for classifier
        ftrr_path = os.path.join(FTRR_DIR, f"{safe}.json")
        with open(ftrr_path, "w", encoding="utf-8") as f:
            json.dump({"full-text-retrieval-response": {"originalText": normalized_text}},
                      f, ensure_ascii=False, indent=2)
        ok_count += 1

        # Sidecar (status/paths)
        sidecar = {"doi": doi, "status": status, "error": error_msg,
                   "raw_file": raw_path, "ftrr_file": ftrr_path,
                   "target_only": TARGET_ONLY, "has_target": has_target}
        with open(os.path.join(EXTR_DIR, f"{safe}.extraction.json"), "w", encoding="utf-8") as f:
            json.dump(sidecar, f, ensure_ascii=False, indent=2)

    # CSV for items with no target sections (or fetch failure)
    pd.DataFrame(no_mr_rows).to_csv(os.path.join(EXTR_DIR, "no_methods_results.csv"), index=False)
    print(f"[OK] Wrote normalized JSONs → {FTRR_DIR}")
    print(f"[OK] Logged items without target sections → {os.path.join(EXTR_DIR, 'no_methods_results.csv')}")

if __name__ == "__main__":
    main()