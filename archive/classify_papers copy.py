#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, yaml, argparse
from collections import defaultdict
import concurrent.futures as cf

import nltk
from gensim.models import Word2Vec
import pandas as pd
from tqdm import tqdm

# ---------------- NLTK bootstrap ----------------
def ensure_nltk():
    pkgs = [
        ("tokenizers/punkt", "punkt"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ]
    for res, pkg in pkgs:
        try:
            nltk.data.find(res)
        except LookupError:
            nltk.download(pkg)

ensure_nltk()

# ---------------- Section parsing ----------------
SECTION_WEIGHTS = {
    "method": 1.0, "methodology": 1.0, "materials": 1.0, "data": 0.95,
    "experiment": 0.95, "implementation": 0.9,
    "results": 0.8, "evaluation": 0.8,
    "abstract": 0.6, "preamble": 0.5,
    "introduction": 0.25, "related": 0.25, "literature": 0.25, "background": 0.25, "review": 0.25,
    "discussion": 0.4, "conclusion": 0.4, "future": 0.3, "appendix": 0.2
}

HEADING_RX = re.compile(r'(?m)^(?:\s*\d+(?:\.\d+)*\s+)?([A-Z][A-Za-z][^\n]{0,100})$')

def normalize_section_key(raw_header: str) -> str:
    h = raw_header.strip().lower()
    if h.startswith('abstract') or 'abstract' in h: return 'abstract'
    if 'method' in h: return 'method'
    if 'material' in h: return 'materials'
    if 'data' in h: return 'data'
    if 'experiment' in h: return 'experiment'
    if 'implement' in h: return 'implementation'
    if 'result' in h: return 'results'
    if 'evaluat' in h or 'validation' in h: return 'evaluation'
    if 'related work' in h or 'related-work' in h or 'review' in h or 'literature' in h: return 'literature'
    if 'background' in h: return 'background'
    if 'intro' in h: return 'introduction'
    if 'discuss' in h: return 'discussion'
    if 'conclusion' in h: return 'conclusion'
    if 'future work' in h or 'future direction' in h: return 'future'
    if 'appendix' in h: return 'appendix'
    return h.split()[0] if h else 'unknown'

def split_sections(original_text: str):
    text = (original_text or "").replace('\r', '\n')
    out = {}
    matches = list(HEADING_RX.finditer(text))
    if not matches:
        return {'preamble': text}
    out['preamble'] = text[:matches[0].start()]
    for i, m in enumerate(matches):
        raw_header = m.group(1)
        key = normalize_section_key(raw_header)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out[key] = text[start:end]
    return out

# ---------------- Ontology expansion ----------------
def expand_terms(model: Word2Vec, terms, k=12, thresh=0.55):
    terms = [t for t in terms if isinstance(t, str) and t.strip()]
    out = set(terms)
    for t in terms:
        t_norm = t.replace(' ', '_')
        if t_norm in model.wv:
            for w, sim in model.wv.most_similar(t_norm, topn=k):
                if sim >= thresh:
                    out.add(w.replace('_', ' '))
    return sorted(out)

def expand_ontology(model: Word2Vec, ont: dict):
    expanded = {}
    for cat, subtree in ont.items():
        if isinstance(subtree, dict):
            expanded[cat] = {}
            for key, terms in subtree.items():
                expanded[cat][key] = expand_terms(model, terms)
        else:
            expanded[cat] = expand_terms(model, subtree)
    return expanded

# ---------------- Helpers ----------------
def contains_term(text, term):
    return re.search(r'\b' + re.escape(term) + r'\b', text, flags=re.I) is not None

def find_terms(text, terms, max_hits=5):
    hits = []
    for t in terms:
        m = re.search(r'.{0,60}\b' + re.escape(t) + r'\b.{0,60}', text, flags=re.I)
        if m:
            hits.append((t, m.group(0)))
            if len(hits) >= max_hits:
                break
    return hits

def to_str(x):
    if x is None: return ""
    if isinstance(x, (list, tuple, set)): return ";".join(map(str, x))
    if isinstance(x, dict): return ";".join(f"{k}:{v:.2f}" for k, v in x.items())
    return str(x)

# ---------------- Labelers ----------------
def label_models_multi(sections, model_paradigm_terms: dict):
    model_hits, evidence = {}, {}
    paradigm_scores = {k: 0.0 for k in model_paradigm_terms.keys()}

    for sec, text in sections.items():
        t = (text or "").lower()
        w = SECTION_WEIGHTS.get(sec, 0.3)
        for paradigm, terms in model_paradigm_terms.items():
            ft = find_terms(t, terms, max_hits=10)
            for term, snip in ft:
                paradigm_scores[paradigm] += w
                model_hits[term] = model_hits.get(term, 0.0) + w
                evidence.setdefault(term, []).append((sec, term, snip))

    total = sum(paradigm_scores.values()) or 1.0
    norm = {k: v / total for k, v in paradigm_scores.items()}
    paradigms = [k for k, v in norm.items() if v >= 0.28]
    if not paradigms and total > 0:
        paradigms = [max(paradigm_scores, key=paradigm_scores.get)]
    return paradigms, model_hits, evidence, paradigm_scores

def label_scale(sections, ont):
    hits = set(); ev = defaultdict(list)
    for sec, text in sections.items():
        t = (text or "").lower()
        for scale, terms in ont.get("scale", {}).items():
            ft = find_terms(t, terms, max_hits=3)
            if ft:
                hits.add(scale); ev[scale].extend([(sec,)*1 + h for h in ft])
    return sorted(hits), ev

def label_data_types(sections, ont):
    found = set(); ev = defaultdict(list)
    for sec, text in sections.items():
        t = (text or "").lower()
        for dtype, terms in ont.get("data_types", {}).items():
            ft = find_terms(t, terms, max_hits=3)
            if ft:
                found.add(dtype); ev[dtype].extend([(sec,)*1 + h for h in ft])
    return sorted(found), ev

SAMPLE_RX = re.compile(r'\b(\d+(?:\.\d+)?)\s*(hz|/s|/min|/hour|s|sec|second|min|minute|h|hr|hour)\b', re.I)
FREQ_WORDS = ["hourly", "daily", "weekly", "monthly", "15-min", "5-min", "1-min", "subhourly", "annual", "yearly"]
DUR_RX = re.compile(r'\b(?:for|over)\s+(\d+(?:\.\d+)?)\s*(day|days|week|weeks|month|months|year|years)\b', re.I)

def extract_sampling(sections):
    hits = []
    for sec, text in sections.items():
        for m in SAMPLE_RX.finditer(text or ""):
            hits.append({"section": sec, "value": m.group(1), "unit": m.group(2)})
    for sec, text in sections.items():
        t = (text or "").lower()
        for w in FREQ_WORDS:
            if contains_term(t, w):
                hits.append({"section": sec, "value": None, "unit": w})
    return hits

def extract_durations(sections):
    hits = []
    for sec, text in sections.items():
        for m in DUR_RX.finditer(text or ""):
            hits.append({"section": sec, "value": m.group(1), "unit": m.group(2)})
    return hits

def label_applications_and_kpi(sections, ont, kpi_priority_order):
    apps = set(); kpis = set()
    ev_apps = defaultdict(list); ev_kpi = defaultdict(list)
    for sec, text in sections.items():
        t = (text or "").lower()
        for app, terms in ont.get("applications", {}).items():
            ft = find_terms(t, terms, max_hits=3)
            if ft:
                apps.add(app); ev_apps[app].extend([(sec,)*1 + h for h in ft])
        for res, terms in ont.get("kpi_resolution", {}).items():
            ft = find_terms(t, terms, max_hits=3)
            if ft:
                kpis.add(res); ev_kpi[res].extend([(sec,)*1 + h for h in ft])
        # Also catch bare tokens like "hourly", etc.
        for kw in ["hourly", "daily", "monthly", "15-min", "5-min", "1-min", "subhourly", "annual", "yearly"]:
            if contains_term(t, kw):
                tag = kw.replace('-', '_')
                kpis.add(tag); ev_kpi[tag].append((sec, kw, kw))

    # Select ONE primary KPI by the order defined in ontology.yaml
    primary_kpi = pick_primary_kpi(ev_kpi, kpi_priority_order)
    return sorted(apps), sorted(kpis), primary_kpi, ev_apps, ev_kpi

def pick_primary_kpi(kpi_ev, kpi_priority_order):
    if not kpi_ev:
        return None
    # Score each detected KPI by section weights
    scored = {k: sum(SECTION_WEIGHTS.get(sec, 0.3) for (sec, _, _) in v)
              for k, v in kpi_ev.items()}
    # Follow ontology order strictly: first key present wins
    for key in kpi_priority_order:
        if key in scored:
            return key
    # If none of the canonical keys hit, fall back to highest score
    return max(scored, key=scored.get)

# Map sampling mentions to a coarse resolution bucket
def sampling_to_resolution(sampling_mentions):
    # Any Hz or seconds/minutes -> subhourly
    for h in sampling_mentions:
        unit = (h["unit"] or "").lower()
        if unit in ("hz", "/s", "s", "sec", "second", "min", "minute", "/min", "5-min", "15-min", "1-min", "30-s", "10-second", "subhourly"):
            return "subhourly"
    # Hourly tokens
    for h in sampling_mentions:
        unit = (h["unit"] or "").lower()
        if unit in ("h", "hr", "hour", "/hour", "hourly", "per hour", "1-hour"):
            return "hourly"
    # Daily/monthly/yearly hints (rare as sampling, but handle)
    for h in sampling_mentions:
        unit = (h["unit"] or "").lower()
        if "daily" in unit: return "daily"
        if "monthly" in unit: return "monthly"
        if "year" in unit or "annual" in unit: return "yearly"
    return None

# ---------------- Core paper analysis ----------------
def analyze_paper(doc, w2v_model, ont_expanded, kpi_priority_order):
    raw = doc['full-text-retrieval-response']['originalText']
    sections = split_sections(raw)

    paradigms, model_hits, model_ev, paradigm_scores = label_models_multi(
        sections, model_paradigm_terms=ont_expanded.get("model_paradigm", {})
    )
    scales, scale_ev = label_scale(sections, ont_expanded)
    data_types, data_ev = label_data_types(sections, ont_expanded)
    sampling = extract_sampling(sections)
    durations = extract_durations(sections)
    apps, kpis, primary_kpi, app_ev, kpi_ev = label_applications_and_kpi(
        sections, ont_expanded, kpi_priority_order
    )

    # Derive collected data resolution from sampling; if none, fall back to primary KPI
    collected_data_resolution = sampling_to_resolution(sampling) or primary_kpi

    return {
        "paradigms": paradigms,
        "paradigm_scores": paradigm_scores,
        "model_hits": model_hits,
        "model_evidence": model_ev,
        "scale": scales,
        "scale_evidence": scale_ev,
        "data_types": data_types,
        "data_evidence": data_ev,
        "sampling_mentions": sampling,
        "duration_mentions": durations,
        "applications": apps,
        "kpis": kpis,
        "kpi_primary": primary_kpi,
        "kpi_evidence": kpi_ev,
        "collected_data_resolution": collected_data_resolution,
    }

# ---------------- IO & parallel driver ----------------
def load_jsons(dirpath):
    for fn in os.listdir(dirpath):
        if fn.endswith(".json"):
            p = os.path.join(dirpath, fn)
            with open(p, "r", encoding="utf-8") as f:
                yield fn, json.load(f)

def worker(args):
    fname, doc, model_path, ont_expanded, kpi_priority_order = args
    model = Word2Vec.load(model_path)  # safest across processes
    res = analyze_paper(doc, model, ont_expanded, kpi_priority_order)
    return fname, res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Folder with Elsevier JSONs")
    ap.add_argument("--model", default="word2vec_model.model", help="Path to trained gensim Word2Vec model")
    ap.add_argument("--ontology", default="ontology.yaml", help="Seed ontology YAML/YML")
    ap.add_argument("--output_csv", default="classified_papers.csv", help="Output CSV path")
    ap.add_argument("--output_json", default="", help="Optional evidence JSON")
    ap.add_argument("--jobs", type=int, default=1, help="Parallel workers")
    args = ap.parse_args()

    # Load ontology
    with open(args.ontology, "r", encoding="utf-8") as f:
        ont = yaml.safe_load(f) or {}

    # Load model once (parent) to expand ontology
    w2v_parent = Word2Vec.load(args.model)
    ont_expanded = expand_ontology(w2v_parent, ont)

    # KPI priority order strictly follows the order defined in ontology.yaml
    # e.g., if you want subhourly to win first, list it first in YAML.
    kpi_priority_order = list(ont.get("kpi_resolution", {}).keys())

    items = list(load_jsons(args.input_dir))
    tasks = [(fname, doc, args.model, ont_expanded, kpi_priority_order) for (fname, doc) in items]

    rows = []
    if args.jobs and args.jobs > 1:
        with cf.ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futures = [ex.submit(worker, t) for t in tasks]
            for fut in tqdm(cf.as_completed(futures), total=len(futures), desc="Classifying"):
                try:
                    fname, res = fut.result()
                    rows.append((fname, res))
                except Exception as e:
                    print(f"[WARN] worker failed: {e}")
    else:
        for t in tqdm(tasks, total=len(tasks), desc="Classifying"):
            try:
                rows.append(worker(t))
            except Exception as e:
                print(f"[WARN] worker failed: {e}")

    out_rows = []
    for fname, r in rows:
        out_rows.append({
            "file": fname,
            "paradigms": to_str(r["paradigms"]),
            "model_hits": to_str(r["model_hits"]),
            "scale": to_str(r["scale"]),
            "data_types": to_str(r["data_types"]),
            "applications": to_str(r["applications"]),
            "kpis_all": to_str(r["kpis"]),
            "kpi_primary": r["kpi_primary"] or "",
            "collected_data_resolution": r["collected_data_resolution"] or "",
            "sampling_mentions": to_str([f'{h["section"]}:{h["value"] or ""}{h["unit"]}' for h in r["sampling_mentions"]]),
            "duration_mentions": to_str([f'{h["section"]}:{h["value"]}{h["unit"]}' for h in r["duration_mentions"]]),
        })
    df = pd.DataFrame(out_rows).sort_values("file")
    df.to_csv(args.output_csv, index=False)
    print(f"[OK] Wrote {args.output_csv} with {len(df)} rows.")

    if args.output_json:
        ev = {fname: r for fname, r in rows}
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(ev, f, ensure_ascii=False, indent=2)
        print(f"[OK] Wrote evidence JSON → {args.output_json}")

if __name__ == "__main__":
    main()
