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
    "method":1.0, "methodology": 1.0, "materials": 1.0, "data": 1.0,
    "experiment": 1.0, "implementation": 1.0,
    "results": 1.0, "discussion": 0.25, 
    "introduction": 0.01, "related": 0.01, "literature": 0.01, "background": 0.01, "review": 0.01,
    "conclusion": 0.05, "future": 0.05, "appendix": 0.25
}

# ---------------- Section filtering ----------------
EXCLUDE_SECTIONS = {
    "related", "literature", "review", "background", "introduction",
    "preamble"   # drop if you don't trust preambles from OCR
}

def scannable_sections(sections: dict):
    """Yield (sec, text) for sections that are allowed to contribute signal."""
    for sec, text in sections.items():
        if sec in EXCLUDE_SECTIONS:
            continue
        yield sec, text or ""

HEADING_RX = re.compile(
    r'(?m)^(?:\s*\d+(?:\.\d+)*\s+)?([A-Z][A-Za-z0-9\s\-\(\):]{2,120})\s*:?\s*$'
)

def normalize_section_key(raw_header: str) -> str:
    h = (raw_header or "").strip().lower()

    # canonical reviewish forms first
    if any(kw in h for kw in [
        "related work", "state of the art", "literature", "survey",
        "overview of", "prior work", "previous work", "background and related"
    ]):
        return "literature"

    if h.startswith('abstract') or 'abstract' in h: return 'abstract'
    if 'method' in h: return 'method'
    if 'material' in h: return 'materials'
    if 'data' in h: return 'data'
    if 'experiment' in h: return 'experiment'
    if 'implement' in h: return 'implementation'
    if 'result' in h: return 'results'
    if 'evaluat' in h or 'validation' in h: return 'evaluation'
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

# ---------------- New: Energy-waste concept + KPI trade-offs helpers ----------------
DEFN_PATTERNS = [
    r"\bdefine(?:s|d)?\s+{term}\s+as\b",
    r"\b{term}\s+is\s+defined\s+as\b",
    r"\bwe\s+conceptualiz(?:e|ed)\s+{term}\s+as\b",
    r"\bwe\s+operationaliz(?:e|ed)\s+{term}\s+as\b",
    r"\bthe\s+definition\s+of\s+{term}\b",
    r"\b{term}\s+refers\s+to\b",
]

TRADEOFF_WORDS = [
    "trade-off", "tradeoff", "balance", "balancing", "compromise",
    "pareto", "multiobjective", "multi-objective", "conflict",
    "tension", "competing", "jointly optimize", "simultaneously optimize"
]

# ---- Sentence splitting helper----
def sent_tokenize_safe(text: str):
    try:
        from nltk.tokenize import sent_tokenize
        return sent_tokenize(text or "")
    except Exception:
        return re.split(r'(?<=[\.\?\!])\s+', text or "")

# ---------------- Labelers ----------------
def label_special_focus(sections, ont):
    return label_generic_categories(sections, ont, "special_focus")

def label_energy_waste_concept(sections, ont):
    """
    Uses ont['concept_energy_waste'] dict (already Word2Vec expanded).
    Returns mentions, definition flag, evidence.
    """
    subtree = ont.get("concept_energy_waste", {}) or {}
    mentions = set(); ev_mentions = defaultdict(list); ev_defn = []

    for sec, text in scannable_sections(sections):
        t = text or ""
        tl = t.lower()
        for cat, terms in subtree.items():
            for term in terms:
                ft = find_terms(tl, [term], max_hits=6)
                for _, snip in ft:
                    mentions.add(term)
                    ev_mentions[term].append((sec, term, snip))

                # Definitional cues
                for pat in DEFN_PATTERNS:
                    rx = re.compile(pat.format(term=re.escape(term)), flags=re.I)
                    m = rx.search(tl)
                    if m:
                        s = max(0, m.start() - 80)
                        e = min(len(t), m.end() + 160)
                        ev_defn.append((sec, term, t[s:e]))

    return sorted(mentions), (len(ev_defn) > 0), ev_mentions, ev_defn
def label_kpi_tradeoffs_multi(sections, ont, min_kpis=2):
    """
    Detect trade-offs/balances with >= min_kpis KPI concepts in the same sentence.
    Uses:
      - ont['kpi_tradeoff_concepts']: {concept_key: [terms...]}
      - ont['tradeoff_keywords'] (optional)
    Returns:
      groups:        sorted list of group keys like 'energy_use|thermal_comfort|infection_risk'
      pairs:         sorted list of pair keys like 'energy_use|thermal_comfort'
      keywords:      sorted list of tradeoff/balance cue words actually seen
      ev_groups:     dict[group_key] -> [(section, keyword, sentence), ...]
      ev_pairs:      dict[pair_key]  -> [(section, keyword, sentence), ...]
      group_scores:  dict[group_key] -> section-weighted score (sum of section weights over evidence)
      pair_scores:   dict[pair_key]  -> section-weighted score
      kpis_any:      sorted list of unique KPI concept keys seen in any tradeoff sentence
    """
    concept_map = ont.get("kpi_tradeoff_concepts", {}) or {}
    extra_kw = ont.get("tradeoff_keywords", []) or []
    trade_words = set(TRADEOFF_WORDS + extra_kw)

    # precompile KPI term regex per concept
    kpi_rx = {
        key: [re.compile(r"\b" + re.escape(str(term).lower()) + r"\b", re.I)
              for term in (terms or [])]
        for key, terms in concept_map.items()
    }

    groups, pairs = set(), set()
    kw_found = set()
    ev_groups = defaultdict(list)
    ev_pairs  = defaultdict(list)
    group_scores = defaultdict(float)
    pair_scores  = defaultdict(float)
    kpis_any = set()

    for sec, text in scannable_sections(sections):
        w = SECTION_WEIGHTS.get(sec, 0.3)
        for sent in sent_tokenize_safe(text or ""):
            sl = sent.lower()

            # require one trade-off cue word
            hit_kw = next((kw for kw in trade_words if kw.lower() in sl), None)
            if not hit_kw:
                continue
            kw_found.add(hit_kw)

            # which KPI concepts are present?
            present = []
            for key, regs in kpi_rx.items():
                if any(rx.search(sl) for rx in regs):
                    present.append(key)

            present = sorted(set(present))
            if len(present) < min_kpis:
                continue

            # record the whole group
            group_key = "|".join(present)
            groups.add(group_key)
            ev_groups[group_key].append((sec, hit_kw, sent.strip()))
            group_scores[group_key] += w
            kpis_any.update(present)

            # also emit all unordered pairs for compatibility
            for i in range(len(present)):
                for j in range(i + 1, len(present)):
                    pair_key = f"{present[i]}|{present[j]}"
                    pairs.add(pair_key)
                    ev_pairs[pair_key].append((sec, hit_kw, sent.strip()))
                    pair_scores[pair_key] += w

    return (
        sorted(groups),
        sorted(pairs),
        sorted(kw_found),
        ev_groups,
        ev_pairs,
        dict(group_scores),
        dict(pair_scores),
        sorted(kpis_any),
    )

def label_paradigms_by_scale(sections, ont):
    """
    For each scale key in ont['scale'], detect which model_paradigm terms
    co-occur in the same sentence and accumulate section-weighted scores.

    Returns:
      out: dict keyed by scale label:
        {
          scale_key: {
            "present": bool,
            "paradigms": [winners],                 # e.g., ["whitebox", "blackbox"]
            "paradigm_scores": {paradigm: score},   # raw weighted scores
            "paradigm_term_hits": {term: score},    # term-level hits (EnergyPlus, LSTM, RL, etc.)
            "evidence": {key: [(sec, key, sentence), ...]}  # keys include paradigms and some high-signal terms
          },
          ...
        }
    """
    scale_map = ont.get("scale", {}) or {}
    paradigms_map = ont.get("model_paradigm", {}) or {}

    # compile regexes
    def rx_list(terms):
        return [re.compile(r"\b" + re.escape(str(t).lower()) + r"\b", re.I) for t in (terms or [])]

    scale_rx = {s: rx_list(terms) for s, terms in scale_map.items()}
    # for each paradigm, we’ll have regexes for *all* its (expanded) terms
    paradigm_term_rx = {p: [(t, re.compile(r"\b" + re.escape(str(t).lower()) + r"\b", re.I)) for t in (terms or [])]
                        for p, terms in paradigms_map.items()}

    out = {}
    for s in scale_map.keys():
        out[s] = {
            "present": False,
            "paradigms": [],
            "paradigm_scores": {p: 0.0 for p in paradigms_map.keys()},
            "paradigm_term_hits": {},   # term -> score
            "evidence": defaultdict(list),
        }

    for sec, text in scannable_sections(sections):
        w = SECTION_WEIGHTS.get(sec, 0.3)
        for sent in sent_tokenize_safe(text or ""):
            sl = sent.lower()
            scales_here = [s for s, rxs in scale_rx.items() if any(rx.search(sl) for rx in rxs)]
            if not scales_here:
                continue

            # which paradigm terms appear in this sentence?
            # record both the paradigm-level hit and the exact term(s)
            for p, term_rxs in paradigm_term_rx.items():
                # gather the matched terms for this paradigm in this sentence
                matched_terms = [t for (t, rx) in term_rxs if rx.search(sl)]
                if not matched_terms:
                    continue

                for s in scales_here:
                    out[s]["present"] = True
                    out[s]["paradigm_scores"][p] += w
                    out[s]["evidence"][p].append((sec, p, sent.strip()))
                    for t in matched_terms:
                        out[s]["paradigm_term_hits"][t] = out[s]["paradigm_term_hits"].get(t, 0.0) + w
                        # keep a little evidence keyed by the *term* too (handy for JSON)
                        out[s]["evidence"][t].append((sec, t, sent.strip()))

    # post-process: winners per scale (same 0.28 share rule you use globally)
    for s, rec in out.items():
        ps = rec["paradigm_scores"]
        total = sum(ps.values()) or 1.0
        norm = {k: v / total for k, v in ps.items()}
        winners = [k for k, v in norm.items() if v >= 0.28]
        if not winners and total > 0 and max(ps.values()) > 0:
            winners = [max(ps, key=ps.get)]
        rec["paradigms"] = winners

    return out

def label_optimization_by_scale(sections, ont):
    """
    Detect optimization methods (and optionally objectives) per scale by sentence-level co-occurrence.
    Returns:
      opt_per_scale: {
        scale_key: {
          "present": bool,
          "methods_scores": {method_key: score},   # section-weighted
          "methods": [sorted non-empty method keys],
          "objectives_scores": {obj_key: score},   # optional
          "objectives": [sorted non-empty obj keys],
          "evidence": {key: [(sec, key, sentence), ...]}
        }, ...
      }
    """
    scale_map = ont.get("scale", {}) or {}
    meth_map  = ont.get("optimization_methods", {}) or {}
    obj_map   = ont.get("optimization_objectives", {}) or {}

    def rx_list(terms):
        return [re.compile(r"\b" + re.escape(str(t).lower()) + r"\b", re.I) for t in (terms or [])]

    scale_rx = {s: rx_list(terms) for s, terms in scale_map.items()}
    meth_rx  = {k: rx_list(terms) for k, terms in meth_map.items()}
    obj_rx   = {k: rx_list(terms) for k, terms in obj_map.items()}

    out = {}
    for s in scale_map.keys():
        out[s] = {
            "present": False,
            "methods_scores": defaultdict(float),
            "methods": [],
            "objectives_scores": defaultdict(float),
            "objectives": [],
            "evidence": defaultdict(list),
        }

    for sec, text in scannable_sections(sections):
        w = SECTION_WEIGHTS.get(sec, 0.3)
        for sent in sent_tokenize_safe(text or ""):
            sl = sent.lower()

            scales_here = [s for s, rxs in scale_rx.items() if any(rx.search(sl) for rx in rxs)]
            if not scales_here:
                continue

            meth_here = [m for m, rxs in meth_rx.items() if any(rx.search(sl) for rx in rxs)]
            obj_here  = [o for o, rxs in obj_rx.items()  if any(rx.search(sl) for rx in rxs)]

            if not meth_here and not obj_here:
                continue

            for s in scales_here:
                out[s]["present"] = True
                for m in meth_here:
                    out[s]["methods_scores"][m] += w
                    out[s]["evidence"][m].append((sec, m, sent.strip()))
                for o in obj_here:
                    out[s]["objectives_scores"][o] += w
                    out[s]["evidence"][o].append((sec, o, sent.strip()))

    # finalize winners per scale (keep all nonzero, sorted by score desc)
    for s, rec in out.items():
        if rec["methods_scores"]:
            rec["methods"] = [k for k,_ in sorted(rec["methods_scores"].items(), key=lambda kv: kv[1], reverse=True)]
        if rec["objectives_scores"]:
            rec["objectives"] = [k for k,_ in sorted(rec["objectives_scores"].items(), key=lambda kv: kv[1], reverse=True)]

    return out

def label_online_learning(sections, ont):
    scale_map = ont.get("scale", {}) or {}
    online_terms = (ont.get("online_learning", {}) or {}).get("online_learning", [])  # dict form for expansion
    if not online_terms:
        # also support list form
        if isinstance(ont.get("online_learning", None), list):
            online_terms = ont["online_learning"]

    def rx_list(terms):
        return [re.compile(r"\b" + re.escape(str(t).lower()) + r"\b", re.I) for t in (terms or [])]

    scale_rx = {s: rx_list(terms) for s, terms in scale_map.items()}
    online_rx = rx_list(online_terms)

    global_flag = False
    global_ev = []
    per_scale_flags = {s: False for s in scale_map.keys()}
    per_scale_ev = {s: [] for s in scale_map.keys()}

    for sec, text in scannable_sections(sections):
        for sent in sent_tokenize_safe(text or ""):
            sl = sent.lower()
            has_online = any(rx.search(sl) for rx in online_rx)
            if not has_online:
                continue
            global_flag = True
            global_ev.append((sec, "online_learning", sent.strip()))
            for s, rxs in scale_rx.items():
                if any(rx.search(sl) for rx in rxs):
                    per_scale_flags[s] = True
                    per_scale_ev[s].append((sec, "online_learning", sent.strip()))

    return global_flag, global_ev, per_scale_flags, per_scale_ev

def label_models_multi(sections, model_paradigm_terms: dict):
    model_hits, evidence = {}, {}
    paradigm_scores = {k: 0.0 for k in model_paradigm_terms.keys()}

    for sec, text in scannable_sections(sections):
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
    for sec, text in scannable_sections(sections):
        t = (text or "").lower()
        for scale, terms in ont.get("scale", {}).items():
            ft = find_terms(t, terms, max_hits=3)
            if ft:
                hits.add(scale); ev[scale].extend([(sec,)*1 + h for h in ft])
    return sorted(hits), ev

def label_data_types(sections, ont):
    found = set(); ev = defaultdict(list)
    for sec, text in scannable_sections(sections):
        t = (text or "").lower()
        for dtype, terms in ont.get("data_types", {}).items():
            ft = find_terms(t, terms, max_hits=3)
            if ft:
                found.add(dtype); ev[dtype].extend([(sec,)*1 + h for h in ft])
    return sorted(found), ev

SAMPLE_RX = re.compile(r'\b(\d+(?:\.\d+)?)\s*(hz|/s|/min|/hour|s|sec|second|min|minute|h|hr|hour)\b', re.I)
FREQ_WORDS = ["hourly", "daily", "weekly", "monthly", "15-min", "5-min", "1-min", "subhourly", "annual", "yearly"]

def extract_sampling(sections):
    hits = []
    for sec, text in scannable_sections(sections):
        for m in SAMPLE_RX.finditer(text or ""):
            hits.append({"section": sec, "value": m.group(1), "unit": m.group(2)})
    for sec, text in scannable_sections(sections):
        t = (text or "").lower()
        for w in FREQ_WORDS:
            if contains_term(t, w):
                hits.append({"section": sec, "value": None, "unit": w})
    return hits

def label_applications_and_kpi(sections, ont, kpi_priority_order):
    apps = set(); kpis = set()
    ev_apps = defaultdict(list); ev_kpi = defaultdict(list)
    for sec, text in scannable_sections(sections):
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

# ---------------- New labelers for KPI type and model inputs ----------------

def label_generic_categories(sections, ont, ont_section_key):
    """
    Generic multi-label detector: returns (sorted_labels, evidence_dict).
    evidence_dict[label] -> list of (section, term, snippet)
    """
    found = set(); ev = defaultdict(list)
    subtree = ont.get(ont_section_key, {})
    if not isinstance(subtree, dict):
        return [], {}
    for sec, text in scannable_sections(sections):
        t = (text or "").lower()
        for label, terms in subtree.items():
            ft = find_terms(t, terms, max_hits=3)
            if ft:
                found.add(label)
                ev[label].extend([(sec,)*1 + h for h in ft])
    return sorted(found), ev

def label_kpi_types(sections, ont):
    return label_generic_categories(sections, ont, "kpi_type")


def label_model_development(sections, ont):
    # ont["model_development"] can be either a list or dict of lists
    found = set(); ev = defaultdict(list)
    subtree = ont.get("model_development", {})
    mode_map = {}
    if isinstance(subtree, dict):
        mode_map = subtree
    elif isinstance(subtree, list):
        # Treat as flat list under a single bucket named "mode"
        mode_map = {m: [m] for m in subtree}
    for sec, text in scannable_sections(sections):
        t = (text or "").lower()
        for mode, terms in mode_map.items():
            ft = find_terms(t, terms, max_hits=3)
            if ft:
                found.add(mode); ev[mode].extend([(sec,)*1 + h for h in ft])
    return sorted(found), ev

def label_model_inputs_and_resolution(sections, ont, input_kpi_priority_order):
    """
    Detects model input *types* (e.g., occupancy, load, weather) and their *temporal resolution*,
    analogous to KPI resolution. Also captures generic frequency tokens (hourly, daily, etc.).
    """
    inputs = set(); input_ev = defaultdict(list)
    input_resolutions = set(); input_res_ev = defaultdict(list)

    # Inputs
    for sec, text in scannable_sections(sections):
        t = (text or "").lower()
        for inp, terms in ont.get("model_inputs", {}).items():
            ft = find_terms(t, terms, max_hits=3)
            if ft:
                inputs.add(inp); input_ev[inp].extend([(sec,)*1 + h for h in ft])

    # Resolutions (explicit dictionary like kpi_resolution)
    for sec, text in scannable_sections(sections):
        t = (text or "").lower()
        for res, terms in ont.get("input_resolution", {}).items():
            ft = find_terms(t, terms, max_hits=3)
            if ft:
                input_resolutions.add(res)
                input_res_ev[res].extend([(sec,)*1 + h for h in ft])

        # Bare tokens too
        for kw in ["hourly", "daily", "monthly", "15-min", "5-min", "1-min", "subhourly", "annual", "yearly"]:
            if contains_term(t, kw):
                tag = kw.replace('-', '_')
                input_resolutions.add(tag); input_res_ev[tag].append((sec, kw, kw))

    primary_input_resolution = pick_primary_kpi(input_res_ev, input_kpi_priority_order) if input_res_ev else None
    return sorted(inputs), sorted(input_resolutions), primary_input_resolution, input_ev, input_res_ev

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
def analyze_paper(doc, ont_expanded, kpi_priority_order, input_kpi_priority_order):
    raw = doc['full-text-retrieval-response']['originalText']
    sections = split_sections(raw)

    paradigms, model_hits, model_ev, paradigm_scores = label_models_multi(
        sections, model_paradigm_terms=ont_expanded.get("model_paradigm", {})
    )
    scales, scale_ev = label_scale(sections, ont_expanded)
        
    data_types, data_ev = label_data_types(sections, ont_expanded)
    sampling = extract_sampling(sections)
    apps, kpis, primary_kpi, app_ev, kpi_ev = label_applications_and_kpi(
        sections, ont_expanded, kpi_priority_order
    )

    # Energy-waste + tradeoffs
    ew_mentions, ew_defn_flag, ew_ev_mentions, ew_ev_defn = label_energy_waste_concept(sections, ont_expanded)
    (trade_groups, trade_pairs, trade_keywords, trade_ev_groups, trade_ev_pairs,
    trade_group_scores, trade_pair_scores, trade_kpis_any) = label_kpi_tradeoffs_multi(sections, ont_expanded)

    # Per-scale paradigms (white/grey/black + specific terms)
    per_scale = label_paradigms_by_scale(sections, ont_expanded)
    opt_per_scale = label_optimization_by_scale(sections, ont_expanded)
    # KPI types, model development, inputs + input resolution
    kpi_types, kpi_type_ev = label_kpi_types(sections, ont_expanded)
    model_development_modes, model_development_ev = label_model_development(sections, ont_expanded)
    model_inputs, input_resolutions, primary_input_resolution, model_inputs_ev, input_res_ev = \
        label_model_inputs_and_resolution(sections, ont_expanded, input_kpi_priority_order)

    # Online learning (global + per-scale)
    online_flag, online_ev, online_per_scale, online_ev_per_scale = label_online_learning(sections, ont_expanded)

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
        "applications": apps,
        "applications_evidence": app_ev,
        "kpis": kpis,
        "kpi_primary": primary_kpi,
        "kpi_evidence": kpi_ev,
        "collected_data_resolution": collected_data_resolution,
        "kpi_types": kpi_types,
        "kpi_type_evidence": kpi_type_ev,
        "model_development": model_development_modes,
        "model_development_evidence": model_development_ev,
        "model_inputs": model_inputs,
        "input_resolutions": input_resolutions,
        "input_resolution_primary": primary_input_resolution,
        "model_inputs_evidence": model_inputs_ev,
        "input_resolution_evidence": input_res_ev,



        "energy_waste_mentions": ew_mentions,
        "energy_waste_definition_found": ew_defn_flag,
        "energy_waste_evidence_mentions": ew_ev_mentions,      # dict: term -> [(sec, term, snippet)]
        "energy_waste_evidence_definitions": ew_ev_defn,       # list: [(sec, term, snippet)]

        # Trade-offs (multi-KPI)
        "kpi_tradeoff_groups": trade_groups,
        "kpi_tradeoff_pairs": trade_pairs,
        "kpi_tradeoff_keywords": trade_keywords,
        "kpi_tradeoff_groups_evidence": trade_ev_groups,
        "kpi_tradeoff_pairs_evidence": trade_ev_pairs,
        "kpi_tradeoff_group_scores": trade_group_scores,
        "kpi_tradeoff_pair_scores": trade_pair_scores,
        "kpi_tradeoff_kpis_any": trade_kpis_any,
        
        # NEW → include per-scale + online-learning outputs
        "per_scale": per_scale,
        "optimization_per_scale": opt_per_scale,
        "online_learning": online_flag,
        "online_learning_evidence": online_ev,
        "online_learning_per_scale": online_per_scale,
        "online_learning_evidence_per_scale": online_ev_per_scale,
    }

# ---------------- IO & parallel driver ----------------
def load_jsons(dirpath):
    for fn in os.listdir(dirpath):
        if fn.endswith(".json"):
            p = os.path.join(dirpath, fn)
            with open(p, "r", encoding="utf-8") as f:
                yield fn, json.load(f)

def worker(args):
    fname, doc, ont_expanded, kpi_priority_order, input_kpi_priority_order = args
    # No need to load the model here anymore
    res = analyze_paper(doc, ont_expanded, kpi_priority_order, input_kpi_priority_order)
    return fname, res
def allow_ev(sec: str) -> bool:
    return sec not in EXCLUDE_SECTIONS

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Folder with Elsevier JSONs")
    ap.add_argument("--model", default="word2vec_model.model", help="Path to trained gensim Word2Vec model")
    ap.add_argument("--ontology", default="ontology.yaml", help="Seed ontology YAML/YML")
    ap.add_argument("--output_csv", default="classified_papers.csv", help="Output CSV path")
    ap.add_argument("--output_json", default="", help="Optional evidence JSON")
    ap.add_argument("--jobs", type=int, default=8, help="Parallel workers")
    args = ap.parse_args()

    # Load ontology
    with open(args.ontology, "r", encoding="utf-8") as f:
        ont = yaml.safe_load(f) or {}

    # Load model once (parent) to expand ontology
    w2v_parent = Word2Vec.load(args.model)
    ont_expanded = expand_ontology(w2v_parent, ont)
    kpi_priority_order = list(ont.get("kpi_resolution", {}).keys())
    # KPI priority order strictly follows the order defined in ontology.yaml
    
    input_kpi_priority_order = list(ont.get("input_resolution", {}).keys())

    items = list(load_jsons(args.input_dir))
    
    tasks = [(fname, doc, ont_expanded, kpi_priority_order, input_kpi_priority_order)
         for (fname, doc) in items]

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

    def nm_if_empty(x):
        if not x: return "NM"
        if isinstance(x, (list, set, tuple)):
            s = ";".join(map(str, x)).strip()
            return s or "NM"
        if isinstance(x, dict):
            if not x: return "NM"
            keys_sorted = sorted(x.items(), key=lambda kv: kv[1], reverse=True)
            return ";".join(k for k, _ in keys_sorted) or "NM"
        return str(x)

    def ps_vals(per_scale_dict, scale_key):
        rec = (per_scale_dict or {}).get(scale_key, {})
        paradigms = rec.get("paradigms") or []
        term_hits = rec.get("paradigm_term_hits") or {}
        return nm_if_empty(paradigms), nm_if_empty(term_hits)
    
    def join_or_nm(xs):
        if not xs: return "NM"
        if isinstance(xs, dict):
            if not xs: return "NM"
            return ";".join(k for k,_ in sorted(xs.items(), key=lambda kv: kv[1], reverse=True)) or "NM"
        if isinstance(xs, (list, set, tuple)):
            s = ";".join(map(str, xs)).strip()
            return s or "NM"
        return str(xs)

    def opt_vals(opt_per_scale, scale_key):
        rec = (opt_per_scale or {}).get(scale_key, {})
        return (
            join_or_nm(rec.get("methods")),
            join_or_nm(rec.get("objectives")),
            join_or_nm(rec.get("terms"))
        )

    for fname, r in rows:
        ps = r.get("per_scale", {})

        bm_p, bm_terms = ps_vals(ps, "building_model")
        sm_p, sm_terms = ps_vals(ps, "system_model")
        om_p, om_terms = ps_vals(ps, "occupancy_model")
        cm_p, cm_terms = ps_vals(ps, "climate_model")
        opt_ps = r.get("optimization_per_scale", {})

        bm_meth, bm_obj, bm_terms_opt = opt_vals(opt_ps, "building_model")
        sm_meth, sm_obj, sm_terms_opt = opt_vals(opt_ps, "system_model")
        om_meth, om_obj, om_terms_opt = opt_vals(opt_ps, "occupancy_model")
        cm_meth, cm_obj, cm_terms_opt = opt_vals(opt_ps, "climate_model")
        # Energy waste: flatten sentences for CSV

        ew_sentences_flat = to_str([
            f"{sec}: {snip}"
            for term, hits in (r.get("energy_waste_evidence_mentions") or {}).items()
            for sec, _, snip in hits if allow_ev(sec)
        ])
        ew_defns_flat = to_str([
            f"{sec}: {snip}"
            for sec, term, snip in (r.get("energy_waste_evidence_definitions") or [])
        ])

        # (Optional) tradeoff top group by score
        trade_group_scores = r.get("kpi_tradeoff_group_scores") or {}
        trade_top_group = max(trade_group_scores.items(), key=lambda kv: kv[1])[0] if trade_group_scores else "NM"

        out_rows.append({
            "file": fname,
            "scale": to_str(r["scale"]),
            "paradigms": to_str(r["paradigms"]),
            "model_hits": to_str(r["model_hits"]),
            # Per-scale (paradigms + matched terms)
            "building_model_paradigms": bm_p,
            "building_model_paradigm_terms": bm_terms,
            "system_model_paradigms": sm_p,
            "system_model_paradigm_terms": sm_terms,
            "occupancy_model_paradigms": om_p,
            "occupancy_model_paradigm_terms": om_terms,
            "climate_model_paradigms": cm_p,
            "climate_model_paradigm_terms": cm_terms,
            # per-scale optimisation (methods + optional objectives)
            "building_model_optim_methods": bm_meth,
            "building_model_optim_objectives": bm_obj,
            "building_model_optim_terms": bm_terms_opt,

            "system_model_optim_methods": sm_meth,
            "system_model_optim_objectives": sm_obj,
            "system_model_optim_terms": sm_terms_opt,

            "occupancy_model_optim_methods": om_meth,
            "occupancy_model_optim_objectives": om_obj,
            "occupancy_model_optim_terms": om_terms_opt,

            "climate_model_optim_methods": cm_meth,
            "climate_model_optim_objectives": cm_obj,
            "climate_model_optim_terms": cm_terms_opt,
            # applications + KPIs
            "applications": to_str(r["applications"]),
            "kpis_all": to_str(r["kpis"]),
            "kpi_primary": r["kpi_primary"] or "",
            "kpi_types": to_str(r.get("kpi_types")),

            # Concepts/tradeoffs
            "energy_waste_mentions": to_str(r.get("energy_waste_mentions")),
            "energy_waste_definition_found": str(bool(r.get("energy_waste_definition_found"))),
            "energy_waste_sentences": ew_sentences_flat,
            "energy_waste_definitions": ew_defns_flat,
            "kpi_tradeoff_groups": to_str(r.get("kpi_tradeoff_groups")),
            "kpi_tradeoff_top_group": trade_top_group,
            "kpi_tradeoff_kpis_any": to_str(r.get("kpi_tradeoff_kpis_any")),
            # data + sampling
            "data_types": to_str(r["data_types"]),
            "collected_data_resolution": r["collected_data_resolution"] or "",
            "sampling_mentions": to_str([f'{h["section"]}:{h["value"] or ""}{h["unit"]}' for h in r["sampling_mentions"]]),
            # model development + inputs
            "model_development": to_str(r.get("model_development")),
            "model_inputs": to_str(r.get("model_inputs")),
            "input_resolutions_all": to_str(r.get("input_resolutions")),
            "input_resolution_primary": r.get("input_resolution_primary") or "",
            # Online learning
            "online_learning_any": "yes" if r.get("online_learning") else "NM",
            "online_learning_components": nm_if_empty([k for k, v in (r.get("online_learning_per_scale") or {}).items() if v]),

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

