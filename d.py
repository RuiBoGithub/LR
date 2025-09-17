#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Focused paper analyzer (Word2Vec-constrained expansion)

What it reports per paper:
  1) Applications
  2) Models used (by scale: building/system/occupancy/climate): paradigms + exact term hits
     + a flat list of CANONICAL model/tool/algorithm names seen (ENERGYPLUS, LINEAR REGRESSION, ...)
  3) System types (AHU/VAV/HP/...)
  4) Optimization methods + scopes (and a best method->scope pairing)
  5) Data collected + input resolution (prefers sampling mentions if present)
  6) KPI used + KPI resolutions (primary per KPI), including full phrases

Word2Vec expansion is **constrained** to avoid abstract single-word junk:
  - multi-seed agreement (min_seed_support>=2)
  - centroid consistency
  - noun/NP filters
  - blocklist (default + YAML-provided)
  - min_tokens=2 by default (no single tokens)

YAML can configure expansion behavior with an optional block:
expansion_policy:
  default:
    min_tokens: 2
    min_seed_support: 2
    thresh_seed: 0.55
    thresh_centroid: 0.60
    mutual: false
    allow_single_if_suffixes: ["model","estimation","prediction","forecast"]
    head_hints: ["model","modelling","modeling","estimation","prediction","forecast"]
  overrides:
    model_paradigm:
      expand: false  # keep exact paradigm terms
    systems:
      min_tokens: 1  # systems may have short tokens like "ahu","vav"
    applications:
      min_seed_support: 1

You can also add a top-level list:
blocklist: ["matlab","simulink","framework","pipeline","toolbox","integrated","co-simulation","cosimulation"]

Run:
  pip install pyyaml pandas nltk tqdm gensim
  python focused_paper_extractor_w2v.py --input_dir ... --ontology ... --model path/to/w2v.model
"""

import os, re, json, yaml, argparse, math
from collections import defaultdict
from typing import Dict, List, Tuple, Any

import pandas as pd
from tqdm import tqdm

# --- NLTK ---
import nltk
def ensure_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
ensure_nltk()
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag

# --- Gensim ---
from gensim.models import Word2Vec

# ---------------- Section parsing ----------------

SECTION_WEIGHTS = {
    "method":1.0, "methodology": 1.0, "materials": 1.0, "data": 1.0,
    "experiment": 1.0, "implementation": 1.0,
    "results": 1.0, "evaluation": 0.75,
    "abstract": 0.25, "preamble": 0.0,
    "introduction": 0.0, "related": 0.0, "literature": 0.0, "background": 0.0, "review": 0.0,
    "discussion": 0.25, "conclusion": 0.75, "future": 0.05, "appendix": 0.25
}

EXCLUDE_SECTIONS = {"related", "literature", "review", "background", "introduction"}

HEADING_RX = re.compile(r'(?m)^(?:\s*\d+(?:\.\d+)*\s+)?([A-Z][A-Za-z0-9\s\-\(\):]{2,120})\s*:?\s*$')

def normalize_section_key(raw_header: str) -> str:
    h = (raw_header or "").strip().lower()
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

def scannable_sections(sections: Dict[str, str]):
    for sec, text in sections.items():
        if sec in EXCLUDE_SECTIONS:
            continue
        yield sec, (text or "")

# -------------- Regex helpers --------------
def any_rx(terms: List[str]):
    rxs = []
    for t in (terms or []):
        t = str(t).strip()
        if not t:
            continue
        t_ = re.sub(r'\s+', r'[ -]', re.escape(t))
        rxs.append(re.compile(rf'\b{t_}\b', re.I))
    return rxs

def any_hit(rxs, s: str) -> bool:
    return any(rx.search(s) for rx in rxs)

def find_hits(rxs, s: str) -> List[str]:
    return [m.group(0) for rx in rxs for m in rx.finditer(s)]

def sent_tokenize_safe(text: str) -> List[str]:
    try:
        return sent_tokenize(text or "")
    except Exception:
        return re.split(r'(?<=[\.\?\!])\s+', text or "")

def snippet_around(hit: str, text: str, width: int = 80) -> str:
    m = re.search(re.escape(hit), text, re.I)
    if not m:
        return hit
    s = max(0, m.start() - width)
    e = min(len(text), m.end() + width)
    return text[s:e].strip()

# -------------- Context gating / disambiguation --------------

DEFAULT_ANCHORS = {
    "occupancy_model": {
        "require_any": ["model", "modelling", "modeling", "estimat", "infer", "predict", "classifier", "regression", "svm", "neural", "lstm", "grey-box", "gray-box"],
        "forbid_any": ["measured", "measurement", "sensor", "ground truth", "survey", "manual count"]
    },
    "temperature_indoor": {
        "require_any": ["indoor", "zone", "room", "space", "setpoint", "supply air", "return air", "operative", "iaq"],
        "forbid_any": ["outdoor", "ambient", "weather", "meteorolog", "external"]
    },
    "temperature_outdoor": {
        "require_any": ["outdoor", "ambient", "weather", "meteorolog", "external", "weather station"],
        "forbid_any": []
    }
}

def anchors_from_yaml(ont: Dict[str, Any]) -> Dict[str, Dict[str, List[str]]]:
    return (ont.get("anchors") or {}) if isinstance(ont, dict) else {}

def context_gate(sentence: str, require_any: List[str], forbid_any: List[str]) -> bool:
    s = sentence.lower()
    if forbid_any and any(re.search(rf'\b{re.escape(w.lower())}\b', s) for w in forbid_any):
        return False
    if require_any and not any(re.search(rf'\b{re.escape(w.lower())}\b', s) for w in require_any):
        return False
    return True

# -------------- Canonical naming for models --------------
CANONICAL_MODEL_NAMES = {
    "energyplus": "ENERGYPLUS",
    "trnsys": "TRNSYS",
    "ies-ve": "IES-VE",
    "modelica": "MODELICA",
    "ida-ice": "IDA-ICE",
    "state-space": "STATE-SPACE",
    "rc model": "RC MODEL",
    "rc network": "RC NETWORK",
    "resistor-capacitor": "RC MODEL",
    "kalman": "KALMAN FILTER",
    "linear regression": "LINEAR REGRESSION",
    "linear model": "LINEAR REGRESSION",
    "lasso": "LASSO",
    "ridge": "RIDGE",
    "elastic net": "ELASTIC NET",
    "neural network": "NEURAL NETWORK",
    "ann": "NEURAL NETWORK",
    "lstm": "LSTM",
    "gru": "GRU",
    "svm": "SVM",
    "svr": "SVR",
    "random forest": "RANDOM FOREST",
    "xgboost": "XGBOOST",
    "bayesian network": "BAYESIAN NETWORK",
    "hmm": "HMM",
    "k-nn": "KNN",
    "knn": "KNN",
    "naive bayes": "NAIVE BAYES",
    "gaussian process": "GAUSSIAN PROCESS",
    "degree-day": "DEGREE-DAY",
    "bin method": "BIN METHOD",
}

def canonicalize_term(term: str) -> str:
    t = (term or "").strip().lower()
    return CANONICAL_MODEL_NAMES.get(t, term.upper())

# -------------- Refined Word2Vec expansion --------------

ADMISSIBLE_POS = {"NN","NNS","NNP","NNPS"}

DEFAULT_BLOCKLIST = {
    "matlab","simulink","framework","pipeline","toolbox","integrated",
    "co-simulation","cosimulation","generic","baseline","improved","proposed"
}

def head_token(s: str):
    s = s.replace("_"," ").strip()
    if not s:
        return ""
    return s.split()[-1].lower()

def looks_like_term(s: str, *, min_tokens: int, allow_single_if_suffixes: List[str], head_hints: List[str], blocklist: set) -> bool:
    s = s.strip().lower()
    if not s or s in blocklist:
        return False
    tokens = s.replace("_"," ").split()
    if len(tokens) < min_tokens:
        # allow single tokens only with suffix hints (e.g., "...model")
        if len(tokens) == 1 and allow_single_if_suffixes:
            for suf in allow_single_if_suffixes:
                if s.endswith(suf.lower()):
                    return True
        # POS check to allow *nouns only* if policy accepts singles
        try:
            tag = pos_tag(word_tokenize(s))[0][1]
        except Exception:
            tag = None
        return len(tokens) >= min_tokens and (tag in ADMISSIBLE_POS)
    # multiword → prefer NP-like heads
    ht = head_token(s)
    return (ht in [h.lower() for h in head_hints]) or any(s.endswith(suf.lower()) for suf in allow_single_if_suffixes)

def cosine(u, v):
    from numpy import dot
    from numpy.linalg import norm
    return float(dot(u, v) / (norm(u) * norm(v) + 1e-12))

def expand_terms_w2v(model: Word2Vec, seed_terms: List[str], policy: Dict[str, Any]) -> List[str]:
    # policy defaults
    min_tokens = int(policy.get("min_tokens", 2))
    min_seed_support = int(policy.get("min_seed_support", 2))
    thresh_seed = float(policy.get("thresh_seed", 0.55))
    thresh_centroid = float(policy.get("thresh_centroid", 0.60))
    mutual = bool(policy.get("mutual", False))
    allow_single_if_suffixes = list(policy.get("allow_single_if_suffixes", ["model","estimation","prediction","forecast"]))
    head_hints = list(policy.get("head_hints", ["model","modelling","modeling","estimation","prediction","forecast"]))
    blocklist = set(policy.get("blocklist", [])) | set(policy.get("block_list", [])) | set(DEFAULT_BLOCKLIST)

    seeds = [t.strip().lower().replace(" ", "_") for t in seed_terms if isinstance(t, str) and t.strip()]
    seeds_in = [t for t in seeds if t in model.wv]
    if not seeds_in:
        return sorted(set(seed_terms))

    # centroid of seed vectors
    import numpy as np
    C = np.mean([model.wv[t] for t in seeds_in], axis=0)

    cand_counts = {}      # candidate -> #seeds that support it
    cand_best_sim = {}    # candidate -> best sim to any seed

    for s in seeds_in:
        for cand, sim in model.wv.most_similar(s, topn=25):
            if sim < thresh_seed:
                continue
            c_norm = cand.lower()
            if not looks_like_term(c_norm, min_tokens=min_tokens,
                                   allow_single_if_suffixes=allow_single_if_suffixes,
                                   head_hints=head_hints, blocklist=blocklist):
                continue
            if mutual:
                try:
                    neigh = [x for x,_ in model.wv.most_similar(cand, topn=50)]
                    if s not in neigh:
                        continue
                except KeyError:
                    continue
            cand_counts[c_norm] = cand_counts.get(c_norm, 0) + 1
            cand_best_sim[c_norm] = max(cand_best_sim.get(c_norm, 0.0), float(sim))

    supported = [c for c, n in cand_counts.items() if n >= min_seed_support]

    scored = []
    for c in supported:
        v = model.wv[c]
        c_cent = cosine(v, C)
        if c_cent >= thresh_centroid:
            score = 0.6 * c_cent + 0.4 * cand_best_sim.get(c, 0.0)
            scored.append((score, c))

    scored.sort(reverse=True)
    expanded = [c.replace("_"," ") for _, c in scored[:12]]

    original = [t for t in seed_terms if isinstance(t, str) and t.strip()]
    return sorted(set(original + expanded))

def expand_ontology(model: Word2Vec, ont: dict) -> dict:
    # Load expansion policy from YAML
    policy = (ont.get("expansion_policy") or {})
    defaults = policy.get("default", {})
    overrides = policy.get("overrides", {})

    # Global blocklist additions
    global_block = set(ont.get("blocklist", []) or [])

    expanded = {}
    for cat, subtree in ont.items():
        # skip meta keys
        if cat in {"expansion_policy", "blocklist", "anchors"}:
            expanded[cat] = subtree
            continue

        # per-category override policy
        p = dict(defaults)
        p.update(overrides.get(cat, {}))
        # merge global blocklist
        p["blocklist"] = list(set(p.get("blocklist", [])) | global_block | DEFAULT_BLOCKLIST)

        if isinstance(subtree, dict):
            expanded[cat] = {}
            for key, terms in subtree.items():
                # model_paradigm is often best kept exact; but let override decide
                if (cat == "model_paradigm") and not bool(p.get("expand", False)):
                    expanded[cat][key] = terms
                else:
                    expanded[cat][key] = expand_terms_w2v(model, terms, p)
        else:
            if (cat == "model_paradigm") and not bool(p.get("expand", False)):
                expanded[cat] = subtree
            else:
                expanded[cat] = expand_terms_w2v(model, subtree, p)
    return expanded

# -------------- Systems (AHU/VAV/HP/...) --------------
def label_system_types(sections, ont):
    sys_map = ont.get("systems", {}) or {}
    sys_rx = {k: any_rx(v) for k, v in sys_map.items()}
    found = set(); ev = defaultdict(list)

    for sec, text in scannable_sections(sections):
        for sys_key, rxs in sys_rx.items():
            hits = find_hits(rxs, text)
            for h in hits:
                found.add(sys_key)
                ev[sys_key].append((sec, h, snippet_around(h, text)))
    return sorted(found), ev

# -------------- Model types by scale + paradigms + exact term hits --------------
def label_model_types(sections, ont, anchors: Dict[str, Dict[str, List[str]]]):
    scale_map = ont.get("scale", {}) or {}
    paradigm_map = ont.get("model_paradigm", {}) or {}

    scale_rx = {s: any_rx(terms) for s, terms in scale_map.items()}
    parad_rx = {p: any_rx(terms) for p, terms in paradigm_map.items()}

    out = {
        s: {
            "present": False,
            "paradigms_scores": defaultdict(float),
            "paradigms": [],
            "term_hits": defaultdict(float),     # term -> weighted hits
            "evidence": defaultdict(list),       # key -> [(sec, key, sentence)]
        } for s in scale_map.keys()
    }

    occ_anchor = DEFAULT_ANCHORS["occupancy_model"]
    occ_anchor = anchors.get("occupancy_model", occ_anchor)

    for sec, text in scannable_sections(sections):
        w = SECTION_WEIGHTS.get(sec, 0.3)
        for sent in sent_tokenize_safe(text):
            sl = sent.lower()

            scales_here = [s for s, rxs in scale_rx.items() if any_hit(rxs, sl)]
            if not scales_here:
                continue

            if any_hit(scale_rx.get("occupancy_model", []), sl):
                if not context_gate(sl, occ_anchor.get("require_any", []), occ_anchor.get("forbid_any", [])):
                    continue

            for p, rxs in parad_rx.items():
                hits = find_hits(rxs, sl)
                if not hits:
                    continue
                for s in scales_here:
                    out[s]["present"] = True
                    out[s]["paradigms_scores"][p] += w
                    out[s]["evidence"][p].append((sec, p, sent.strip()))
                    for h in set(hits):
                        out[s]["term_hits"][h] += w
                        out[s]["evidence"][h].append((sec, h, sent.strip()))

    for s, rec in out.items():
        ps = rec["paradigms_scores"]
        total = sum(ps.values()) or 1.0
        if total > 0 and ps:
            norm = {k: v / total for k, v in ps.items()}
            winners = [k for k, v in norm.items() if v >= 0.28] or [max(ps, key=ps.get)]
        else:
            winners = []
        rec["paradigms"] = winners
    return out

# -------------- Optimization methods + scope --------------
def label_optimization_methods_and_scope(sections, ont):
    meth_map = ont.get("optimization_methods", {}) or {}
    scope_map = {
        "building": ["whole-building", "building-level", "zone-level", "zone model", "ubem", "energyplus", "trnsys"],
        "system": ["ahu", "vav", "heat pump", "chiller", "boiler", "cooling tower", "fan coil", "component", "system-level"],
        "occupancy": ["occupancy", "occupant schedule", "presence", "occupancy profile"],
        "weather": ["weather", "climate", "nwp", "numerical weather prediction", "forecast", "meteorology"],
    }
    meth_rx  = {k: any_rx(v) for k, v in meth_map.items()}
    scope_rx = {k: any_rx(v) for k, v in scope_map.items()}

    methods = set(); scopes = set()
    ev = defaultdict(list)
    pair_scope = defaultdict(lambda: defaultdict(float))

    for sec, text in scannable_sections(sections):
        w = SECTION_WEIGHTS.get(sec, 0.3)
        for sent in sent_tokenize_safe(text):
            sl = sent.lower()
            meth_here = [m for m, rxs in meth_rx.items() if any_hit(rxs, sl)]
            scope_here = [s for s, rxs in scope_rx.items() if any_hit(rxs, sl)]
            if not meth_here and not scope_here:
                continue
            for m in meth_here:
                methods.add(m)
                ev[m].append((sec, m, sent.strip()))
            for s in scope_here:
                scopes.add(s)
                ev[s].append((sec, s, sent.strip()))
            for m in meth_here:
                for s in scope_here:
                    pair_scope[m][s] += w

    method_primary_scope = {m: (max(sc.items(), key=lambda kv: kv[1])[0] if sc else None)
                            for m, sc in pair_scope.items()}
    return sorted(methods), sorted(scopes), method_primary_scope, ev

# -------------- Data collected + resolution --------------

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
            if re.search(rf'\b{re.escape(w)}\b', t):
                hits.append({"section": sec, "value": None, "unit": w})
    return hits

def sampling_to_resolution(sampling_mentions):
    for h in sampling_mentions:
        unit = (h["unit"] or "").lower()
        if unit in ("hz", "/s", "s", "sec", "second", "min", "minute", "/min", "5-min", "15-min", "1-min", "30-s", "10-second", "subhourly"):
            return "subhourly"
    for h in sampling_mentions:
        unit = (h["unit"] or "").lower()
        if unit in ("h", "hr", "hour", "/hour", "hourly", "per hour", "1-hour"):
            return "hourly"
    for h in sampling_mentions:
        unit = (h["unit"] or "").lower()
        if "daily" in unit: return "daily"
        if "monthly" in unit: return "monthly"
        if "year" in unit or "annual" in unit: return "yearly"
    return None

def label_data_types(sections, ont):
    found = set(); ev = defaultdict(list)
    dt_map = ont.get("data_types", {}) or {}
    dt_rx  = {k: any_rx(v) for k, v in dt_map.items()}

    temp_ind_anchor = DEFAULT_ANCHORS["temperature_indoor"]
    temp_out_anchor = DEFAULT_ANCHORS["temperature_outdoor"]

    for sec, text in scannable_sections(sections):
        for sent in sent_tokenize_safe(text):
            sl = sent.lower()
            for dtype, rxs in dt_rx.items():
                hits = find_hits(rxs, sl)
                for h in hits:
                    if dtype in ("air_temperature", "environment"):
                        if context_gate(sl, temp_out_anchor["require_any"], temp_out_anchor["forbid_any"]):
                            key = "air_temperature_outdoor"
                        elif context_gate(sl, temp_ind_anchor["require_any"], temp_ind_anchor["forbid_any"]):
                            key = "air_temperature_indoor"
                        else:
                            key = dtype
                        found.add(key)
                        ev[key].append((sec, h, snippet_around(h, text)))
                    else:
                        found.add(dtype)
                        ev[dtype].append((sec, h, snippet_around(h, text)))
    return sorted(found), ev

def consolidate_data_and_resolution(sections, ont, input_kpi_priority_order):
    data_types, data_ev = label_data_types(sections, ont)

    input_res_map = ont.get("input_resolution", {}) or {}
    input_res_rx = {k: any_rx(v) for k, v in input_res_map.items()}
    input_resolutions = set(); input_res_ev = defaultdict(list)
    for sec, text in scannable_sections(sections):
        for res, rxs in input_res_rx.items():
            hits = find_hits(rxs, text)
            if hits:
                input_resolutions.add(res)
                input_res_ev[res].append((sec, res, snippet_around(hits[0], text)))

    sampling = extract_sampling(sections)
    sampling_bucket = sampling_to_resolution(sampling)
    final_input_resolution = sampling_bucket or (input_kpi_priority_order[0] if input_kpi_priority_order else None)
    for r in input_kpi_priority_order:
        if r in input_resolutions:
            final_input_resolution = r
            break
    return {
        "data_types": data_types,
        "data_evidence": data_ev,
        "input_resolutions_all": sorted(input_resolutions),
        "input_resolution_primary": final_input_resolution or "",
        "input_resolution_evidence": input_res_ev,
        "sampling_mentions": sampling
    }

# -------------- KPI + resolution --------------
KPI_FULL_RX = [
    re.compile(r'\bhourly\s+(energy\s+(?:performance|use|consumption|demand)|power)\b', re.I),
    re.compile(r'\bannual(?:ly)?\s+(energy\s+(?:use|consumption|demand)|emissions|cost)\b', re.I),
    re.compile(r'\byearly\s+(energy\s+(?:use|consumption)|emissions|cost)\b', re.I),
]

def label_kpis_with_resolution(sections, ont, kpi_priority_order):
    kpi_map = ont.get("kpi", {}) or {}
    res_map = ont.get("kpi_resolution", {}) or {}

    kpi_rx = {k: any_rx(v) for k, v in kpi_map.items()}
    res_rx = {k: any_rx(v) for k, v in res_map.items()}

    kpis = set(); reses = set()
    ev = defaultdict(list)
    kpi_res_pairs = defaultdict(lambda: defaultdict(float))

    for sec, text in scannable_sections(sections):
        w = SECTION_WEIGHTS.get(sec, 0.3)
        for para in re.split(r'\n\s*\n+', text):
            if not para.strip():
                continue
            k_here = [k for k, rxs in kpi_rx.items() if any_hit(rxs, para)]
            r_here = [r for r, rxs in res_rx.items() if any_hit(rxs, para)]

            for rx in KPI_FULL_RX:
                for m in rx.finditer(para):
                    phrase = m.group(0)
                    pl = phrase.lower()
                    if "hourly" in pl:
                        res = "hourly"
                    elif "annual" in pl or "yearly" in pl:
                        res = "yearly"
                    else:
                        res = None
                    if "performance" in pl:
                        ksel = "energy_performance"
                    elif "power" in pl:
                        ksel = "power"
                    else:
                        ksel = "energy_use"
                    if res:
                        kpis.add(ksel); reses.add(res)
                        ev[f"{ksel}@{res}"].append((sec, phrase, phrase))
                        kpi_res_pairs[ksel][res] += w

            if k_here:
                for k in k_here:
                    kpis.add(k)
                    ev[k].append((sec, k, para[:240].strip()))
                if r_here:
                    for r in r_here:
                        reses.add(r)
                        ev[r].append((sec, r, para[:240].strip()))
                        for k in k_here:
                            kpi_res_pairs[k][r] += w

    primary_by_kpi = {}
    for k, res_scores in kpi_res_pairs.items():
        if not res_scores:
            continue
        for r in kpi_priority_order:
            if r in res_scores:
                primary_by_kpi[k] = r
                break
        if k not in primary_by_kpi:
            primary_by_kpi[k] = max(res_scores.items(), key=lambda kv: kv[1])[0]

    global_primary = ""
    pairs = [((k, r), v) for k, scores in kpi_res_pairs.items() for r, v in scores.items()]
    if pairs:
        global_primary = max(pairs, key=lambda kv: kv[1])[0][1]

    return sorted(kpis), sorted(reses), primary_by_kpi, global_primary, ev

# -------------- Applications --------------
def label_applications(sections, ont):
    apps_map = ont.get("applications", {}) or {}
    apps_rx = {k: any_rx(v) for k, v in apps_map.items()}
    apps = set(); ev = defaultdict(list)
    for sec, text in scannable_sections(sections):
        for app, rxs in apps_rx.items():
            hits = find_hits(rxs, text)
            if hits:
                apps.add(app)
                ev[app].append((sec, hits[0], snippet_around(hits[0], text)))
    return sorted(apps), ev

# -------------- Pipeline driver --------------
def analyze_paper(doc, ont):
    anchors = anchors_from_yaml(ont)
    raw = doc.get('full-text-retrieval-response', {}).get('originalText', '')
    sections = split_sections(raw)

    # Applications
    apps, app_ev = label_applications(sections, ont)

    # Systems
    systems, systems_ev = label_system_types(sections, ont)

    # Models by scale
    models_by_scale = label_model_types(sections, ont, anchors)

    # Optimization
    opt_methods, opt_scopes, method_primary_scope, opt_ev = label_optimization_methods_and_scope(sections, ont)

    # Data + resolution
    input_kpi_priority_order = list((ont.get("input_resolution") or {}).keys())
    data_block = consolidate_data_and_resolution(sections, ont, input_kpi_priority_order)

    # KPIs + resolution
    kpi_priority_order = list((ont.get("kpi_resolution") or {}).keys())
    kpis_used, kpi_resolutions, kpi_primary_by_kpi, kpi_global_primary_res, kpi_ev = \
        label_kpis_with_resolution(sections, ont, kpi_priority_order)

    # Flat canonical model names across scales
    canonical_models = set()
    for rec in models_by_scale.values():
        for term in rec.get("term_hits", {}).keys():
            canonical_models.add(canonicalize_term(term))

    return {
        "applications": apps,
        "applications_evidence": app_ev,

        "systems": systems,
        "systems_evidence": systems_ev,

        "models_by_scale": models_by_scale,
        "models_canonical": sorted(canonical_models),

        "optimization_methods": opt_methods,
        "optimization_scopes": opt_scopes,
        "optimization_method_primary_scope": method_primary_scope,
        "optimization_evidence": opt_ev,

        "data_types": data_block["data_types"],
        "data_evidence": data_block["data_evidence"],
        "input_resolutions_all": data_block["input_resolutions_all"],
        "input_resolution_primary": data_block["input_resolution_primary"],
        "input_resolution_evidence": data_block["input_resolution_evidence"],
        "sampling_mentions": data_block["sampling_mentions"],

        "kpis_used": kpis_used,
        "kpi_resolutions_all": kpi_resolutions,
        "kpi_primary_by_kpi": kpi_primary_by_kpi,
        "kpi_global_primary_resolution": kpi_global_primary_res,
        "kpi_evidence": kpi_ev,
    }

# -------------- IO --------------
def load_jsons(dirpath):
    for fn in os.listdir(dirpath):
        if fn.endswith(".json"):
            p = os.path.join(dirpath, fn)
            with open(p, "r", encoding="utf-8") as f:
                yield fn, json.load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Folder with Elsevier JSONs")
    ap.add_argument("--ontology", required=True, help="Ontology YAML")
    ap.add_argument("--model", required=True, help="Path to trained gensim Word2Vec model")
    ap.add_argument("--output_csv", default="focused_papers_w2v.csv", help="Output CSV path")
    ap.add_argument("--output_json", default="", help="Optional evidence JSON")
    args = ap.parse_args()

    # Load ontology (raw)
    with open(args.ontology, "r", encoding="utf-8") as f:
        ont_raw = yaml.safe_load(f) or {}

    # Load Word2Vec model
    w2v = Word2Vec.load(args.model)

    # Expand ontology with constrained policy
    ont = expand_ontology(w2v, ont_raw)

    items = list(load_jsons(args.input_dir))

    rows = []
    for fname, doc in tqdm(items, total=len(items), desc="Analyzing"):
        try:
            res = analyze_paper(doc, ont)
            rows.append((fname, res))
        except Exception as e:
            print(f"[WARN] {fname} failed: {e}")

    # ---- CSV shaping ----
    def nm(x):
        return "NM" if not x else (",".join(x) if isinstance(x, (list,set,tuple)) else str(x))

    def term_scores_to_str(d):
        if not d: return "NM"
        return ";".join(f"{k}:{v:.2f}" for k,v in sorted(d.items(), key=lambda kv: kv[1], reverse=True))

    def summarize_models_by_scale(m):
        rows = {}
        for scale, rec in (m or {}).items():
            rows[f"{scale}_paradigms"] = nm(rec.get("paradigms"))
            rows[f"{scale}_paradigm_terms"] = term_scores_to_str(rec.get("term_hits", {}))
        return rows

    out_rows = []
    for fname, r in rows:
        model_summary = summarize_models_by_scale(r.get("models_by_scale"))
        sampling_str = ";".join(f'{h["section"]}:{h["value"] or ""}{h["unit"]}' for h in r.get("sampling_mentions") or []) or "NM"

        out_rows.append({
            "file": fname,
            "applications": nm(r.get("applications")),
            **model_summary,
            "models_canonical": nm(r.get("models_canonical")),
            "systems": nm(r.get("systems")),
            "optimization_methods": nm(r.get("optimization_methods")),
            "optimization_scopes": nm(r.get("optimization_scopes")),
            "optim_method_primary_scope": ";".join(f"{m}:{s or 'NM'}" for m,s in (r.get("optimization_method_primary_scope") or {}).items()) or "NM",
            "data_types": nm(r.get("data_types")),
            "input_resolution_primary": r.get("input_resolution_primary") or "NM",
            "input_resolutions_all": nm(r.get("input_resolutions_all")),
            "sampling_mentions": sampling_str,
            "kpis_used": nm(r.get("kpis_used")),
            "kpi_resolutions_all": nm(r.get("kpi_resolutions_all")),
            "kpi_primary_by_kpi": ";".join(f"{k}:{v}" for k,v in (r.get("kpi_primary_by_kpi") or {}).items()) or "NM",
            "kpi_global_primary_resolution": r.get("kpi_global_primary_resolution") or "NM",
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
