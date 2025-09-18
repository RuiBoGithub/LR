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
    "results": 1.0, "evaluation": 0.75,
    "abstract": 0.75, "preamble": 0.75,
    "introduction": 0.05, "related": 0.05, "literature": 0.05, "background": 0.05, "review": 0.05,
    "discussion": 0.25, "conclusion": 0.75, "future": 0.05, "appendix": 0.25
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


# ---- Sentence splitting helper----
def sent_tokenize_safe(text: str):
    try:
        from nltk.tokenize import sent_tokenize
        return sent_tokenize(text or "")
    except Exception:
        return re.split(r'(?<=[\.\?\!])\s+', text or "")

# ---------------- Labelers ----------------

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

    for sec, text in sections.items():
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

    for sec, text in sections.items():
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

    for sec, text in sections.items():
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
    for sec, text in sections.items():
        t = (text or "").lower()
        for mode, terms in mode_map.items():
            ft = find_terms(t, terms, max_hits=3)
            if ft:
                found.add(mode); ev[mode].extend([(sec,)*1 + h for h in ft])
    return sorted(found), ev


# ---------------- Core paper analysis ----------------
def analyze_paper(doc, ont_expanded, kpi_priority_order, input_kpi_priority_order):
    raw = doc['full-text-retrieval-response']['originalText']
    sections = split_sections(raw)

    paradigms, model_hits, model_ev, paradigm_scores = label_models_multi(
        sections, model_paradigm_terms=ont_expanded.get("model_paradigm", {})
    )
    scales, scale_ev = label_scale(sections, ont_expanded)
        
    data_types, data_ev = label_data_types(sections, ont_expanded)

    # Per-scale paradigms (white/grey/black + specific terms)
    per_scale = label_paradigms_by_scale(sections, ont_expanded)
    opt_per_scale = label_optimization_by_scale(sections, ont_expanded)


    # Online learning (global + per-scale)
    online_flag, online_ev, online_per_scale, online_ev_per_scale = label_online_learning(sections, ont_expanded)

    return {
        "paradigms": paradigms,
        "paradigm_scores": paradigm_scores,
        "model_hits": model_hits,
        "model_evidence": model_ev,
        "scale": scales,
        "scale_evidence": scale_ev,
        "data_types": data_types,
        "data_evidence": data_ev,
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

            # data + sampling
            "data_types": to_str(r["data_types"]),

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

