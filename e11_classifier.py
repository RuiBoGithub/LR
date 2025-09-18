#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, yaml, argparse
import concurrent.futures as cf
from collections import defaultdict
import numpy as np
import nltk
from gensim.models import Word2Vec
import pandas as pd
from tqdm import tqdm

# ---------------- NLTK ----------------
def ensure_nltk():
    pkgs = [
        ("tokenizers/punkt", "punkt"),
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

def sents(text: str):
    try:
        from nltk.tokenize import sent_tokenize
        return sent_tokenize(text or "")
    except Exception:
        return re.split(r'(?<=[\.\?\!])\s+', text or "")

# ---------------- Context ----------------
MODELLING_CONTEXT_RX = re.compile(
    r"\b("
    r"model|models|modelled|modeled|modelling|modeling|"
    r"estimate|estimates|estimated|estimation|"
    r"predict|predicts|predicted|prediction|predictive|"
    r"simulate|simulates|simulated|simulation|"
    r"identif(?:y|ies|ied|ication)|"
    r"infer|infers|inferred|inference"
    r")\b", re.I
)
def has_modelling_context(sentence: str) -> bool:
    return MODELLING_CONTEXT_RX.search(sentence or "") is not None

# ---------------- Ontology expansion ----------------
def expand_terms(model: Word2Vec, seeds, topn=20, thresh=0.70, min_count=10):
    """Guarded expansion with similarity threshold and identifier filtering."""
    seeds = [s for s in (seeds or []) if isinstance(s, str) and s.strip()]
    out = set([s.lower() for s in seeds])
    identifiers = {"model", "learning", "algorithm", "method", "estimation", "prediction", "simulation"}
    for t in seeds:
        key = t.replace(" ", "_")
        if key not in model.wv:
            continue
        for cand, sim in model.wv.most_similar(key, topn=topn):
            if sim < thresh: continue
            try:
                if model.wv.get_vecattr(cand, "count") < min_count:
                    continue
            except Exception: pass
            s = cand.replace("_", " ").lower()
            # require identifiers + 2-gram minimum
            if len(s.split()) < 2: continue
            if not any(idf in s for idf in identifiers): continue
            out.add(s)
    return sorted(out)

def expand_ontology(model: Word2Vec, ont: dict, do_not_expand: set,
                    topn=20, thresh=0.70, min_count=10):
    def _expand_list(lst):
        return expand_terms(model, lst, topn=topn, thresh=thresh, min_count=min_count)
    expanded = {}
    for bucket, subtree in (ont or {}).items():
        if isinstance(subtree, dict):
            expanded[bucket] = {}
            for k, v in subtree.items():
                expanded[bucket][k] = (sorted(set(map(str.lower, v)))
                                       if bucket in do_not_expand else _expand_list(v))
        elif isinstance(subtree, list):
            expanded[bucket] = (sorted(set(map(str.lower, subtree)))
                                if bucket in do_not_expand else _expand_list(subtree))
        else:
            expanded[bucket] = subtree
    return expanded

# ---------------- Canonicalisation ----------------
def build_canon_map(ont):
    """Build surface→canonical map for every YAML term."""
    canon_map = defaultdict(set)
    for bucket, subtree in (ont or {}).items():
        if isinstance(subtree, dict):
            for k, terms in subtree.items():
                for t in (terms or []):
                    t = str(t).lower().strip()
                    canon_map[t].add(k if bucket != "scale" else t)
        elif isinstance(subtree, list):
            for t in subtree:
                t = str(t).lower().strip()
                canon_map[t].add(t)
    return canon_map

def canonicalize(term, canon_map):
    """Map surface form to canonical YAML term if possible."""
    t = term.lower().strip()
    # strip suffixes like -based, simulation, models, etc.
    t = re.sub(r"[- ]?(based|simulation|model(s)?|approach(es)?|method(s)?|technique(s)?)$", "", t).strip()
    for canon, variants in canon_map.items():
        if t == canon or t in variants:
            return canon
    return None

# ---------------- Regex search ----------------
def _flexify(term: str) -> str:
    t = re.sub(r"\s+", r"[ _-]+", re.escape(term.strip()))
    return t + r"([a-zA-Z0-9_-]+)?"

def build_regexes_from_terms(terms):
    regs = []
    for t in (terms or []):
        if not t: continue
        regs.append(re.compile(r"\b" + _flexify(t) + r"\b", re.I))
    return regs

def search_terms_sentence(sentence: str, regs, canon_map, max_hits=5):
    hits = []
    for rx in regs:
        m = rx.search(sentence)
        if m:
            raw = sentence[m.start():m.end()]
            canon = canonicalize(raw, canon_map)
            if canon: hits.append(canon)
            if len(hits) >= max_hits:
                break
    return hits

# ---------------- Load/IO ----------------
def load_jsons(dirpath):
    for fn in os.listdir(dirpath):
        if fn.endswith(".json"):
            p = os.path.join(dirpath, fn)
            with open(p, "r", encoding="utf-8") as f:
                yield fn, json.load(f)

def make_json_safe(obj):
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, np.generic): return obj.item()
    if isinstance(obj, (list, tuple)): return [make_json_safe(x) for x in obj]
    if isinstance(obj, set): return sorted(make_json_safe(x) for x in obj)
    if isinstance(obj, (dict, defaultdict)):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray): return make_json_safe(obj.tolist())
    return str(obj)

# ---------------- Analysis ----------------
def analyze_paper(doc, ont, canon_map):
    text = doc.get("full-text-retrieval-response", {}).get("originalText", "") or ""
    sentences = sents(text)

    scale_map = ont.get("scale", {}) or {}
    paradigms_map = ont.get("model_paradigm", {}) or {}
    opt_methods_map = ont.get("optimization_methods", {}) or {}

    # regexes
    scale_regs = {k: build_regexes_from_terms(v) for k, v in scale_map.items()}
    paradigm_regs = {p: build_regexes_from_terms(terms) for p, terms in paradigms_map.items()}
    opt_regs = {cat: build_regexes_from_terms(terms) for cat, terms in opt_methods_map.items()}

    # accumulators
    systems, bm_terms, sm_terms, cm_terms, om_terms = set(), set(), set(), set(), set()
    bm_par, sm_par, cm_par, om_par = set(), set(), set(), set()
    bm_par_terms, sm_par_terms, cm_par_terms, om_par_terms = set(), set(), set(), set()
    opt_methods, opt_terms = set(), set()

    for sent in sentences:
        sl = sent.lower()
        # systems
        sys_hits = search_terms_sentence(sl, scale_regs.get("system_model", []), canon_map)
        systems.update(sys_hits)
        # modelling terms
        if has_modelling_context(sl):
            bm_terms.update(search_terms_sentence(sl, scale_regs.get("building_model", []), canon_map))
            sm_terms.update(search_terms_sentence(sl, scale_regs.get("system_model", []), canon_map))
            cm_terms.update(search_terms_sentence(sl, scale_regs.get("climate_model", []), canon_map))
            om_terms.update(search_terms_sentence(sl, scale_regs.get("occupancy_model", []), canon_map))
        # paradigms
        if any([bm_terms, sm_terms, cm_terms, om_terms]):
            for p, regs in paradigm_regs.items():
                hits = search_terms_sentence(sl, regs, canon_map)
                if hits:
                    if bm_terms: bm_par.add(p); bm_par_terms.update(hits)
                    if sm_terms: sm_par.add(p); sm_par_terms.update(hits)
                    if cm_terms: cm_par.add(p); cm_par_terms.update(hits)
                    if om_terms: om_par.add(p); om_par_terms.update(hits)
            for cat, regs in opt_regs.items():
                hits = search_terms_sentence(sl, regs, canon_map)
                if hits:
                    opt_methods.add(cat)
                    opt_terms.update(hits)

    def nm(xs): return sorted(xs) if xs else ["NM"]

    return {
        "systems": nm(systems),
        "building_modelling_terms": nm(bm_terms),
        "system_modelling_terms": nm(sm_terms),
        "weather_modelling_terms": nm(cm_terms),
        "occupancy_modelling_terms": nm(om_terms),
        "building_paradigms": nm(bm_par),
        "system_paradigms": nm(sm_par),
        "weather_paradigms": nm(cm_par),
        "occupancy_paradigms": nm(om_par),
        "building_paradigm_terms": nm(bm_par_terms),
        "system_paradigm_terms": nm(sm_par_terms),
        "weather_paradigm_terms": nm(cm_par_terms),
        "occupancy_paradigm_terms": nm(om_par_terms),
        "optimization_methods": nm(opt_methods),
        "optimization_method_terms": nm(opt_terms),
    }

def worker(args):
    fname, doc, ont, canon_map = args
    return fname, analyze_paper(doc, ont, canon_map)

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--model", default="word2vec.model")
    ap.add_argument("--ontology", required=True)
    ap.add_argument("--output_csv", default="nlp_output/focused_papers.csv")
    ap.add_argument("--output_json", default="")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--no_expand", action="store_true")
    ap.add_argument("--do_not_expand", default="")
    ap.add_argument("--exp_topn", type=int, default=20)
    ap.add_argument("--exp_thresh", type=float, default=0.70)
    ap.add_argument("--exp_min_count", type=int, default=10)
    args = ap.parse_args()

    with open(args.ontology, "r", encoding="utf-8") as f:
        ont = yaml.safe_load(f) or {}

    if not args.no_expand:
        w2v = Word2Vec.load(args.model)
        do_not_expand = set([s.strip() for s in (args.do_not_expand or "").split(",") if s.strip()])
        ont = expand_ontology(
            w2v, ont, do_not_expand,
            topn=args.exp_topn, thresh=args.exp_thresh, min_count=args.exp_min_count
        )

    canon_map = build_canon_map(ont)
    items = list(load_jsons(args.input_dir))
    tasks = [(fname, doc, ont, canon_map) for fname, doc in items]

    rows = []
    if args.jobs > 1:
        with cf.ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futures = [ex.submit(worker, t) for t in tasks]
            for fut in tqdm(cf.as_completed(futures), total=len(futures), desc="Classifying"):
                rows.append(fut.result())
    else:
        for t in tqdm(tasks, total=len(tasks), desc="Classifying"):
            rows.append(worker(t))

    # CSV
    df = pd.DataFrame([{"file": fname, **res} for fname, res in rows])
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"[OK] Wrote {args.output_csv} with {len(df)} rows.")

    # JSON
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        ev = {fname: make_json_safe(res) for fname, res in rows}
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(ev, f, ensure_ascii=False, indent=2)
        print(f"[OK] Wrote {args.output_json}")

if __name__ == "__main__":
    main()
