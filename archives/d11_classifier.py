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

# ---------------- Config knobs ----------------
CONTEXT_SUFFIXES = [
    "model", "modelling", "modeling", "simulation",
    "estimation", "analytics", "prediction", "control", "representation"
]
# allow “occupancy-sensitive”, “building_energy”, etc.
def _flexify(term: str) -> str:
    # spaces/underscores/hyphens interchangeable, allow extra word chars at end
    t = re.sub(r"\s+", r"[ _-]+", re.escape(term.strip()))
    return t + r"\w*"

def _context_variants(base: str):
    base = base.strip()
    out = {base}
    for suf in CONTEXT_SUFFIXES:
        out.add(f"{base} {suf}")
    return sorted(out)

# ---------------- Simple sent splitter ----------------
def sents(text: str):
    try:
        from nltk.tokenize import sent_tokenize
        return sent_tokenize(text or "")
    except Exception:
        return re.split(r'(?<=[\.\?\!])\s+', text or "")

# ---------------- Ontology expansion ----------------
def expand_terms(model: Word2Vec, seeds, topn=20, thresh=0.60, min_count=10):
    seeds = [s for s in (seeds or []) if isinstance(s, str) and s.strip()]
    out = set([s.lower() for s in seeds])
    for t in seeds:
        key = t.replace(" ", "_")
        if key not in model.wv:
            continue
        for cand, sim in model.wv.most_similar(key, topn=topn):
            if sim < thresh:
                continue
            # freq guard
            try:
                if model.wv.get_vecattr(cand, "count") < min_count:
                    continue
            except Exception:
                pass
            out.add(cand.replace("_", " ").lower())
    return sorted(out)

def expand_ontology(model: Word2Vec, ont: dict, do_not_expand: set):
    expanded = {}
    for cat, subtree in (ont or {}).items():
        if not isinstance(subtree, dict):
            # flat list bucket
            if cat in do_not_expand:
                expanded[cat] = sorted(set([str(x).lower() for x in (subtree or [])]))
            else:
                expanded[cat] = expand_terms(model, subtree)
            continue
        expanded[cat] = {}
        for key, seeds in subtree.items():
            if cat in do_not_expand:
                expanded[cat][key] = sorted(set([str(x).lower() for x in (seeds or [])]))
            else:
                expanded[cat][key] = expand_terms(model, seeds)
    return expanded

# ---------------- Regex builders ----------------
def build_regexes_from_terms(terms):
    """
    Build case-insensitive regex objects for a list of base terms,
    adding context variants and flexible separators.
    """
    regs = []
    bases = set()
    for t in (terms or []):
        t = str(t).strip()
        if not t:
            continue
        bases.add(t)
        for v in _context_variants(t):
            regs.append(re.compile(r"\b" + _flexify(v) + r"\b", re.I))
    # also include the bare base with flexible ending
    for b in bases:
        regs.append(re.compile(r"\b" + _flexify(b) + r"\b", re.I))
    return regs

def search_terms_sentence(sentence: str, regs, max_hits=5):
    hits = []
    for rx in regs:
        m = rx.search(sentence)
        if m:
            tok = sentence[m.start():m.end()]
            hits.append(tok)
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

# ---------------- Core analysis ----------------
def analyze_paper(doc, ont):
    """
    Workflow:
      1) detect system types (from scale.system_model seeds)
      2) detect modelling by scale (building/system/weather/occupancy) -> collect terms
      3) within the same sentence, detect paradigms (white/grey/black) and link them to the scale
      4) detect optimization methods and terms
    Outputs CSV-ready fields + JSON evidence.
    """
    text = doc["full-text-retrieval-response"]["originalText"]
    sentences = sents(text)

    # Prepare regex maps
    scale_map = ont.get("scale", {}) or {}
    # Disturbances (weather/occupancy)
    weather_terms = (scale_map.get("climate_model") or []) + ["weather", "meteorological", "numerical weather prediction", "nwp"]
    occupancy_terms = (scale_map.get("occupancy_model") or []) + ["occupancy", "occupant", "presence", "people count"]

    scale_regs = {k: build_regexes_from_terms(v) for k, v in scale_map.items()}
    weather_regs = build_regexes_from_terms(weather_terms)
    occupancy_regs = build_regexes_from_terms(occupancy_terms)

    # System types dictionary = the *base tokens* from system_model bucket
    system_type_seeds = []
    for t in (scale_map.get("system_model") or []):
        # strip trailing ' model' variants
        tt = t.lower().strip()
        tt = re.sub(r"\bmodel(l)?ing?\b", "", tt).strip()
        system_type_seeds.append(tt)
    system_regs = build_regexes_from_terms(sorted(set(system_type_seeds)))

    # Paradigms + optimization
    paradigms_map = ont.get("model_paradigm", {}) or {}
    opt_methods_map = ont.get("optimization_methods", {}) or {}

    paradigm_regs = {p: build_regexes_from_terms(terms) for p, terms in paradigms_map.items()}
    opt_regs = {cat: build_regexes_from_terms(terms) for cat, terms in opt_methods_map.items()}

    # Accumulators
    systems_detected = set()
    building_mod_terms, system_mod_terms = set(), set()
    weather_mod_terms, occupancy_mod_terms = set(), set()

    paradigms_found = set()
    paradigm_terms = set()

    opt_methods_found = set()
    opt_method_terms = set()

    # Sentence-level pass
    for sent in sentences:
        sl = sent.lower()

        # (A) System types
        sys_hits = search_terms_sentence(sl, system_regs, max_hits=10)
        if sys_hits:
            systems_detected.update(h.strip() for h in sys_hits)

        # (B) Scales presence & modelling terms
        bm_hits = search_terms_sentence(sl, scale_regs.get("building_model", []), max_hits=10)
        sm_hits = search_terms_sentence(sl, scale_regs.get("system_model", []), max_hits=10)
        cm_hits = search_terms_sentence(sl, weather_regs, max_hits=10)
        om_hits = search_terms_sentence(sl, occupancy_regs, max_hits=10)

        if bm_hits:
            building_mod_terms.update(bm_hits)
        if sm_hits:
            system_mod_terms.update(sm_hits)
        if cm_hits:
            weather_mod_terms.update(cm_hits)
        if om_hits:
            occupancy_mod_terms.update(om_hits)

        # (C) Paradigms in the same sentence (anchor to scales if present)
        scales_here = []
        if bm_hits: scales_here.append("building_model")
        if sm_hits: scales_here.append("system_model")
        if cm_hits: scales_here.append("climate_model")
        if om_hits: scales_here.append("occupancy_model")

        if scales_here:
            # look for any paradigm cue in this sentence
            for p, regs in paradigm_regs.items():
                p_hits = search_terms_sentence(sl, regs, max_hits=10)
                if p_hits:
                    paradigms_found.add(p)
                    paradigm_terms.update(p_hits)

        # (D) Optimization methods
        for cat, regs in opt_regs.items():
            o_hits = search_terms_sentence(sl, regs, max_hits=10)
            if o_hits:
                opt_methods_found.add(cat)
                opt_method_terms.update(o_hits)

    # Fill NM where appropriate
    def nm_list(xs):
        xs = sorted(set([x.strip() for x in xs if str(x).strip()]))
        return xs if xs else ["NM"]

    # Build JSON record
    result = {
        "systems": sorted(set(systems_detected)),
        "building_modelling_terms": sorted(set(building_mod_terms)),
        "system_modelling_terms": sorted(set(system_mod_terms)),
        "weather_modelling_terms": sorted(set(weather_mod_terms)),
        "occupancy_modelling_terms": sorted(set(occupancy_mod_terms)),
        "paradigms": sorted(set(paradigms_found)) if paradigms_found else ["NM"],
        "paradigm_term_hits": sorted(set(paradigm_terms)) if paradigm_terms else ["NM"],
        "optimization_methods": sorted(set(opt_methods_found)) if opt_methods_found else ["NM"],
        "optimization_method_terms": sorted(set(opt_method_terms)) if opt_method_terms else ["NM"],
    }

    # If we detected building/system/weather/occupancy modelling but no paradigm,
    # keep "NM" in paradigms — you asked to report NM when none.
    return result

def worker(args):
    fname, doc, ont = args
    res = analyze_paper(doc, ont)
    return fname, res

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Folder with Elsevier JSONs")
    ap.add_argument("--model", default="word2vec.model", help="Path to trained gensim Word2Vec model")
    ap.add_argument("--ontology", required=True, help="Seed ontology YAML/YML")
    ap.add_argument("--output_csv", default="nlp_output/focused_papers.csv", help="Output CSV path")
    ap.add_argument("--output_json", default="", help="Optional evidence JSON")
    ap.add_argument("--jobs", type=int, default=8, help="Parallel workers")
    ap.add_argument("--no_expand", action="store_true",
                    help="Disable Word2Vec expansion and use ontology seeds as-is.")
    ap.add_argument("--do_not_expand", default="",
                    help="Comma-separated ontology buckets to NOT expand (e.g., model_paradigm,optimization_methods)")
    ap.add_argument("--exp_topn", type=int, default=20)
    ap.add_argument("--exp_thresh", type=float, default=0.60)
    ap.add_argument("--exp_min_count", type=int, default=10)
    args = ap.parse_args()

    # Load ontology
    with open(args.ontology, "r", encoding="utf-8") as f:
        ont = yaml.safe_load(f) or {}

    # Expand ontology if requested
    if not args.no_expand:
        w2v = Word2Vec.load(args.model)
        do_not_expand = set([s.strip() for s in (args.do_not_expand or "").split(",") if s.strip()])
        # monkey-patch expand params
        global expand_terms
        def expand_terms(model, seeds, topn=args.exp_topn, thresh=args.exp_thresh, min_count=args.exp_min_count):
            seeds = [s for s in (seeds or []) if isinstance(s, str) and s.strip()]
            out = set([s.lower() for s in seeds])
            for t in seeds:
                key = t.replace(" ", "_")
                if key not in model.wv:
                    continue
                for cand, sim in model.wv.most_similar(key, topn=topn):
                    if sim < thresh:
                        continue
                    try:
                        if model.wv.get_vecattr(cand, "count") < min_count:
                            continue
                    except Exception:
                        pass
                    out.add(cand.replace("_", " ").lower())
            return sorted(out)

        ont = expand_ontology(w2v, ont, do_not_expand)
    # else keep seeds as-is

    items = list(load_jsons(args.input_dir))
    tasks = [(fname, doc, ont) for fname, doc in items]

    rows = []
    if args.jobs and args.jobs > 1:
        with cf.ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futures = [ex.submit(worker, t) for t in tasks]
            for fut in tqdm(cf.as_completed(futures), total=len(futures), desc="Classifying"):
                try:
                    rows.append(fut.result())
                except Exception as e:
                    print(f"[WARN] worker failed: {e}")
    else:
        for t in tqdm(tasks, total=len(tasks), desc="Classifying"):
            try:
                rows.append(worker(t))
            except Exception as e:
                print(f"[WARN] worker failed: {e}")

    # CSV
    out_rows = []
    def nm_join(xs):
        xs = [x for x in (xs or []) if str(x).strip()]
        return ";".join(sorted(set(xs))) if xs else "NM"

    for fname, r in rows:
        out_rows.append({
            "file": fname,
            "systems": nm_join(r.get("systems")),
            "building_modelling_terms": nm_join(r.get("building_modelling_terms")),
            "system_modelling_terms": nm_join(r.get("system_modelling_terms")),
            "weather_modelling_terms": nm_join(r.get("weather_modelling_terms")),
            "occupancy_modelling_terms": nm_join(r.get("occupancy_modelling_terms")),
            "paradigms": nm_join(r.get("paradigms")),
            "paradigm_term_hits": nm_join(r.get("paradigm_term_hits")),
            "optimization_methods": nm_join(r.get("optimization_methods")),
            "optimization_method_terms": nm_join(r.get("optimization_method_terms")),
        })

    df = pd.DataFrame(out_rows).sort_values("file")
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"[OK] Wrote {args.output_csv} with {len(df)} rows.")

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        ev = {fname: r for fname, r in rows}
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(ev, f, ensure_ascii=False, indent=2)
        print(f"[OK] Wrote evidence JSON → {args.output_json}")

if __name__ == "__main__":
    main()
