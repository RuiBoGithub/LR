#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, yaml, argparse
import concurrent.futures as cf
from collections import defaultdict
import numpy as np
import pandas as pd

import nltk
from gensim.models import Word2Vec
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

# ---------------- Sentence splitter ----------------
def sents(text: str):
    try:
        from nltk.tokenize import sent_tokenize
        return sent_tokenize(text or "")
    except Exception:
        return re.split(r'(?<=[\.\?\!])\s+', text or "")

# ---------------- Context gate (model-ish) ----------------
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
def has_modelling_context(text: str) -> bool:
    return MODELLING_CONTEXT_RX.search(text or "") is not None

# ---------------- Variants / regex builders ----------------
IDENTIFIER_TAIL = r"(?:[ _-]?(?:based|model|models|modelling|modeling|simulation|method|methods|controller|control|learning|algorithm))?"

def _token_variants_core(t: str):
    t = (t or "").strip()
    if not t:
        return set()
    base = t
    forms = {
        base,
        base.replace("_", " "),
        base.replace("-", " "),
        base.replace(" ", "_"),
        base.replace(" ", "-"),
    }
    out = set()
    suffixes = ["", " model", " models", " modelling", " modeling", " simulation",
                " method", " methods", " approach", " approaches",
                " algorithm", " algorithms", " controller", " control",
                " based", "-based"]
    for f in forms:
        stem = f
        out.add(stem)
        for suf in suffixes:
            out.add((stem + suf).strip())
        if not stem.endswith("s"):
            out.add(stem + "s")
            for suf in suffixes:
                out.add((stem + "s" + suf).strip())
        for es_end in ("x", "s", "z", "ch", "sh"):
            if stem.endswith(es_end):
                out.add(stem + "es")
                for suf in suffixes:
                    out.add((stem + "es" + suf).strip())
    return {v.strip() for v in out if v.strip()}

def _acronym_family_variants(acronym: str):
    # BCNN / BCNNs / B.C.N.N. / B.C.N.N.s
    letters = list(acronym)
    dotted = r"\.?".join(map(re.escape, letters))
    core = rf"(?:{re.escape(acronym)}|{dotted})"
    return rf"{core}s?"

def build_regexes_from_terms(terms, acronyms_extra=None):
    regs = []
    seen = set()
    acronyms_extra = set(acronyms_extra or [])
    for raw in (terms or []):
        t = str(raw).strip()
        if not t:
            continue
        if re.fullmatch(r"[A-Z]{3,6}", t) or t.upper() in acronyms_extra:
            pat = _acronym_family_variants(t.upper()) + IDENTIFIER_TAIL
            key = ("ACRO", t.upper())
            if key not in seen:
                seen.add(key)
                regs.append(re.compile(rf"\b{pat}\b", re.I))
            continue
        for v in _token_variants_core(t):
            key = v.lower()
            if key in seen:
                continue
            seen.add(key)
            regs.append(re.compile(r"\b" + re.sub(r"\s+", r"[ _-]+", re.escape(v)) + IDENTIFIER_TAIL + r"\b", re.I))
    return regs

def search_terms_sentence(sentence: str, regs, max_hits=20):
    hits = []
    for rx in regs:
        m = rx.search(sentence)
        if m:
            tok = sentence[m.start():m.end()]
            hits.append(tok)
            if len(hits) >= max_hits:
                break
    return hits

# ---------------- Canonicalization / aliasing ----------------
SYSTEM_ALIASES = {
    "ac": "hvac", "acs": "hvac",
    "aircon": "hvac", "air con": "hvac", "air conditioner": "hvac", "air conditioners": "hvac",
    "air-conditioning": "hvac", "air conditioning": "hvac",
    "ahu": "ahu", "ahus": "ahu",
    "doas": "doas",
}

def build_canon_map(ont):
    canon_map = defaultdict(set)
    # scale→system_model canonical terms
    for t in (ont.get("scale", {}).get("system_model") or []):
        canon_map[t.lower()].add(t)
    # paradigms
    for p, terms in (ont.get("model_paradigm") or {}).items():
        for t in terms or []:
            canon_map[t.lower()].add(t)
    # optimization
    for cat, terms in (ont.get("optimization_methods") or {}).items():
        for t in terms or []:
            canon_map[t.lower()].add(t)
    # systems aliases
    for a, tgt in SYSTEM_ALIASES.items():
        canon_map[a.lower()].add(tgt)
    return canon_map

def canonicalize_surface(surface: str, canon_map):
    s = (surface or "").strip().lower()
    s = re.sub(r"(?:[ _-]?(?:based|model|models|modelling|modeling|simulation|method|methods|controller|control|learning|algorithm))\b$", "", s)
    s = re.sub(r"[ _-]+", " ", s).strip()
    if s.endswith("s") and len(s) > 3 and s[:-1] in canon_map:
        s = s[:-1]
    if s in canon_map:
        return sorted(canon_map[s])[0]
    m = re.fullmatch(r"([a-z]{2,6})s?", s)
    if m:
        token = m.group(1).upper().replace(".", "")
        ACRO_CANON = {
            "BCNN": "neural network", "CNN": "neural network", "ANN": "neural network",
            "DNN": "neural network", "RNN": "neural network",
            "LSTM": "lstm", "GRU": "gru",
            "SVM": "svm", "SVR": "svr", "HMM": "hmm", "KNN": "knn",
        }
        if token in ACRO_CANON:
            return ACRO_CANON[token]
    if s in SYSTEM_ALIASES:
        return SYSTEM_ALIASES[s]
    return None

# optional W2V alias snapping (2–4 gram neighbors only)
def build_w2v_alias_table(model, buckets_dict, sim_thresh=0.72, min_count=10):
    if model is None:
        return {}
    alias_map = {}
    for bucket, terms in (buckets_dict or {}).items():
        for seed in (terms or []):
            key = seed.replace(" ", "_")
            if key not in model.wv:
                continue
            alias_map[seed.lower()] = seed
            for cand, sim in model.wv.most_similar(key, topn=40):
                if sim < sim_thresh:
                    continue
                try:
                    if model.wv.get_vecattr(cand, "count") < min_count:
                        continue
                except Exception:
                    pass
                surf = cand.replace("_", " ").lower().strip()
                if len(surf) < 3:
                    continue
                if not (2 <= len(surf.split()) <= 4):
                    continue
                alias_map[surf] = seed
    return alias_map

# ---------------- IO ----------------
def load_jsons(dirpath):
    for fn in os.listdir(dirpath):
        if fn.endswith(".json"):
            p = os.path.join(dirpath, fn)
            try:
                with open(p, "r", encoding="utf-8") as f:
                    yield fn, json.load(f)
            except Exception as e:
                print(f"[WARN] Failed to read {fn}: {e}")

# ---------------- Anchors & windows ----------------
def window_indices(i, n, k=1):
    lo = max(0, i - k); hi = min(n - 1, i + k)
    return list(range(lo, hi + 1))

# ---------------- Analysis: Pass A (anchor-first) ----------------
def analyze_anchor_first(doc, ont, canon_map, w2v=None, context_k=1):
    text = doc.get("full-text-retrieval-response", {}).get("originalText", "") or ""
    sent_list = sents(text)
    n = len(sent_list)

    scale = ont.get("scale", {}) or {}
    paradigms = ont.get("model_paradigm", {}) or {}
    opt = ont.get("optimization_methods", {}) or {}

    # Regexes
    scale_rx = {k: build_regexes_from_terms(v) for k, v in scale.items()}
    # Expand paradigms with acronym families
    par_rx = {p: build_regexes_from_terms(
        terms, acronyms_extra={"CNN","BCNN","ANN","DNN","RNN","LSTM","GRU","SVM","SVR","HMM","KNN"}
    ) for p, terms in (paradigms or {}).items()}
    opt_rx = {cat: build_regexes_from_terms(terms) for cat, terms in (opt or {}).items()}

    # system types (strip "-model" words if any were kept)
    sys_terms = []
    for t in (scale.get("system_model") or []):
        tt = re.sub(r"\bmodel(l)?ing?\b", "", str(t).lower()).strip()
        if tt:
            sys_terms.append(tt)
    system_rx = build_regexes_from_terms(sorted(set(sys_terms)))

    # w2v aliases into canon_map (snap near-misses)
    if w2v is not None:
        alias = build_w2v_alias_table(w2v, {
            "system_model": scale.get("system_model", []),
            "whitebox": paradigms.get("whitebox", []),
            "greybox": paradigms.get("greybox", []),
            "blackbox": paradigms.get("blackbox", []),
        }, sim_thresh=0.72, min_count=10)
        for k, v in alias.items():
            canon_map.setdefault(k, set()).add(v)

    # anchors
    anchors = [set() for _ in range(n)]
    systems_detected = set()

    for i, s in enumerate(sent_list):
        sl = s.lower()
        if search_terms_sentence(sl, system_rx):
            anchors[i].add("system_model")
            for tok in search_terms_sentence(sl, system_rx):
                can = canonicalize_surface(tok, canon_map)
                if can:
                    systems_detected.add(can)
        for k in ("building_model","system_model","climate_model","occupancy_model"):
            if search_terms_sentence(sl, scale_rx.get(k, [])):
                anchors[i].add(k)

    # collect results
    out = {
        "systems": sorted(systems_detected) if systems_detected else ["NM"],
        "building_modelling_terms": set(),
        "system_modelling_terms": set(),
        "weather_modelling_terms": set(),
        "occupancy_modelling_terms": set(),
        "building_paradigms": set(),
        "system_paradigms": set(),
        "weather_paradigms": set(),
        "occupancy_paradigms": set(),
        "building_paradigm_terms": set(),
        "system_paradigm_terms": set(),
        "weather_paradigm_terms": set(),
        "occupancy_paradigm_terms": set(),
        "optimization_methods": set(),
        "optimization_method_terms": set(),
    }

    for i, s in enumerate(sent_list):
        sl = s.lower()
        if not anchors[i]:
            continue
        ctx = " ".join(sent_list[j].lower() for j in window_indices(i, n, k=context_k))
        # modelling terms only if context gate is on
        if has_modelling_context(ctx):
            if "building_model" in anchors[i]:
                out["building_modelling_terms"].update(search_terms_sentence(ctx, scale_rx.get("building_model", [])))
            if "system_model" in anchors[i]:
                hits = []
                hits += search_terms_sentence(ctx, scale_rx.get("system_model", []))
                hits += search_terms_sentence(ctx, system_rx)
                out["system_modelling_terms"].update(hits)
            if "climate_model" in anchors[i]:
                out["weather_modelling_terms"].update(search_terms_sentence(ctx, scale_rx.get("climate_model", [])))
            if "occupancy_model" in anchors[i]:
                out["occupancy_modelling_terms"].update(search_terms_sentence(ctx, scale_rx.get("occupancy_model", [])))

            # paradigms anchored
            for p, rx in par_rx.items():
                ph = search_terms_sentence(ctx, rx)
                if not ph:
                    continue
                if "building_model" in anchors[i]:
                    out["building_paradigms"].add(p); out["building_paradigm_terms"].update(ph)
                if "system_model" in anchors[i]:
                    out["system_paradigms"].add(p); out["system_paradigm_terms"].update(ph)
                if "climate_model" in anchors[i]:
                    out["weather_paradigms"].add(p); out["weather_paradigm_terms"].update(ph)
                if "occupancy_model" in anchors[i]:
                    out["occupancy_paradigms"].add(p); out["occupancy_paradigm_terms"].update(ph)

            # optimization anchored
            for cat, rx in opt_rx.items():
                oh = search_terms_sentence(ctx, rx)
                if oh:
                    out["optimization_methods"].add(cat)
                    out["optimization_method_terms"].update(oh)

    # stringify NM
    for k in list(out.keys()):
        if isinstance(out[k], set):
            out[k] = sorted(out[k]) if out[k] else ["NM"]
    return out

# ---------------- Analysis: Pass B (model-first) ----------------
def analyze_model_first(doc, ont, canon_map, w2v=None, context_k=1):
    text = doc.get("full-text-retrieval-response", {}).get("originalText", "") or ""
    sent_list = sents(text)
    n = len(sent_list)

    scale = ont.get("scale", {}) or {}
    paradigms = ont.get("model_paradigm", {}) or {}
    opt = ont.get("optimization_methods", {}) or {}

    # Regexes
    scale_rx = {k: build_regexes_from_terms(v) for k, v in scale.items()}
    par_rx = {p: build_regexes_from_terms(
        terms, acronyms_extra={"CNN","BCNN","ANN","DNN","RNN","LSTM","GRU","SVM","SVR","HMM","KNN"}
    ) for p, terms in (paradigms or {}).items()}
    opt_rx = {cat: build_regexes_from_terms(terms) for cat, terms in (opt or {}).items()}

    sys_terms = []
    for t in (scale.get("system_model") or []):
        tt = re.sub(r"\bmodel(l)?ing?\b", "", str(t).lower()).strip()
        if tt:
            sys_terms.append(tt)
    system_rx = build_regexes_from_terms(sorted(set(sys_terms)))

    if w2v is not None:
        alias = build_w2v_alias_table(w2v, {
            "system_model": scale.get("system_model", []),
            "whitebox": paradigms.get("whitebox", []),
            "greybox": paradigms.get("greybox", []),
            "blackbox": paradigms.get("blackbox", []),
        }, sim_thresh=0.72, min_count=10)
        for k, v in alias.items():
            canon_map.setdefault(k, set()).add(v)

    # rows: model/optimizer hit -> host scale(s) seen in window
    rows = []
    for i, s in enumerate(sent_list):
        ctx = " ".join(sent_list[j].lower() for j in window_indices(i, n, k=context_k))

        # skip if no modelling context at all
        if not has_modelling_context(ctx):
            continue

        # collect paradigm hits (by bucket)
        par_hits = []
        for p, rx in par_rx.items():
            hits = search_terms_sentence(ctx, rx)
            if hits:
                par_hits.extend([(p, h) for h in hits])

        # collect optimization hits
        opt_hits = []
        for cat, rx in opt_rx.items():
            oh = search_terms_sentence(ctx, rx)
            if oh:
                opt_hits.extend([(cat, h) for h in oh])

        if not par_hits and not opt_hits:
            continue

        # host scales in the window (both explicit & system types)
        host_scales = set()
        if search_terms_sentence(ctx, scale_rx.get("building_model", [])): host_scales.add("building_model")
        if search_terms_sentence(ctx, scale_rx.get("system_model", [])): host_scales.add("system_model")
        if search_terms_sentence(ctx, system_rx): host_scales.add("system_model")
        if search_terms_sentence(ctx, scale_rx.get("climate_model", [])): host_scales.add("climate_model")
        if search_terms_sentence(ctx, scale_rx.get("occupancy_model", [])): host_scales.add("occupancy_model")

        # systems mentioned in window
        sys_hit_tokens = search_terms_sentence(ctx, system_rx)
        systems = sorted({canonicalize_surface(t, canon_map) or t for t in sys_hit_tokens}) if sys_hit_tokens else []

        # record paradigm rows
        for p, h in par_hits:
            rows.append({
                "paradigm_bucket": p,
                "paradigm_hit": h,
                "host_scales": ";".join(sorted(host_scales)) if host_scales else "NM",
                "systems_in_window": ";".join(systems) if systems else "NM",
                "sentence": sent_list[i].strip(),
            })
        # record optimization rows
        for cat, h in opt_hits:
            rows.append({
                "paradigm_bucket": f"[optim] {cat}",
                "paradigm_hit": h,
                "host_scales": ";".join(sorted(host_scales)) if host_scales else "NM",
                "systems_in_window": ";".join(systems) if systems else "NM",
                "sentence": sent_list[i].strip(),
            })

    return rows

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--ontology", required=True)
    ap.add_argument("--model", default="", help="Optional Word2Vec .model for alias snapping")
    ap.add_argument("--anchor_csv", default="nlp_output/anchor_first.csv")
    ap.add_argument("--model_csv",  default="nlp_output/model_first.csv")
    ap.add_argument("--excel_out",  default="nlp_output/bidirectional_results.xlsx")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--context_k", type=int, default=1, help="Sentence window size on each side")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.anchor_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.model_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.excel_out), exist_ok=True)

    with open(args.ontology, "r", encoding="utf-8") as f:
        ont = yaml.safe_load(f) or {}

    w2v = None
    if args.model and os.path.exists(args.model):
        try:
            w2v = Word2Vec.load(args.model)
        except Exception as e:
            print(f"[WARN] Could not load Word2Vec model: {e}")

    canon_map = build_canon_map(ont)

    items = list(load_jsons(args.input_dir))
    tasks = [(fname, doc, ont, canon_map, w2v, args.context_k) for fname, doc in items]

    # run in parallel
    anchor_rows = []
    model_rows = []
    if args.jobs and args.jobs > 1:
        with cf.ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futsA = [ex.submit(analyze_anchor_first, doc, ont, canon_map, w2v, args.context_k) for _, doc in items]
            futsB = [ex.submit(analyze_model_first,  doc, ont, canon_map, w2v, args.context_k) for _, doc in items]
            for (fname, _), fa, fb in tqdm(zip(items, futsA, futsB), total=len(items), desc="Analyzing"):
                try:
                    A = fa.result()
                    anchor_rows.append((fname, A))
                except Exception as e:
                    print(f"[WARN] anchor-first failed for {fname}: {e}")
                try:
                    B = fb.result()
                    for r in B:
                        r["_file"] = fname
                    model_rows.extend(B)
                except Exception as e:
                    print(f"[WARN] model-first failed for {fname}: {e}")
    else:
        for fname, doc in tqdm(items, total=len(items), desc="Analyzing"):
            try:
                A = analyze_anchor_first(doc, ont, canon_map, w2v, args.context_k)
                anchor_rows.append((fname, A))
            except Exception as e:
                print(f"[WARN] anchor-first failed for {fname}: {e}")
            try:
                B = analyze_model_first(doc, ont, canon_map, w2v, args.context_k)
                for r in B:
                    r["_file"] = fname
                model_rows.extend(B)
            except Exception as e:
                print(f"[WARN] model-first failed for {fname}: {e}")

    # ---- Build dataframes ----
    def nm_join(xs):
        xs = [x for x in (xs or []) if str(x).strip()]
        return ";".join(sorted(set(xs))) if xs else "NM"

    outA = []
    for fname, A in anchor_rows:
        outA.append({
            "file": fname,
            "systems": nm_join(A.get("systems")),
            "building_modelling_terms": nm_join(A.get("building_modelling_terms")),
            "system_modelling_terms":   nm_join(A.get("system_modelling_terms")),
            "weather_modelling_terms":  nm_join(A.get("weather_modelling_terms")),
            "occupancy_modelling_terms":nm_join(A.get("occupancy_modelling_terms")),
            "building_paradigms":       nm_join(A.get("building_paradigms")),
            "system_paradigms":         nm_join(A.get("system_paradigms")),
            "weather_paradigms":        nm_join(A.get("weather_paradigms")),
            "occupancy_paradigms":      nm_join(A.get("occupancy_paradigms")),
            "building_paradigm_terms":  nm_join(A.get("building_paradigm_terms")),
            "system_paradigm_terms":    nm_join(A.get("system_paradigm_terms")),
            "weather_paradigm_terms":   nm_join(A.get("weather_paradigm_terms")),
            "occupancy_paradigm_terms": nm_join(A.get("occupancy_paradigm_terms")),
            "optimization_methods":     nm_join(A.get("optimization_methods")),
            "optimization_method_terms":nm_join(A.get("optimization_method_terms")),
        })
    dfA = pd.DataFrame(outA).sort_values("file")

    dfB = pd.DataFrame(model_rows)
    if not dfB.empty:
        dfB = dfB.rename(columns={
            "_file":"file",
            "paradigm_bucket":"bucket_or_optimizer",
            "paradigm_hit":"term_hit",
        })
        dfB = dfB[["file","bucket_or_optimizer","term_hit","host_scales","systems_in_window","sentence"]]
        dfB = dfB.sort_values(["file","bucket_or_optimizer","term_hit"])
    else:
        dfB = pd.DataFrame(columns=["file","bucket_or_optimizer","term_hit","host_scales","systems_in_window","sentence"])

    # ---- Write outputs: two CSVs + one Excel with two sheets ----
    dfA.to_csv(args.anchor_csv, index=False)
    dfB.to_csv(args.model_csv, index=False)

    with pd.ExcelWriter(args.excel_out, engine="xlsxwriter") as xw:
        dfA.to_excel(xw, sheet_name="anchor_first", index=False)
        dfB.to_excel(xw, sheet_name="model_first", index=False)

    print(f"[OK] wrote:\n  - {args.anchor_csv}\n  - {args.model_csv}\n  - {args.excel_out}")

if __name__ == "__main__":
    main()
