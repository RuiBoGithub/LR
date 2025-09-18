#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, yaml, argparse
from dataclasses import dataclass, field
from collections import defaultdict
import concurrent.futures as cf

import nltk
from gensim.models import Word2Vec
import pandas as pd
from tqdm import tqdm
import numpy as np

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

# ---------------- Config (no section weights; all sections count equally) ----------------
DEFAULT_EXP_TOPN = 12
DEFAULT_EXP_MIN_COUNT = 20
DEFAULT_KEEP_Z = 0.5
DEFAULT_CENTROID_MARGIN = 0.05
DEFAULT_ALLOWED_POS = {"NN", "NNS", "NNP", "NNPS"}  # nouns

@dataclass
class Config:
    dump_sections: bool = False

    # expansion knobs
    do_not_expand: set = field(default_factory=set)  # e.g., {"model_paradigm"}
    exp_topn: int = DEFAULT_EXP_TOPN
    exp_thresh: float = 0.70          # used only if dynamic_z=False
    exp_min_count: int = DEFAULT_EXP_MIN_COUNT
    use_dynamic_z: bool = True
    keep_z: float = DEFAULT_KEEP_Z

    # centroid guard across buckets
    use_centroid_guard: bool = True
    centroid_margin: float = DEFAULT_CENTROID_MARGIN

    # token filtering
    allowed_pos: set = field(default_factory=lambda: set(DEFAULT_ALLOWED_POS))
    banned_terms: set = field(default_factory=set)

    # anchors
    enable_model_anchors: bool = True
    rc_anchor_window: int = 80

# ---------------- Section parsing (robust for headings like "1 Introduction", etc.) ----------------
INTRO_REVIEW_INLINE_RX = re.compile(
    r"""
    (?im)
    (^|\n)\s*
    (?:section \s+\d+|[ivxlcdm]+\.?|\d+(?:\.\d+)*|\d+\))?
    \s*[:\-\)\.]?\s*
    ([A-Za-z\-]+(?:\s+[A-Za-z\-]+)?)      # 1–2 words
    \b
    """,
    re.VERBOSE,
)

def _is_intro_or_review_two_word_max(hdr: str) -> bool:
    h = (hdr or "").strip().lower()
    toks = h.split()
    return (1 <= len(toks) <= 2) and ("intro" in h or "review" in h)

def force_intro_review_headings_to_own_lines(text: str) -> str:
    def _repl(m: re.Match):
        heading_words = m.group(2) or ""
        if not _is_intro_or_review_two_word_max(heading_words):
            return m.group(0)
        s = m.group(0)
        cut = m.end(2) - m.start(0)
        return s[:cut] + "\n" + s[cut:]
    return INTRO_REVIEW_INLINE_RX.sub(_repl, text or "")

HEADING_RX = re.compile(
    r"""
    (?im)^\s*
    (?:section\s+\d+|[ivxlcdm]+\.?|\d+(?:\.\d+)*|\d+\))?
    \s*[:\-\)\.]?\s*
    ([A-Za-z][^\n]{0,100})
    \s*$
    """,
    re.VERBOSE,
)

def normalize_section_key(raw_header: str) -> str:
    h = (raw_header or "").strip().lower()
    h = re.sub(r'[:\-\—]+$', '', h).strip()
    if h.startswith('abstract') or 'abstract' in h: return 'abstract'
    if 'methodolog' in h: return 'methodology'
    if 'method' in h: return 'method'
    if 'material' in h: return 'materials'
    if 'data' in h: return 'data'
    if 'experiment' in h: return 'experiment'
    if 'implement' in h: return 'implementation'
    if 'result' in h: return 'results'
    if 'evaluat' in h or 'validation' in h: return 'evaluation'
    if 'related work' in h or 'related-work' in h or 'review' in h or 'literature' in h: return 'literature'
    if 'background' in h: return 'background'
    if 'intro' in h or h == 'introduction': return 'introduction'
    if 'discuss' in h: return 'discussion'
    if 'conclusion' in h: return 'conclusion'
    if 'future work' in h or 'future direction' in h or h == 'future': return 'future'
    if 'appendix' in h: return 'appendix'
    return h.split()[0] if h else 'unknown'

def split_sections(original_text: str):
    text = (original_text or "").replace("\r", "\n")
    text = force_intro_review_headings_to_own_lines(text)
    out = {}
    matches = list(HEADING_RX.finditer(text))
    if not matches:
        return {"preamble": text}
    out["preamble"] = text[:matches[0].start()]
    for i, m in enumerate(matches):
        raw_header = (m.group(1) or "").strip()
        if len(raw_header.split()) > 12 or re.search(r"[.!?]{2,}", raw_header):
            continue
        key = normalize_section_key(raw_header)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out[key] = text[start:end]
    return out

def iter_sections(sections: dict, cfg: Config):
    # No exclusions; all sections are yielded
    for sec, text in (sections or {}).items():
        yield sec, text

# ---------------- Helpers ----------------
def sent_tokenize_safe(text: str):
    try:
        from nltk.tokenize import sent_tokenize
        return sent_tokenize(text or "")
    except Exception:
        return re.split(r'(?<=[\.\?\!])\s+', text or "")

def to_str(x):
    if x is None: return ""
    if isinstance(x, (list, tuple, set)): return ";".join(map(str, x))
    if isinstance(x, dict): return ";".join(f"{k}:{v:.2f}" for k, v in x.items())
    return str(x)

def looks_like_citation_hit(term: str, snippet: str) -> bool:
    rx = re.compile(rf"\b{re.escape(term)}\b\s*\(\d{{4}}\)", re.I)
    return rx.search(snippet or "") is not None

# ------- robust token variants (no expansion; just surface robustness) -------
def token_variants(term: str):
    t = (term or "").strip().lower()
    if not t: return [t]
    variants = {t}
    v = t.replace(" – ", " ").replace(" — ", " ").replace("–", " ").replace("—", " ").replace("-", " ")
    variants |= {v, v.replace("  ", " ").strip(), v.replace(" ", "-"), v.replace(" ", "")}
    return sorted(x for x in variants if x)

def compile_term_regex(term: str):
    alts = [re.escape(x) for x in token_variants(term)]
    pat = r"\b(?:%s)\b" % "|".join(alts)
    return re.compile(pat, re.I)

def find_terms(text, terms, max_hits=5):
    hits = []
    for t in terms:
        rx = compile_term_regex(t)
        m = rx.search(text or "")
        if m:
            s = max(0, m.start()-60)
            e = m.end()+60
            hits.append((t, (text or "")[s:e]))
            if len(hits) >= max_hits:
                break
    return hits

# ---------------- Word2Vec expansion ----------------
def is_bad_token(term: str, cfg: Config) -> bool:
    t = (term or "").strip().lower()
    if not t: return True
    if t in cfg.banned_terms: return True
    if len(t) < 3: return True
    if re.search(r"\d", t): return True
    return False

def is_good_pos(term: str, cfg: Config) -> bool:
    try:
        tok = term.split()[-1]
        pos = nltk.pos_tag([tok])[0][1]
        return pos in cfg.allowed_pos
    except Exception:
        return True

def term_to_w2v_token(term: str) -> str:
    return term.replace(' ', '_')

def centroid(model: Word2Vec, terms):
    vecs = []
    for t in terms or []:
        w = term_to_w2v_token(t)
        if w in model.wv:
            vecs.append(model.wv[w])
    if not vecs: return None
    return np.mean(np.vstack(vecs), axis=0)

def cosine(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-12
    return float(np.dot(a, b) / denom)

def build_bucket_centroids(model: Word2Vec, ont: dict):
    cents = {}
    for cat, subtree in (ont or {}).items():
        if cat in ("blacklist", "whitelist"): continue
        seeds = []
        if isinstance(subtree, list):
            seeds = subtree
        elif isinstance(subtree, dict):
            for _, terms in subtree.items():
                if isinstance(terms, list):
                    seeds.extend(terms)
        c = centroid(model, seeds)
        if c is not None:
            cents[cat] = c
    return cents

def dynamic_z_keep(sim_list, keep_z: float):
    if not sim_list: return -1.0
    mu = float(np.mean(sim_list))
    sd = float(np.std(sim_list)) or 1e-6
    return mu + keep_z * sd

def expand_terms(model: Word2Vec, terms, target_cat: str, cfg: Config, bucket_centroids=None):
    terms = [t for t in terms if isinstance(t, str) and t.strip()]
    out = set()
    for t in terms:
        if not is_bad_token(t, cfg) and is_good_pos(t, cfg):
            out.add(t.lower())
    for t in terms:
        w = term_to_w2v_token(t)
        if w not in model.wv: continue
        cutoff = dynamic_z_keep([sim for _, sim in model.wv.most_similar(w, topn=50)], cfg.keep_z) if cfg.use_dynamic_z else cfg.exp_thresh
        for cand, sim in model.wv.most_similar(w, topn=cfg.exp_topn):
            if sim < cutoff: continue
            try:
                if model.wv.get_vecattr(cand, "count") < cfg.exp_min_count:
                    continue
            except Exception:
                pass
            cand_str = cand.replace('_', ' ').lower()
            if is_bad_token(cand_str, cfg) or not is_good_pos(cand_str, cfg): continue
            if cfg.use_centroid_guard and bucket_centroids and target_cat in bucket_centroids:
                v = model.wv[cand]
                sims = {cat: cosine(v, c) for cat, c in bucket_centroids.items()}
                ordered = sorted(sims.items(), key=lambda kv: kv[1], reverse=True)
                if ordered:
                    best_cat, best = ordered[0]
                    second = ordered[1][1] if len(ordered) > 1 else -1.0
                    if not (best_cat == target_cat and (best - second) >= cfg.centroid_margin):
                        continue
            out.add(cand_str)
    return sorted(out)

def expand_ontology(model: Word2Vec, ont: dict, cfg: Config):
    expanded = {}
    if isinstance(ont.get("blacklist", None), list):
        cfg.banned_terms |= set(map(str.lower, ont["blacklist"]))
    bucket_centroids = build_bucket_centroids(model, ont) if cfg.use_centroid_guard else None
    for cat, subtree in ont.items():
        if cat in ("blacklist", "whitelist"): continue
        if isinstance(subtree, dict):
            expanded[cat] = {}
            for key, terms in subtree.items():
                if not isinstance(terms, list):
                    expanded[cat][key] = terms
                    continue
                if cat in cfg.do_not_expand:
                    expanded[cat][key] = sorted({t.lower() for t in terms if not is_bad_token(t, cfg) and is_good_pos(t, cfg)})
                else:
                    expanded[cat][key] = expand_terms(model, terms, target_cat=cat, cfg=cfg, bucket_centroids=bucket_centroids)
        elif isinstance(subtree, list):
            if cat in cfg.do_not_expand:
                expanded[cat] = sorted({t.lower() for t in subtree if not is_bad_token(t, cfg) and is_good_pos(t, cfg)})
            else:
                expanded[cat] = expand_terms(model, subtree, target_cat=cat, cfg=cfg, bucket_centroids=bucket_centroids)
        else:
            expanded[cat] = subtree
    return expanded

# ---------------- Anchors & labelers ----------------
MODEL_ANCHORS_RX = [compile_term_regex("model"), compile_term_regex("modelling"), compile_term_regex("modeling")]

def anchored_rc_hit(sent: str, window: int = 80) -> bool:
    s = (sent or "").lower()
    rc_like = re.search(r"\b(first[- ]?order\s+)?r\.?c\.?\b", s) or re.search(r"\brc\s+(network|model)\b", s)
    if not rc_like:
        return False
    pos = rc_like.start()
    for rx in MODEL_ANCHORS_RX:
        m = rx.search(s)
        if m and abs(m.start() - pos) <= window:
            return True
    return False

def label_paradigms_by_scale(sections, ont, cfg: Config):
    """
    Per-scale detection of paradigms by sentence-level co-occurrence with scale terms.
    Returns per-scale paradigm winners AND exact term hits per scale.
    """
    scale_map = ont.get("scale", {}) or {}
    paradigms_map = ont.get("model_paradigm", {}) or {}

    def rx_list(terms):
        return [compile_term_regex(str(t)) for t in (terms or [])]

    scale_rx = {s: rx_list(terms) for s, terms in scale_map.items()}
    paradigm_term_rx = {p: [(t, compile_term_regex(str(t))) for t in (terms or [])]
                        for p, terms in paradigms_map.items()}

    out = {}
    for s in scale_map.keys():
        out[s] = {
            "present": False,
            "paradigms": [],
            "paradigm_scores": {p: 0.0 for p in paradigms_map.keys()},
            "paradigm_term_hits": {},   # term -> count (float for consistency)
            "evidence": defaultdict(list),
        }

    for sec, text in iter_sections(sections, cfg):
        for sent in sent_tokenize_safe(text or ""):
            sl = sent.lower()
            scales_here = [s for s, rxs in scale_rx.items() if any(rx.search(sl) for rx in rxs)]
            if not scales_here:
                continue

            for p, term_rxs in paradigm_term_rx.items():
                matched_terms = []
                for t, rx in term_rxs:
                    if rx.search(sl) and not looks_like_citation_hit(t, sent):
                        matched_terms.append(t)

                # Optional RC-anchor bonus: if greybox bucket includes rc terms but text says “first-order RC … model”
                if cfg.enable_model_anchors and p.lower().startswith("grey"):
                    if anchored_rc_hit(sent, cfg.rc_anchor_window):
                        matched_terms.append("rc(model-anchored)")

                if not matched_terms:
                    continue

                for s in scales_here:
                    out[s]["present"] = True
                    out[s]["paradigm_scores"][p] += 1.0   # <-- no section weights; equal contribution
                    out[s]["evidence"][p].append((sec, p, sent.strip()))
                    for t in matched_terms:
                        out[s]["paradigm_term_hits"][t] = out[s]["paradigm_term_hits"].get(t, 0.0) + 1.0
                        out[s]["evidence"][t].append((sec, t, sent.strip()))

    # winners per scale (share>=0.28, else top-1)
    for s, rec in out.items():
        ps = rec["paradigm_scores"]; total = sum(ps.values()) or 1.0
        norm = {k: v / total for k, v in ps.items()}
        winners = [k for k, v in norm.items() if v >= 0.28]
        if not winners and total > 0 and max(ps.values()) > 0:
            winners = [max(ps, key=ps.get)]
        rec["paradigms"] = winners

    return out

def paradigms_from_per_scale(per_scale):
    bag = defaultdict(float)
    for s, rec in (per_scale or {}).items():
        for p in rec.get("paradigms") or []:
            bag[p] += 1.0
    if not bag:
        return []
    total = sum(bag.values())
    share = {k: v/total for k, v in bag.items()}
    winners = [k for k, v in share.items() if v >= 0.28]
    if not winners:
        winners = [max(bag, key=bag.get)]
    return winners

def label_models_multi_global(sections, model_paradigm_terms: dict):
    """
    Global bag of exact term hits (for CSV transparency). No section weights, no scale anchor.
    Returns: model_hits(term->count), evidence(term->snippets)
    """
    model_hits, evidence = {}, {}
    for sec, text in iter_sections(sections, Config()):
        t = (text or "").lower()
        for paradigm, terms in (model_paradigm_terms or {}).items():
            for term in (terms or []):
                rx = compile_term_regex(term)
                for m in rx.finditer(t):
                    snip = t[max(0, m.start()-120): m.end()+120]
                    if looks_like_citation_hit(term, snip):
                        continue
                    model_hits[term] = model_hits.get(term, 0.0) + 1.0
                    evidence.setdefault(term, []).append((sec, term, snip))
    return model_hits, evidence

def label_scale(sections, ont):
    hits = set(); ev = defaultdict(list)
    for sec, text in iter_sections(sections, Config()):
        t = (text or "").lower()
        for scale, terms in ont.get("scale", {}).items():
            ft = find_terms(t, terms, max_hits=3)
            if ft:
                hits.add(scale); ev[scale].extend([(sec,)*1 + h for h in ft])
    return sorted(hits), ev

def label_data_types(sections, ont):
    found = set(); ev = defaultdict(list)
    for sec, text in iter_sections(sections, Config()):
        t = (text or "").lower()
        for dtype, terms in ont.get("data_types", {}).items():
            ft = find_terms(t, terms, max_hits=3)
            if ft:
                found.add(dtype); ev[dtype].extend([(sec,)*1 + h for h in ft])
    return sorted(found), ev

def label_optimization_by_scale(sections, ont):
    scale_map = ont.get("scale", {}) or {}
    meth_map  = ont.get("optimization_methods", {}) or {}
    obj_map   = ont.get("optimization_objectives", {}) or {}

    def rx_list(terms):
        return [compile_term_regex(str(t)) for t in (terms or [])]

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

    for sec, text in iter_sections(sections, Config()):
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
                    out[s]["methods_scores"][m] += 1.0
                    out[s]["evidence"][m].append((sec, m, sent.strip()))
                for o in obj_here:
                    out[s]["objectives_scores"][o] += 1.0
                    out[s]["evidence"][o].append((sec, o, sent.strip()))

    for s, rec in out.items():
        if rec["methods_scores"]:
            rec["methods"] = [k for k,_ in sorted(rec["methods_scores"].items(), key=lambda kv: kv[1], reverse=True)]
        if rec["objectives_scores"]:
            rec["objectives"] = [k for k,_ in sorted(rec["objectives_scores"].items(), key=lambda kv: kv[1], reverse=True)]
    return out

# ---------------- Core paper analysis ----------------
def analyze_paper(doc, ont_expanded, cfg):
    raw = doc['full-text-retrieval-response']['originalText']
    sections = split_sections(raw)

    if getattr(cfg, "dump_sections", False):
        import sys
        print("[sections]", sorted(set(sections.keys())), file=sys.stderr)

    # Per-scale paradigm detection (anchored to scale, sentence-level)
    per_scale = label_paradigms_by_scale(sections, ont_expanded, cfg)
    paradigms = paradigms_from_per_scale(per_scale)

    # Global bag of paradigm terms (for audit)
    model_hits, model_ev = label_models_multi_global(sections, ont_expanded.get("model_paradigm", {}))

    # Other optional labels if you keep them in YAML
    scales, scale_ev = label_scale(sections, ont_expanded)
    data_types, data_ev = label_data_types(sections, ont_expanded)
    opt_per_scale = label_optimization_by_scale(sections, ont_expanded)

    return {
        "paradigms": paradigms,
        "model_hits": model_hits,
        "model_evidence": model_ev,
        "scale": scales,
        "scale_evidence": scale_ev,
        "data_types": data_types,
        "data_evidence": data_ev,
        "per_scale": per_scale,
        "optimization_per_scale": opt_per_scale,
    }

# ---------------- IO & parallel driver ----------------
def load_jsons(dirpath):
    for fn in os.listdir(dirpath):
        if fn.endswith(".json"):
            p = os.path.join(dirpath, fn)
            with open(p, "r", encoding="utf-8") as f:
                yield fn, json.load(f)

def worker(args):
    fname, doc, ont_expanded, cfg = args
    res = analyze_paper(doc, ont_expanded, cfg)
    return fname, res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Folder with Elsevier JSONs")
    ap.add_argument("--model", default="word2vec_model.model", help="Path to trained gensim Word2Vec model")
    ap.add_argument("--ontology", default="ontology.yaml", help="Seed ontology YAML/YML")
    ap.add_argument("--output_csv", default="classified_papers.csv", help="Output CSV path")
    ap.add_argument("--output_json", default="", help="Optional evidence JSON")
    ap.add_argument("--jobs", type=int, default=8, help="Parallel workers")

    # controls
    ap.add_argument("--dump_sections", action="store_true", help="Print parsed section keys per file to stderr for debugging.")
    ap.add_argument("--no_expand_model_paradigm", action="store_true", help="Do not expand the model_paradigm bucket (keep curated).")
    ap.add_argument("--no_expand_all", action="store_true", help="Disable expansion for ALL ontology buckets.")

    ap.add_argument("--exp_topn", type=int, default=DEFAULT_EXP_TOPN, help="Top-N neighbors per seed for expansion.")
    ap.add_argument("--exp_thresh", type=float, default=0.70, help="Cosine similarity threshold if dynamic z is off.")
    ap.add_argument("--exp_min_count", type=int, default=DEFAULT_EXP_MIN_COUNT, help="Minimum word frequency in Word2Vec vocab.")
    ap.add_argument("--no_dynamic_z", action="store_true", help="Disable dynamic z-score thresholding.")
    ap.add_argument("--keep_z", type=float, default=DEFAULT_KEEP_Z, help="z-score threshold (how many std dev above mean).")
    ap.add_argument("--no_centroid_guard", action="store_true", help="Disable centroid/margin guard across buckets.")
    ap.add_argument("--centroid_margin", type=float, default=DEFAULT_CENTROID_MARGIN, help="Margin for centroid guard.")
    ap.add_argument("--banned_terms", default="", help="Comma-separated blacklist terms to always drop during expansion/matching.")
    ap.add_argument("--allowed_pos", default="NN,NNS,NNP,NNPS", help="Comma-separated POS tags to keep (default nouns).")

    args = ap.parse_args()

    banned = set(t.strip().lower() for t in (args.banned_terms or "").split(",") if t.strip())
    allowed_pos = set(p.strip().upper() for p in (args.allowed_pos or "").split(",") if p.strip())

    # Config
    do_not_expand = set()
    if args.no_expand_model_paradigm:
        do_not_expand.add("model_paradigm")
    if args.no_expand_all:
        do_not_expand |= set()  # we’ll compute after reading YAML

    cfg = Config(
        do_not_expand=do_not_expand,
        exp_topn=args.exp_topn,
        exp_thresh=args.exp_thresh,
        exp_min_count=args.exp_min_count,
        use_dynamic_z=(not args.no_dynamic_z),
        keep_z=args.keep_z,
        use_centroid_guard=(not args.no_centroid_guard),
        centroid_margin=args.centroid_margin,
        allowed_pos=allowed_pos if allowed_pos else set(DEFAULT_ALLOWED_POS),
        banned_terms=banned,
    )
    cfg.dump_sections = bool(args.dump_sections)

    # Load ontology
    with open(args.ontology, "r", encoding="utf-8") as f:
        ont = yaml.safe_load(f) or {}

    # If --no_expand_all, apply to all top-level buckets in YAML (except lists like blacklist/whitelist)
    if args.no_expand_all:
        cfg.do_not_expand |= set([k for k in ont.keys() if k not in ("blacklist", "whitelist")])

    # Load model once (parent) to expand ontology
    w2v_parent = Word2Vec.load(args.model)
    ont_expanded = expand_ontology(w2v_parent, ont, cfg)

    items = list(load_jsons(args.input_dir))
    tasks = [(fname, doc, ont_expanded, cfg) for (fname, doc) in items]

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
        return (join_or_nm(rec.get("methods")), join_or_nm(rec.get("objectives")))

    for fname, r in rows:
        ps = r.get("per_scale", {})
        bm_p, bm_terms = ps_vals(ps, "building_model")
        sm_p, sm_terms = ps_vals(ps, "system_model")
        om_p, om_terms = ps_vals(ps, "occupancy_model")
        cm_p, cm_terms = ps_vals(ps, "climate_model")
        opt_ps = r.get("optimization_per_scale", {})
        bm_meth, bm_obj = opt_vals(opt_ps, "building_model")
        sm_meth, sm_obj = opt_vals(opt_ps, "system_model")

        out_rows.append({
            "file": fname,
            "paradigms": nm_if_empty(r["paradigms"]),
            "paradigm_terms_found": nm_if_empty(r["model_hits"]),
            "scale": nm_if_empty(r["scale"]),
            "data_types": nm_if_empty(r["data_types"]),
            # Per-scale (paradigms + matched terms)
            "building_model_paradigms": bm_p,
            "building_model_paradigm_terms": bm_terms,
            "system_model_paradigms": sm_p,
            "system_model_paradigm_terms": sm_terms,
            "occupancy_model_paradigms": om_p,
            "occupancy_model_paradigm_terms": om_terms,
            "climate_model_paradigms": cm_p,
            "climate_model_paradigm_terms": cm_terms,
            # optimisation (optional)
            "building_model_optim_methods": bm_meth,
            "building_model_optim_objectives": bm_obj,
            "system_model_optim_methods": sm_meth,
            "system_model_optim_objectives": sm_obj,
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
