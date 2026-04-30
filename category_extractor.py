#!/usr/bin/env python3
"""
category_extractor.py
─────────────────────
Category extraction using trafilatura keyword extraction,
then spaCy + GLiNER for quality filtering and NER classification.

Two functions exported to app.py:
  run_extraction(cleaned_text, raw_html) -> dict
  render_cat_results(result_dict)
"""

import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import psutil
import streamlit as st

# ── Model cache dirs ──────────────────────────────────────────────────────────

MODELS_DIR      = Path(os.environ.get("MODELS_DIR", Path(__file__).parent / "models"))
HF_CACHE_DIR    = MODELS_DIR / "huggingface"
SPACY_MODEL_ID  = "en_core_web_lg"
GLINER_MODEL_ID = "urchade/gliner_medium-v2.1"


# ── Quality filter constants ──────────────────────────────────────────────────

JUNK_STANDALONE = {
    "news","look","first","things","it","this","there","after","to","in","on",
    "at","if","the","a","an","how","why","when","what","who","where","said",
    "says","latest","update","updates","new","old","big","small","good","bad",
    "best","top","all","more","less","own","out","up","down","off","over",
}
NOISE_VERBS = {
    "say","see","tell","know","think","go","come","be","exit","push","visit",
    "want","walk","look","rescind","talk","approve","make","do","using","taking",
    "getting","giving","having","saying","knowing","thinking","finding","asking",
    "trying","leaving","following","showing","keeping","calling","working",
    "running","moving","building","writing","becoming","opening","cutting",
}
HONORIFIC_PREFIXES = frozenset({
    "dr","dr.","prof","prof.","professor","mr","mr.","mrs","mrs.","ms","ms.",
    "miss","sir","rev","rev.","gen","gen.","col","col.","lt","lt.","capt",
    "capt.","sgt","sgt.","cpl","cpl.",
})
PERSON_TITLE_ONLY = frozenset({
    "president","vice","prime","minister","senator","governor","mayor","chief",
    "justice","judge","secretary","chairman","chairwoman","director",
    "commissioner","chancellor","ambassador","consul","sheriff",
    "superintendent","commander","admiral","general","colonel","lieutenant",
    "captain","sergeant","corporal","private","representative","delegate",
    "councillor","councilor","alderman","speaker","treasurer","comptroller",
    "auditor",
})
PERSON_HEADLINE_JUNK = frozenset({
    "news","update","updates","video","interview","press","photos","photo",
    "latest","breaking","exclusive","report","statement","conference",
    "briefing","speech","remarks","announces","says","said","advertisement",
    "live","watch","read","today","roundup","recap","preview","profile",
})
PERSON_BLACKLIST = {"news","update","video","photos","advertisement","live"}
ORG_HINT_WORDS   = {
    "inc","llc","ltd","plc","corp","company","co","group","bank","university",
    "ministry","department","agency","government","council","committee","authority",
}


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class Category:
    name:        str
    score:       float     = 0.0
    source:      str       = ""       # trafilatura | spacy_ner
    entity_type: str       = "unknown"
    lq_reasons:  list[str] = field(default_factory=list)
    is_clean:    bool      = True


# ── Quality filter helpers ────────────────────────────────────────────────────

def _is_non_ascii(name: str) -> bool:
    if not name:
        return False
    return sum(1 for c in name if ord(c) > 127) / max(len(name), 1) > 0.50


def _is_verb_heavy(doc) -> bool:
    free = sum(1 for t in doc if t.pos_ in {"VERB","AUX"} and t.ent_type_ == "")
    return (free / max(len(doc), 1)) > 0.40


def _format_reasons(name: str) -> list[str]:
    r, s = [], name.strip()
    if not s:                                                   return ["empty_name"]
    if len(s) <= 2:                                             r.append("too_short")
    if re.fullmatch(r"[\W_]+", s):                              r.append("only_symbols")
    if re.fullmatch(r"\d+", s):                                 r.append("only_digits")
    if re.search(r"(.)\1{4,}", s.lower()):                      r.append("repeated_chars")
    if len(re.findall(r"[^A-Za-z0-9\s]", s)) / max(len(s),1) > 0.4:
                                                                r.append("high_symbol_ratio")
    return r


def _quality_reasons(name: str, doc) -> list[str]:
    r = _format_reasons(name)
    if _is_non_ascii(name):                                     r.append("non_english")
    if name.strip().lower() in JUNK_STANDALONE:                 r.append("junk_standalone")
    tokens = [t.text.lower() for t in doc if not t.is_punct and not t.is_space]
    if tokens and all(t in NOISE_VERBS for t in tokens):       r.append("all_verb_tokens")
    if _is_verb_heavy(doc):                                     r.append("verb_heavy")
    if re.match(r"^(how|why|when|what|who|where)\b", name.strip(), re.I):
                                                                r.append("question_prefix")
    return r


def _validate_person(raw: str) -> bool:
    tokens = raw.strip().split()
    while tokens and tokens[0].rstrip(".").lower() in HONORIFIC_PREFIXES:
        tokens = tokens[1:]
    if not tokens or len(tokens) < 2 or len(tokens) > 4:
        return False
    lo = {t.lower().rstrip(".,") for t in tokens}
    if (lo - {"the","a","an"}) <= PERSON_TITLE_ONLY:
        return False
    if lo & PERSON_HEADLINE_JUNK or lo & PERSON_BLACKLIST:
        return False
    return True


def _is_org(name: str) -> bool:
    lo = re.sub(r"[^a-z0-9 ]+", " ", name.strip().lower()).strip()
    tokens = lo.split()
    if not tokens:
        return False
    return any(t in ORG_HINT_WORDS for t in tokens) or bool(
        re.fullmatch(r"[A-Z]{2,8}", name.strip())
    )


# ── Model paths ──────────────────────────────────────────────────────────────

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
GLINER_PATH     = os.path.join(BASE_DIR, "local-models", "gliner-medium-v2.1")


# ── Model loaders — check local path first, download if missing ───────────────

@st.cache_resource
def _load_spacy():
    import spacy
    # Installed via requirements.txt wheel URL — just load directly
    for model_id in (SPACY_MODEL_ID, "en_core_web_md", "en_core_web_sm"):
        try:
            return spacy.load(model_id)
        except OSError:
            continue
    st.error(
        "No spaCy model found. Add this to requirements.txt:\n"
        "en-core-web-lg @ https://github.com/explosion/spacy-models/releases/"
        "download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl"
    )
    return None


@st.cache_resource
def _load_gliner():
    from gliner import GLiNER
    if not os.path.exists(GLINER_PATH):
        os.makedirs(GLINER_PATH, exist_ok=True)
        model = GLiNER.from_pretrained(GLINER_MODEL_ID)
        model.save_pretrained(GLINER_PATH)
    return GLiNER.from_pretrained(GLINER_PATH)


# ── Category extraction ───────────────────────────────────────────────────────

def _extract_from_trafilatura(raw_html: Optional[str], cleaned_text: str) -> list[Category]:
    """
    Uses trafilatura's built-in keyword extraction which runs TF-IDF
    on the document structure. Fast, no extra model needed.
    Falls back to simple frequency counting if raw_html is unavailable.
    """
    candidates: list[Category] = []

    if raw_html:
        try:
            # trafilatura.extract() with include_tables=True and output_format="xml"
            # exposes keywords via trafilatura.utils or directly from metadata.
            # The stable public API is trafilatura.extract() with output_format="xml"
            # then parse keywords from <keywords> tags, OR use bare_extraction().
            from trafilatura import bare_extraction
            meta = bare_extraction(raw_html, include_comments=False)
            kws  = meta.get("tags") or [] if meta else []
            if kws:
                for i, term in enumerate(kws[:30]):
                    if not isinstance(term, str) or not term.strip():
                        continue
                    candidates.append(Category(
                        name   = term.strip().title(),
                        score  = round(1.0 - i * 0.03, 3),
                        source = "trafilatura",
                    ))
                if candidates:
                    return candidates
        except Exception:
            pass

    # Fallback: simple word frequency on cleaned text
    stop = {
        "the","a","an","and","or","but","in","on","at","to","for","of","with",
        "is","are","was","were","be","been","have","has","had","will","would",
        "could","should","may","might","this","that","these","those","it","its",
        "he","she","they","we","i","you","said","says","told","also","when",
        "which","who","what","where","how","why","after","before","during",
        "since","about","from","into","through","between","against","without",
    }
    words = re.findall(r"[A-Za-z][a-z]{2,}", cleaned_text)
    freq: dict[str, int] = defaultdict(int)
    for w in words:
        if w.lower() not in stop:
            freq[w.lower()] += 1
    top = sorted(freq.items(), key=lambda x: -x[1])[:30]
    max_f = top[0][1] if top else 1
    for word, count in top:
        candidates.append(Category(
            name   = word.title(),
            score  = round(count / max_f, 3),
            source = "frequency",
        ))
    return candidates


def _extract_spacy_ner(cleaned_text: str, nlp) -> list[Category]:
    """
    Runs spaCy NER on the article. Deduplicates by lowercase,
    ranks by frequency of mention.
    """
    doc = nlp(cleaned_text[:100_000])
    label_map = {
        "PERSON": "person", "ORG": "organization",
        "GPE": "place",     "LOC": "place",
        "FAC": "place",     "NORP": "organization",
        "EVENT": "unknown", "PRODUCT": "unknown",
        "WORK_OF_ART": "unknown",
    }
    seen: dict[str, int]    = defaultdict(int)
    ent_map: dict[str, str] = {}
    for ent in doc.ents:
        if ent.label_ not in label_map:
            continue
        key = ent.text.strip().lower()
        if len(key) < 3:
            continue
        seen[key]   += 1
        ent_map[key] = label_map[ent.label_]

    top   = sorted(seen.items(), key=lambda x: -x[1])[:25]
    max_f = top[0][1] if top else 1
    return [
        Category(
            name        = key.title(),
            score       = round(count / max_f, 3),
            source      = "spacy_ner",
            entity_type = ent_map[key],
        )
        for key, count in top
    ]


def _quality_filter(categories: list[Category], nlp) -> list[Category]:
    """
    Runs spaCy quality filter on every category.
    Assigns entity_type for those without one (trafilatura/frequency sources).
    """
    from collections import Counter
    label_map = {
        "PERSON": "person", "ORG": "organization",
        "GPE": "place", "LOC": "place", "FAC": "place", "NORP": "organization",
    }
    names = [c.name for c in categories]
    docs  = list(nlp.pipe(names))
    out   = []
    for cat, doc in zip(categories, docs):
        reasons = _quality_reasons(cat.name, doc)
        if reasons:
            cat.is_clean   = False
            cat.lq_reasons = reasons
            out.append(cat)
            continue
        if cat.entity_type == "unknown":
            spacy_ents = [(e.label_, e.text) for e in doc.ents]
            if spacy_ents:
                mapped = [label_map.get(l, "unknown") for l, _ in spacy_ents]
                cat.entity_type = Counter(mapped).most_common(1)[0][0]
            elif _validate_person(cat.name):
                cat.entity_type = "person"
            elif _is_org(cat.name):
                cat.entity_type = "organization"
        out.append(cat)
    return out


def _gliner_classify(categories: list[Category], gliner, threshold: float = 0.55) -> list[Category]:
    """
    Re-classifies entity type of clean categories using GLiNER.
    Runs on all clean categories regardless of source.
    """
    clean = [c for c in categories if c.is_clean]
    if not clean or gliner is None:
        return categories
    label_map = {
        "person": "person", "organization": "organization",
        "location": "place", "city": "place", "country": "place",
    }
    names = [c.name for c in clean]
    try:
        all_ents = gliner.batch_predict_entities(
            names,
            labels=["Person","Organization","Location","City","Country",
                    "Profession","Occupation"],
            threshold=threshold,
        )
    except Exception:
        return categories
    for cat, entities in zip(clean, all_ents):
        if not entities:
            continue
        scores: dict[str, float] = defaultdict(float)
        for e in entities:
            mapped = label_map.get(e.get("label","").lower(), "unknown")
            scores[mapped] += float(e.get("score", 0))
        if scores:
            cat.entity_type = max(scores, key=scores.__getitem__)
    return categories


# ── Main extraction function (called from app.py) ─────────────────────────────

def run_extraction(cleaned_text: str, raw_html: Optional[str]) -> dict:
    """
    Full extraction pipeline:
      1. trafilatura keyword extraction (fast, no ML model)
      2. spaCy NER extraction
      3. Merge and deduplicate
      4. spaCy quality filter on all candidates
      5. GLiNER NER re-classification on clean candidates

    Returns a dict ready for render_cat_results().
    """
    proc = psutil.Process(os.getpid())
    ram0 = proc.memory_info().rss / 1024 / 1024
    t0   = time.monotonic()

    with st.spinner("Loading spaCy..."):
        nlp = _load_spacy()

    if nlp is None:
        st.error("spaCy model could not be loaded. Check requirements.txt.")
        return {"categories": [], "elapsed_s": 0, "ram_mb": 0,
                "n_clean": 0, "n_lq": 0, "n_person": 0,
                "n_org": 0, "n_place": 0, "n_unknown": 0}

    with st.spinner("Loading GLiNER..."):
        gliner = _load_gliner()

    # Step 1: extract candidates from two sources
    traf_cats  = _extract_from_trafilatura(raw_html, cleaned_text)
    spacy_cats = _extract_spacy_ner(cleaned_text, nlp)

    # Step 2: merge, deduplicate by lowercase name
    seen_names: set[str] = set()
    merged: list[Category] = []
    for cat in traf_cats + spacy_cats:
        key = cat.name.strip().lower()
        if key and key not in seen_names:
            seen_names.add(key)
            merged.append(cat)

    # Step 3: quality filter
    filtered = _quality_filter(merged, nlp)

    # Step 4: GLiNER NER classification
    classified = _gliner_classify(filtered, gliner)

    elapsed  = time.monotonic() - t0
    ram_used = proc.memory_info().rss / 1024 / 1024 - ram0

    n_clean  = sum(1 for c in classified if c.is_clean)
    n_lq     = sum(1 for c in classified if not c.is_clean)

    return {
        "categories": classified,
        "elapsed_s":  round(elapsed, 3),
        "ram_mb":     round(ram_used, 1),
        "n_clean":    n_clean,
        "n_lq":       n_lq,
        "n_person":   sum(1 for c in classified if c.is_clean and c.entity_type == "person"),
        "n_org":      sum(1 for c in classified if c.is_clean and c.entity_type == "organization"),
        "n_place":    sum(1 for c in classified if c.is_clean and c.entity_type == "place"),
        "n_unknown":  sum(1 for c in classified if c.is_clean and c.entity_type == "unknown"),
    }


# ── Rendering (called from app.py) ────────────────────────────────────────────

ENTITY_STYLE = {
    "person":       "color:#388bfd;background:#1c3a6b;border:1px solid #388bfd",
    "organization": "color:#3fb950;background:#1a3a22;border:1px solid #3fb950",
    "place":        "color:#d2a8ff;background:#2d1f5e;border:1px solid #d2a8ff",
    "unknown":      "color:#8b949e;background:#1e2128;border:1px solid #30363d",
}


def _badge(entity_type: str) -> str:
    style = ENTITY_STYLE.get(entity_type, ENTITY_STYLE["unknown"])
    return (
        f'<span style="{style};padding:2px 8px;border-radius:4px;'
        f'font-size:11px;font-weight:700;font-family:monospace;'
        f'letter-spacing:0.06em">'
        f'{entity_type.upper()}</span>'
    )


def _bar(score: float, color: str = "#58a6ff") -> str:
    pct = int(score * 100)
    return (
        f'<div style="display:flex;align-items:center;gap:8px">'
        f'<div style="flex:1;background:#21262d;border-radius:3px;height:5px">'
        f'<div style="width:{pct}%;background:{color};height:5px;border-radius:3px"></div>'
        f'</div>'
        f'<span style="font-family:monospace;font-size:11px;color:#8b949e;'
        f'min-width:34px">{score:.2f}</span></div>'
    )


def render_cat_results(result: dict):
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("Time",         f"{result['elapsed_s']}s")
    with c2: st.metric("RAM",          f"{result['ram_mb']:.0f} MB")
    with c3: st.metric("Clean",        result["n_clean"])
    with c4: st.metric("Filtered out", result["n_lq"])
    with c5: st.metric("Total found",  len(result["categories"]))

    st.markdown("---")

    col_left, col_right = st.columns(2)
    with col_left:
        st.caption("Entity type breakdown")
        import pandas as pd
        st.dataframe(
            pd.DataFrame([
                {"Type": "Person",       "Count": result["n_person"]},
                {"Type": "Organization", "Count": result["n_org"]},
                {"Type": "Place",        "Count": result["n_place"]},
                {"Type": "Unknown",      "Count": result["n_unknown"]},
            ]),
            hide_index=True,
            use_container_width=True,
        )

    clean = [c for c in result["categories"] if c.is_clean]
    dirty = [c for c in result["categories"] if not c.is_clean]

    if clean:
        st.markdown("**Extracted categories**")
        entity_color_map = {
            "person":       "#388bfd",
            "organization": "#3fb950",
            "place":        "#d2a8ff",
            "unknown":      "#8b949e",
        }
        rows = ""
        for cat in sorted(clean, key=lambda c: -c.score):
            clr  = entity_color_map.get(cat.entity_type, "#8b949e")
            rows += (
                f'<tr>'
                f'<td style="padding:7px 12px;font-size:13px;font-weight:500">{cat.name}</td>'
                f'<td style="padding:7px 12px">{_badge(cat.entity_type)}</td>'
                f'<td style="padding:7px 12px;color:#8b949e;font-size:12px;'
                f'font-family:monospace">{cat.source}</td>'
                f'<td style="padding:7px 12px;min-width:130px">{_bar(cat.score, clr)}</td>'
                f'</tr>'
            )
        st.markdown(
            f'<table style="width:100%;border-collapse:collapse;border-radius:8px;overflow:hidden">'
            f'<thead><tr style="background:#21262d">'
            f'<th style="padding:8px 12px;text-align:left;font-size:11px;'
            f'color:#8b949e;font-family:monospace;text-transform:uppercase">Category</th>'
            f'<th style="padding:8px 12px;text-align:left;font-size:11px;'
            f'color:#8b949e;font-family:monospace;text-transform:uppercase">Entity Type</th>'
            f'<th style="padding:8px 12px;text-align:left;font-size:11px;'
            f'color:#8b949e;font-family:monospace;text-transform:uppercase">Source</th>'
            f'<th style="padding:8px 12px;text-align:left;font-size:11px;'
            f'color:#8b949e;font-family:monospace;text-transform:uppercase">Score</th>'
            f'</tr></thead><tbody>{rows}</tbody></table>',
            unsafe_allow_html=True,
        )

    if dirty:
        with st.expander(f"{len(dirty)} categories removed by quality filter"):
            for cat in dirty:
                st.markdown(
                    f'<span style="font-family:monospace;font-size:12px;color:#8b949e">'
                    f'<b>{cat.name}</b> — {" | ".join(cat.lq_reasons)}</span>',
                    unsafe_allow_html=True,
                )

    if clean:
        import pandas as pd
        csv = pd.DataFrame([{
            "name":        c.name,
            "entity_type": c.entity_type,
            "score":       c.score,
            "source":      c.source,
        } for c in clean]).to_csv(index=False).encode()
        st.download_button(
            "Download categories (CSV)",
            csv,
            file_name="categories.csv",
            mime="text/csv",
        )