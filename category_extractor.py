#!/usr/bin/env python3
"""
category_extractor.py
─────────────────────
Category extraction and NER tagging for news articles.

Three approaches:
  1. KeyBERT     — semantic keyphrase extraction via sentence-transformers
  2. spaCy NER   — named entity extraction using spaCy en_core_web_lg
  3. GLiNER NER  — zero-shot entity classification on KeyBERT candidates

spaCy quality filter runs on every extracted category regardless of approach.
Models are loaded from the models/ folder; downloaded on first run if absent.
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
import trafilatura

# ── Model cache dirs ──────────────────────────────────────────────────────────
MODELS_DIR      = Path(os.environ.get("MODELS_DIR", Path(__file__).parent / "models"))
HF_CACHE_DIR    = MODELS_DIR / "huggingface"
KEYBERT_DIR     = MODELS_DIR / "keybert"

SPACY_MODEL_ID  = "en_core_web_lg"
GLINER_MODEL_ID = "urchade/gliner_medium-v2.1"
KEYBERT_MODEL   = "all-MiniLM-L6-v2"


# ── Article fetching ──────────────────────────────────────────────────────────

def fetch_and_extract(url: str) -> Optional[str]:
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return None
    raw = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=False,
        favor_precision=True,
        deduplicate=True,
    )
    if not raw:
        return None
    lines = [l.strip() for l in raw.split("\n") if len(l.strip()) >= 40]
    return " ".join(lines) if lines else None


def clean_text(text: str) -> str:
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


# ── Quality filter constants (same as enhance_categories.py) ─────────────────

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
ORG_HINT_WORDS = {
    "inc","llc","ltd","plc","corp","company","co","group","bank","university",
    "ministry","department","agency","government","council","committee","authority",
}


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ExtractedCategory:
    name:        str
    score:       float     = 0.0
    entity_type: str       = "unknown"
    lq_reasons:  list[str] = field(default_factory=list)
    is_clean:    bool      = True


@dataclass
class ApproachResult:
    name:         str
    categories:   list[ExtractedCategory]
    elapsed_s:    float
    ram_delta_mb: float
    notes:        str = ""


# ── Quality filter helpers ────────────────────────────────────────────────────

def _is_non_ascii(name: str) -> bool:
    if not name:
        return False
    return sum(1 for c in name if ord(c) > 127) / max(len(name), 1) > 0.50


def _is_verb_heavy(doc) -> bool:
    free = sum(1 for t in doc if t.pos_ in {"VERB", "AUX"} and t.ent_type_ == "")
    return (free / max(len(doc), 1)) > 0.40


def _format_reasons(name: str) -> list[str]:
    r, s = [], name.strip()
    if not s:
        return ["empty_name"]
    if len(s) <= 2:
        r.append("too_short")
    if re.fullmatch(r"[\W_]+", s):
        r.append("only_symbols")
    if re.fullmatch(r"\d+", s):
        r.append("only_digits")
    if re.search(r"(.)\1{4,}", s.lower()):
        r.append("repeated_chars")
    if len(re.findall(r"[^A-Za-z0-9\s]", s)) / max(len(s), 1) > 0.4:
        r.append("high_symbol_ratio")
    return r


def _quality_reasons(name: str, doc) -> list[str]:
    r = _format_reasons(name)
    if _is_non_ascii(name):
        r.append("non_english")
    if name.strip().lower() in JUNK_STANDALONE:
        r.append("junk_standalone")
    tokens = [t.text.lower() for t in doc if not t.is_punct and not t.is_space]
    if tokens and all(t in NOISE_VERBS for t in tokens):
        r.append("all_verb_tokens")
    if _is_verb_heavy(doc):
        r.append("verb_heavy")
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
    if (lo - {"the", "a", "an"}) <= PERSON_TITLE_ONLY:
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


def quality_filter_and_classify(
    categories: list[ExtractedCategory], nlp
) -> list[ExtractedCategory]:
    from collections import Counter
    names = [c.name for c in categories]
    docs  = list(nlp.pipe(names))
    out   = []
    label_map = {
        "PERSON": "person", "ORG": "organization",
        "GPE": "place", "LOC": "place", "FAC": "place", "NORP": "organization",
    }
    for cat, doc in zip(categories, docs):
        reasons = _quality_reasons(cat.name, doc)
        if reasons:
            cat.is_clean   = False
            cat.lq_reasons = reasons
            out.append(cat)
            continue
        spacy_ents = [(e.label_, e.text) for e in doc.ents]
        if spacy_ents:
            mapped = [label_map.get(l, "unknown") for l, _ in spacy_ents]
            cat.entity_type = Counter(mapped).most_common(1)[0][0]
        else:
            if _validate_person(cat.name):
                cat.entity_type = "person"
            elif _is_org(cat.name):
                cat.entity_type = "organization"
            else:
                cat.entity_type = "unknown"
        out.append(cat)
    return out


def gliner_classify(
    categories: list[ExtractedCategory], gliner, threshold: float = 0.55
) -> list[ExtractedCategory]:
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
            labels=["Person", "Organization", "Location", "City", "Country",
                    "Profession", "Occupation"],
            threshold=threshold,
        )
    except Exception:
        return categories
    for cat, entities in zip(clean, all_ents):
        if not entities:
            continue
        scores: dict[str, float] = defaultdict(float)
        for e in entities:
            mapped = label_map.get(e.get("label", "").lower(), "unknown")
            scores[mapped] += float(e.get("score", 0))
        if scores:
            cat.entity_type = max(scores, key=scores.__getitem__)
    return categories


# ── Model loaders ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_spacy_model():
    import spacy
    try:
        return spacy.load(SPACY_MODEL_ID)
    except OSError:
        try:
            return spacy.load("en_core_web_md")
        except OSError:
            st.error(
                "spaCy model not found. Add to requirements.txt:\n"
                "en-core-web-lg @ https://github.com/explosion/spacy-models/"
                "releases/download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl"
            )
            st.stop()


@st.cache_resource(show_spinner=False)
def load_keybert_model():
    from keybert import KeyBERT
    from sentence_transformers import SentenceTransformer
    KEYBERT_DIR.mkdir(parents=True, exist_ok=True)
    local = KEYBERT_DIR / KEYBERT_MODEL.replace("/", "_")
    if local.exists():
        st_model = SentenceTransformer(str(local))
    else:
        status = st.status(f"Downloading KeyBERT model...", expanded=True)
        st_model = SentenceTransformer(KEYBERT_MODEL, cache_folder=str(KEYBERT_DIR))
        st_model.save(str(local))
        status.update(label="KeyBERT model cached", state="complete")
    return KeyBERT(st_model)


@st.cache_resource(show_spinner=False)
def load_gliner_model():
    from gliner import GLiNER
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"]            = str(HF_CACHE_DIR)
    os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE_DIR)
    cached = HF_CACHE_DIR / "hub" / ("models--" + GLINER_MODEL_ID.replace("/", "--"))
    if cached.exists():
        return GLiNER.from_pretrained(GLINER_MODEL_ID)
    status = st.status(f"Downloading GLiNER...", expanded=True)
    model  = GLiNER.from_pretrained(GLINER_MODEL_ID)
    status.update(label="GLiNER cached", state="complete")
    return model


# ── Extraction approaches ─────────────────────────────────────────────────────

def extract_keybert(
    text: str, nlp, top_n: int = 15
) -> ApproachResult:
    """
    Uses sentence-transformer embeddings to find keyphrases most
    representative of the document. Understands meaning — finds 'striker'
    and 'Premier League' in a football article even if 'sport' never appears.
    use_maxsum=True ensures diversity — avoids near-duplicate phrases.
    """
    proc = psutil.Process(os.getpid())
    ram0 = proc.memory_info().rss / 1024 / 1024
    t0   = time.monotonic()

    kw_model = load_keybert_model()
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 3),
        stop_words="english",
        use_maxsum=True,
        nr_candidates=40,
        top_n=top_n,
    )
    raw_cats = [
        ExtractedCategory(name=kw.title(), score=round(score, 3))
        for kw, score in keywords
        if len(kw.strip()) > 2
    ]
    cats = quality_filter_and_classify(raw_cats, nlp)

    return ApproachResult(
        name="KeyBERT",
        categories=cats,
        elapsed_s=round(time.monotonic() - t0, 3),
        ram_delta_mb=round(proc.memory_info().rss / 1024 / 1024 - ram0, 1),
        notes="Semantic keyphrases via sentence-transformer embeddings.",
    )


def extract_spacy_ner(text: str, nlp) -> ApproachResult:
    """
    Runs spaCy NER on the full article. Finds explicitly named entities:
    people, organisations, places, events. Deduplicates by lowercase and
    ranks by frequency (entities mentioned more get higher scores).
    """
    proc = psutil.Process(os.getpid())
    ram0 = proc.memory_info().rss / 1024 / 1024
    t0   = time.monotonic()

    doc = nlp(text[:100_000])
    label_map = {
        "PERSON": "person", "ORG": "organization",
        "GPE": "place", "LOC": "place", "FAC": "place", "NORP": "organization",
        "EVENT": "unknown", "PRODUCT": "unknown", "WORK_OF_ART": "unknown",
    }
    seen: dict[str, int]   = defaultdict(int)
    ent_types: dict[str, str] = {}
    for ent in doc.ents:
        if ent.label_ not in label_map:
            continue
        key = ent.text.strip().lower()
        if len(key) < 3:
            continue
        seen[key]     += 1
        ent_types[key] = label_map[ent.label_]

    top = sorted(seen.items(), key=lambda x: -x[1])[:25]
    max_count = top[0][1] if top else 1
    raw_cats  = [
        ExtractedCategory(
            name=key.title(),
            score=round(count / max_count, 3),
            entity_type=ent_types[key],
        )
        for key, count in top
    ]
    cats = quality_filter_and_classify(raw_cats, nlp)

    return ApproachResult(
        name="spaCy NER",
        categories=cats,
        elapsed_s=round(time.monotonic() - t0, 3),
        ram_delta_mb=round(proc.memory_info().rss / 1024 / 1024 - ram0, 1),
        notes="Named entity extraction. Fast, finds explicitly mentioned entities only.",
    )


def extract_keybert_gliner(text: str, nlp, top_n: int = 15) -> ApproachResult:
    """
    Two-stage pipeline:
      Stage 1 — KeyBERT extracts semantically relevant keyphrases.
      Stage 2 — GLiNER classifies each phrase's entity type with
                 higher accuracy than spaCy rule-based NER, handling
                 rare and domain-specific entities.
    """
    proc = psutil.Process(os.getpid())
    ram0 = proc.memory_info().rss / 1024 / 1024
    t0   = time.monotonic()

    kw_model = load_keybert_model()
    gliner   = load_gliner_model()

    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 3),
        stop_words="english",
        use_maxsum=True,
        nr_candidates=40,
        top_n=top_n,
    )
    raw_cats = [
        ExtractedCategory(name=kw.title(), score=round(score, 3))
        for kw, score in keywords
        if len(kw.strip()) > 2
    ]
    cats = quality_filter_and_classify(raw_cats, nlp)
    cats = gliner_classify(cats, gliner)

    return ApproachResult(
        name="KeyBERT + GLiNER",
        categories=cats,
        elapsed_s=round(time.monotonic() - t0, 3),
        ram_delta_mb=round(proc.memory_info().rss / 1024 / 1024 - ram0, 1),
        notes="KeyBERT phrases re-classified by GLiNER. Best accuracy, highest cost.",
    )


# ── UI rendering ──────────────────────────────────────────────────────────────

ENTITY_STYLE = {
    "person":       ("color:#388bfd;background:#1c3a6b;border:1px solid #388bfd"),
    "organization": ("color:#3fb950;background:#1a3a22;border:1px solid #3fb950"),
    "place":        ("color:#d2a8ff;background:#2d1f5e;border:1px solid #d2a8ff"),
    "unknown":      ("color:#8b949e;background:#1e2128;border:1px solid #30363d"),
    "low_quality":  ("color:#f85149;background:#3a1a1a;border:1px solid #f85149"),
}


def _badge(entity_type: str) -> str:
    style = ENTITY_STYLE.get(entity_type, ENTITY_STYLE["unknown"])
    label = entity_type.replace("_", " ").upper()
    return (
        f'<span style="{style};padding:2px 8px;border-radius:4px;'
        f'font-size:11px;font-weight:700;font-family:monospace;'
        f'letter-spacing:0.06em">{label}</span>'
    )


def _score_bar(score: float, color: str = "#58a6ff") -> str:
    pct = int(score * 100)
    return (
        f'<div style="display:flex;align-items:center;gap:8px">'
        f'<div style="flex:1;background:#21262d;border-radius:3px;height:5px">'
        f'<div style="width:{pct}%;background:{color};height:5px;border-radius:3px"></div>'
        f'</div>'
        f'<span style="font-family:monospace;font-size:11px;color:#8b949e;'
        f'min-width:34px">{score:.2f}</span></div>'
    )


def _render_approach(result: ApproachResult):
    n_clean = sum(1 for c in result.categories if c.is_clean)
    n_lq    = sum(1 for c in result.categories if not c.is_clean)

    c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 1])
    with c1:
        st.markdown(f"**{result.name}**")
        st.caption(result.notes)
    with c2:
        st.metric("Time", f"{result.elapsed_s}s")
    with c3:
        st.metric("RAM", f"{result.ram_delta_mb:.0f} MB")
    with c4:
        st.metric("Clean", n_clean)
    with c5:
        st.metric("Filtered", n_lq)

    clean = [c for c in result.categories if c.is_clean]
    dirty = [c for c in result.categories if not c.is_clean]

    if clean:
        rows = ""
        for cat in clean:
            clr = ENTITY_STYLE.get(cat.entity_type, ENTITY_STYLE["unknown"]).split(";")[0].split(":")[1]
            rows += (
                f'<tr>'
                f'<td style="padding:7px 12px;font-size:13px;font-weight:500">{cat.name}</td>'
                f'<td style="padding:7px 12px">{_badge(cat.entity_type)}</td>'
                f'<td style="padding:7px 12px;min-width:130px">{_score_bar(cat.score, clr)}</td>'
                f'</tr>'
            )
        st.markdown(
            f'<table style="width:100%;border-collapse:collapse;'
            f'border-radius:8px;overflow:hidden">'
            f'<thead><tr style="background:#21262d">'
            f'<th style="padding:8px 12px;text-align:left;font-size:11px;'
            f'color:#8b949e;font-family:monospace;text-transform:uppercase">Category</th>'
            f'<th style="padding:8px 12px;text-align:left;font-size:11px;'
            f'color:#8b949e;font-family:monospace;text-transform:uppercase">Entity Type</th>'
            f'<th style="padding:8px 12px;text-align:left;font-size:11px;'
            f'color:#8b949e;font-family:monospace;text-transform:uppercase">Relevance</th>'
            f'</tr></thead><tbody>{rows}</tbody></table>',
            unsafe_allow_html=True,
        )

    if dirty:
        with st.expander(f"{len(dirty)} categories filtered by quality check"):
            for cat in dirty:
                st.markdown(
                    f'<span style="font-family:monospace;font-size:12px;color:#8b949e">'
                    f'<b>{cat.name}</b> — {" | ".join(cat.lq_reasons)}</span>',
                    unsafe_allow_html=True,
                )


def _render_benchmark(results: list[ApproachResult]):
    import pandas as pd
    rows = []
    for r in results:
        rows.append({
            "Approach":    r.name,
            "Time (s)":    r.elapsed_s,
            "RAM (MB)":    r.ram_delta_mb,
            "Clean":       sum(1 for c in r.categories if c.is_clean),
            "Person":      sum(1 for c in r.categories if c.is_clean and c.entity_type == "person"),
            "Org":         sum(1 for c in r.categories if c.is_clean and c.entity_type == "organization"),
            "Place":       sum(1 for c in r.categories if c.is_clean and c.entity_type == "place"),
            "Unknown":     sum(1 for c in r.categories if c.is_clean and c.entity_type == "unknown"),
            "Filtered":    sum(1 for c in r.categories if not c.is_clean),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    if len(results) > 1:
        fastest = min(results, key=lambda r: r.elapsed_s)
        most    = max(results, key=lambda r: sum(1 for c in r.categories if c.is_clean))
        st.caption(
            f"Fastest: {fastest.name} ({fastest.elapsed_s}s)  |  "
            f"Most categories: {most.name} "
            f"({sum(1 for c in most.categories if c.is_clean)} clean)"
        )


# ── Main render function ──────────────────────────────────────────────────────

def render_category_extractor(
    article_text: Optional[str] = None,
    url: Optional[str] = None,
):
    st.markdown("### Category Extraction and NER Tagging")

    with st.expander("Settings", expanded=True):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Approaches**")
            use_keybert   = st.checkbox("KeyBERT (semantic keyphrases)", value=True)
            use_spacy_ner = st.checkbox("spaCy NER (named entities)",    value=True)
            use_kb_gliner = st.checkbox(
                "KeyBERT + GLiNER (combined)",
                value=False,
                help="Downloads GLiNER (~1.5 GB) on first run",
            )
        with col_b:
            top_n = st.slider("Max categories per approach", 5, 30, 15)
            gliner_thresh = st.slider(
                "GLiNER confidence threshold", 0.3, 0.9, 0.55, 0.05,
                help="Only applies to KeyBERT + GLiNER approach",
            )

    # Resolve article text
    text = article_text

    if not text and url:
        with st.spinner("Fetching article for category extraction..."):
            raw = fetch_and_extract(url)
        if raw:
            text = clean_text(raw)
            st.session_state["cat_article_text"] = text
            st.session_state["cat_article_url"]  = url
        else:
            st.warning("Could not fetch article from that URL.")
            return
    elif not text and "cat_article_text" in st.session_state:
        text = st.session_state["cat_article_text"]

    if not text:
        st.info("Paste a URL in the sidebar to extract categories.")
        return

    st.caption(f"Article: {len(text.split()):,} words")

    if not (use_keybert or use_spacy_ner or use_kb_gliner):
        st.warning("Select at least one approach.")
        return

    if not st.button("Extract Categories", use_container_width=True, type="primary"):
        return

    with st.spinner("Loading spaCy model..."):
        nlp = load_spacy_model()

    all_results: list[ApproachResult] = []

    if use_keybert:
        with st.spinner("Running KeyBERT..."):
            all_results.append(extract_keybert(text, nlp, top_n=top_n))

    if use_spacy_ner:
        with st.spinner("Running spaCy NER..."):
            all_results.append(extract_spacy_ner(text, nlp))

    if use_kb_gliner:
        with st.spinner("Running KeyBERT + GLiNER..."):
            all_results.append(extract_keybert_gliner(text, nlp, top_n=top_n))

    if not all_results:
        return

    st.markdown("---")
    st.markdown("**Benchmark comparison**")
    _render_benchmark(all_results)

    st.markdown("---")
    st.markdown("**Results by approach**")
    for result in all_results:
        n_clean = sum(1 for c in result.categories if c.is_clean)
        with st.expander(f"{result.name} — {n_clean} categories — {result.elapsed_s}s", expanded=True):
            _render_approach(result)

    # Download
    import pandas as pd
    all_rows = []
    for r in all_results:
        for cat in r.categories:
            all_rows.append({
                "approach":    r.name,
                "name":        cat.name,
                "entity_type": cat.entity_type,
                "score":       cat.score,
                "is_clean":    cat.is_clean,
                "lq_reasons":  " | ".join(cat.lq_reasons),
            })
    if all_rows:
        csv = pd.DataFrame(all_rows).to_csv(index=False).encode()
        st.download_button(
            "Download extracted categories (CSV)",
            csv,
            file_name="extracted_categories.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    st.set_page_config(page_title="Category Extractor", layout="wide")
    st.title("Category Extraction — Standalone")
    render_category_extractor()