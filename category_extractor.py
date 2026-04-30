import os
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import psutil
import streamlit as st

# ── Model cache dirs (same convention as enhance_categories.py) ───────────────
MODELS_DIR      = Path(os.environ.get("MODELS_DIR", Path(__file__).parent / "models"))
HF_CACHE_DIR    = MODELS_DIR / "huggingface"
SPACY_CACHE_DIR = MODELS_DIR / "spacy"
KEYBERT_DIR     = MODELS_DIR / "keybert"
ZERO_SHOT_DIR   = HF_CACHE_DIR   # HuggingFace manages this itself

SPACY_MODEL_ID  = "en_core_web_md"
GLINER_MODEL_ID = "urchade/gliner_medium-v2.1"
KEYBERT_MODEL   = "all-MiniLM-L6-v2"
ZERO_SHOT_MODEL = "facebook/bart-large-mnli"

# ── Quality filter constants (identical to enhance_categories.py) ─────────────
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

# ────────────────────────────────────────────────────────────────────────────
# Data classes
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class ExtractedCategory:
    name:        str
    score:       float = 0.0          # relevance score from extractor
    entity_type: str   = "unknown"    # person | organization | place | unknown
    lq_reasons:  list[str] = field(default_factory=list)
    is_clean:    bool  = True

@dataclass
class ApproachResult:
    name:          str
    categories:    list[ExtractedCategory]
    elapsed_s:     float
    ram_delta_mb:  float
    notes:         str = ""

# ────────────────────────────────────────────────────────────────────────────
# Quality filter helpers (same logic as enhance_categories.py)
# ────────────────────────────────────────────────────────────────────────────

def _is_non_ascii(name: str) -> bool:
    if not name: return False
    return sum(1 for c in name if ord(c) > 127) / max(len(name), 1) > 0.50

def _is_verb_heavy(doc) -> bool:
    free = sum(1 for t in doc if t.pos_ in {"VERB","AUX"} and t.ent_type_ == "")
    return (free / max(len(doc), 1)) > 0.40

def _format_reasons(name: str) -> list[str]:
    r, s = [], name.strip()
    if not s:                                                  return ["empty_name"]
    if len(s) <= 2:                                            r.append("too_short")
    if re.fullmatch(r"[\W_]+", s):                             r.append("only_symbols")
    if re.fullmatch(r"\d+", s):                                r.append("only_digits")
    if re.search(r"(.)\1{4,}", s.lower()):                     r.append("repeated_chars")
    if len(re.findall(r"[^A-Za-z0-9\s]", s)) / max(len(s),1) > 0.4:
                                                               r.append("high_symbol_ratio")
    return r

def _quality_reasons(name: str, doc) -> list[str]:
    r = _format_reasons(name)
    if _is_non_ascii(name):                                    r.append("non_english")
    if name.strip().lower() in JUNK_STANDALONE:                r.append("junk_standalone")
    tokens = [t.text.lower() for t in doc if not t.is_punct and not t.is_space]
    if tokens and all(t in NOISE_VERBS for t in tokens):      r.append("all_verb_tokens")
    if _is_verb_heavy(doc):                                    r.append("verb_heavy")
    if re.match(r"^(how|why|when|what|who|where)\b", name.strip(), re.I):
                                                               r.append("question_prefix")
    return r

def _validate_person(raw: str) -> bool:
    tokens = raw.strip().split()
    while tokens and tokens[0].rstrip(".").lower() in HONORIFIC_PREFIXES:
        tokens = tokens[1:]
    if not tokens or len(tokens) < 2 or len(tokens) > 4: return False
    lo = {t.lower().rstrip(".,") for t in tokens}
    if (lo - {"the","a","an"}) <= PERSON_TITLE_ONLY:     return False
    if lo & PERSON_HEADLINE_JUNK or lo & PERSON_BLACKLIST: return False
    return True

def _is_org(name: str) -> bool:
    lo = re.sub(r"[^a-z0-9 ]+"," ", name.strip().lower()).strip()
    tokens = lo.split()
    if not tokens: return False
    return (any(t in ORG_HINT_WORDS for t in tokens) or
            bool(re.fullmatch(r"[A-Z]{2,8}", name.strip())))

def quality_filter_and_classify(categories: list[ExtractedCategory], nlp) -> list[ExtractedCategory]:
    """Run spaCy quality filter + rule-based NER on each extracted category."""
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
        # NER classify — spaCy entities first, then rule checks
        spacy_ents = [(e.label_, e.text) for e in doc.ents]
        label_map  = {"PERSON":"person","ORG":"organization",
                      "GPE":"place","LOC":"place","FAC":"place","NORP":"organization"}
        if spacy_ents:
            from collections import Counter
            mapped = [label_map.get(l,"unknown") for l, _ in spacy_ents]
            cat.entity_type = Counter(mapped).most_common(1)[0][0]
        else:
            if _validate_person(cat.name):  cat.entity_type = "person"
            elif _is_org(cat.name):         cat.entity_type = "organization"
            else:                           cat.entity_type = "unknown"
        out.append(cat)
    return out

# ────────────────────────────────────────────────────────────────────────────
# GLiNER NER override (optional — called after quality filter if model loaded)
# ────────────────────────────────────────────────────────────────────────────

def gliner_classify(categories: list[ExtractedCategory], gliner, threshold: float = 0.55):
    """Re-classify entity types using GLiNER for higher accuracy."""
    clean = [c for c in categories if c.is_clean]
    if not clean or gliner is None:
        return categories

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

    label_map = {"person":"person","organization":"organization",
                 "location":"place","city":"place","country":"place"}

    for cat, entities in zip(clean, all_ents):
        if not entities:
            continue
        scores = defaultdict(float)
        for e in entities:
            label = e.get("label","").lower()
            score = float(e.get("score", 0))
            mapped = label_map.get(label, "unknown")
            scores[mapped] += score
        if scores:
            cat.entity_type = max(scores, key=scores.__getitem__)
    return categories

# ────────────────────────────────────────────────────────────────────────────
# Model loaders — self-downloading, cached in models/
# ────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_spacy_model():
    import spacy
    SPACY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    local = SPACY_CACHE_DIR / SPACY_MODEL_ID
    if local.exists():
        return spacy.load(str(local))
    status = st.status(f"Downloading spaCy `{SPACY_MODEL_ID}`...", expanded=True)
    subprocess.run([sys.executable, "-m", "spacy", "download", SPACY_MODEL_ID], check=True)
    import spacy as _spacy
    pkg  = Path(_spacy.util.get_package_path(SPACY_MODEL_ID))
    data = next((p for p in pkg.iterdir() if (p / "config.cfg").exists()), pkg)
    shutil.copytree(str(data), str(local))
    status.update(label=f"spaCy cached", state="complete")
    return _spacy.load(str(local))


@st.cache_resource(show_spinner=False)
def load_keybert_model():
    """KeyBERT uses sentence-transformers under the hood."""
    from keybert import KeyBERT
    from sentence_transformers import SentenceTransformer
    KEYBERT_DIR.mkdir(parents=True, exist_ok=True)
    local = KEYBERT_DIR / KEYBERT_MODEL.replace("/", "_")
    if local.exists():
        st_model = SentenceTransformer(str(local))
    else:
        status = st.status(f"Downloading KeyBERT model `{KEYBERT_MODEL}`...", expanded=True)
        st_model = SentenceTransformer(KEYBERT_MODEL, cache_folder=str(KEYBERT_DIR))
        st_model.save(str(local))
        status.update(label="KeyBERT model cached", state="complete")
    return KeyBERT(st_model)


@st.cache_resource(show_spinner=False)
def load_zero_shot_model():
    """facebook/bart-large-mnli for zero-shot classification."""
    from transformers import pipeline
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"]            = str(HF_CACHE_DIR)
    os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE_DIR)
    cached = HF_CACHE_DIR / "hub" / ("models--" + ZERO_SHOT_MODEL.replace("/","--"))
    if cached.exists():
        return pipeline("zero-shot-classification", model=ZERO_SHOT_MODEL)
    status = st.status(f"Downloading zero-shot model `{ZERO_SHOT_MODEL}`...", expanded=True)
    status.write("This is ~1.6 GB, one-time download...")
    pipe = pipeline("zero-shot-classification", model=ZERO_SHOT_MODEL)
    status.update(label="Zero-shot model cached", state="complete")
    return pipe


@st.cache_resource(show_spinner=False)
def load_gliner_model():
    from gliner import GLiNER
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"]            = str(HF_CACHE_DIR)
    os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE_DIR)
    cached = HF_CACHE_DIR / "hub" / ("models--" + GLINER_MODEL_ID.replace("/","--"))
    if cached.exists():
        return GLiNER.from_pretrained(GLINER_MODEL_ID)
    status = st.status(f"Downloading GLiNER `{GLINER_MODEL_ID}`...", expanded=True)
    model = GLiNER.from_pretrained(GLINER_MODEL_ID)
    status.update(label="GLiNER cached", state="complete")
    return model

# ────────────────────────────────────────────────────────────────────────────
# Approach 1 — KeyBERT
# ────────────────────────────────────────────────────────────────────────────

def run_keybert(
    text: str, nlp, gliner, top_n: int = 15, use_gliner: bool = False
) -> ApproachResult:
    """
    KeyBERT uses sentence-transformer embeddings to find keyphrases
    that are semantically most representative of the document.

    Unlike TF-IDF it understands meaning — a football article will surface
    'striker', 'goal', 'match' even if 'sports' never appears.

    Returns multi-word phrases (ngrams 1-3) scored by cosine similarity
    to the document embedding.
    """
    proc = psutil.Process(os.getpid())
    ram0 = proc.memory_info().rss / 1024 / 1024
    t0   = time.monotonic()

    kw_model = load_keybert_model()

    # Extract keyphrases — ngram range (1,3) catches both single words
    # and multi-word phrases like "Premier League" or "climate change"
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 3),
        stop_words="english",
        use_maxsum=True,        # MaxSum diversity — avoids near-duplicate phrases
        nr_candidates=30,
        top_n=top_n,
    )

    raw_cats = [
        ExtractedCategory(name=kw.title(), score=round(score, 3))
        for kw, score in keywords
        if len(kw.strip()) > 2
    ]

    cats    = quality_filter_and_classify(raw_cats, nlp)
    if use_gliner and gliner:
        cats = gliner_classify(cats, gliner)

    elapsed = time.monotonic() - t0
    ram_d   = proc.memory_info().rss / 1024 / 1024 - ram0

    return ApproachResult(
        name="KeyBERT",
        categories=cats,
        elapsed_s=round(elapsed, 3),
        ram_delta_mb=round(ram_d, 1),
        notes="Semantic keyphrases via sentence-transformer embeddings. "
              "Finds topics even when exact words are absent.",
    )

# ────────────────────────────────────────────────────────────────────────────
# Approach 2 — spaCy NER
# ────────────────────────────────────────────────────────────────────────────

def run_spacy_ner(text: str, nlp) -> ApproachResult:
    """
    Extracts named entities from the article text using spaCy's built-in NER.

    Good for: people, organisations, places, events that are explicitly named.
    Weak for: abstract topics like 'sports', 'economy' — spaCy won't find those.

    Entity types extracted: PERSON, ORG, GPE, LOC, FAC, EVENT, PRODUCT, WORK_OF_ART
    Deduplicates by lowercase text so "Google" and "google" count once.
    """
    proc = psutil.Process(os.getpid())
    ram0 = proc.memory_info().rss / 1024 / 1024
    t0   = time.monotonic()

    doc = nlp(text[:100_000])   # cap at 100k chars to avoid memory issues

    label_map = {
        "PERSON": "person", "ORG": "organization",
        "GPE": "place", "LOC": "place", "FAC": "place",
        "NORP": "organization", "EVENT": "unknown",
        "PRODUCT": "unknown", "WORK_OF_ART": "unknown",
    }

    # Deduplicate by lowercase, keep highest-frequency
    seen: dict[str, int] = defaultdict(int)
    ent_types: dict[str, str] = {}
    for ent in doc.ents:
        if ent.label_ not in label_map:
            continue
        key = ent.text.strip().lower()
        if len(key) < 3:
            continue
        seen[key]      += 1
        ent_types[key]  = label_map[ent.label_]

    # Sort by frequency, take top 20
    top = sorted(seen.items(), key=lambda x: -x[1])[:20]

    raw_cats = [
        ExtractedCategory(
            name       = text_key.title(),
            score      = round(count / max(seen.values(), 1), 3),
            entity_type= ent_types[text_key],
        )
        for text_key, count in top
    ]

    # Quality filter only — entity type already set above
    cats = quality_filter_and_classify(raw_cats, nlp)

    elapsed = time.monotonic() - t0
    ram_d   = proc.memory_info().rss / 1024 / 1024 - ram0

    return ApproachResult(
        name="spaCy NER",
        categories=cats,
        elapsed_s=round(elapsed, 3),
        ram_delta_mb=round(ram_d, 1),
        notes="Extracts explicitly named entities. Fast but misses abstract topics.",
    )

# ────────────────────────────────────────────────────────────────────────────
# Approach 3 — Zero-shot classification (BART-MNLI)
# ────────────────────────────────────────────────────────────────────────────

CANDIDATE_CATEGORIES = [
    # Sports
    "Football", "Cricket", "Tennis", "Basketball", "Sports", "Olympics",
    # Politics
    "Politics", "Elections", "Government", "Parliament", "Policy",
    # Business
    "Business", "Economy", "Finance", "Stock Market", "Startup", "Technology",
    # Entertainment
    "Bollywood", "Hollywood", "Movies", "Music", "Television", "Celebrity",
    # Science
    "Science", "Space", "Climate Change", "Health", "Medicine", "Research",
    # World
    "International Relations", "War", "Diplomacy", "United Nations",
    # Social
    "Education", "Culture", "Religion", "Society", "Human Rights",
    # Crime
    "Crime", "Court", "Law", "Corruption",
]

def run_zero_shot(
    text: str, nlp, top_n: int = 15, custom_labels: Optional[list[str]] = None
) -> ApproachResult:
    """
    Uses facebook/bart-large-mnli (Natural Language Inference) to decide
    whether this article 'entails' each candidate category label.

    This is the most powerful approach for abstract topics:
    - A football article scores high for 'Sports' and 'Football'
    - Even if those exact words are absent from the text

    Limitation: you must provide candidate labels up front.
    The model picks from your list — it won't invent new categories.
    """
    proc = psutil.Process(os.getpid())
    ram0 = proc.memory_info().rss / 1024 / 1024
    t0   = time.monotonic()

    pipe   = load_zero_shot_model()
    labels = custom_labels or CANDIDATE_CATEGORIES

    # Truncate text to avoid token limit (BART max is 1024 tokens)
    text_trunc = text[:3000]

    result = pipe(text_trunc, candidate_labels=labels, multi_label=True)

    # result["labels"] and result["scores"] are parallel lists
    raw_cats = []
    for label, score in zip(result["labels"][:top_n], result["scores"][:top_n]):
        if score < 0.05:
            continue
        raw_cats.append(ExtractedCategory(name=label, score=round(score, 3)))

    cats    = quality_filter_and_classify(raw_cats, nlp)
    elapsed = time.monotonic() - t0
    ram_d   = proc.memory_info().rss / 1024 / 1024 - ram0

    return ApproachResult(
        name="Zero-Shot (BART-MNLI)",
        categories=cats,
        elapsed_s=round(elapsed, 3),
        ram_delta_mb=round(ram_d, 1),
        notes="NLI model decides if article 'implies' each candidate label. "
              "Best for abstract topics. Requires candidate list.",
    )

# ────────────────────────────────────────────────────────────────────────────
# Approach 4 — KeyBERT + GLiNER combined
# ────────────────────────────────────────────────────────────────────────────

def run_keybert_gliner(text: str, nlp, top_n: int = 15) -> ApproachResult:
    """
    Two-stage pipeline:
      Stage 1: KeyBERT extracts semantically relevant keyphrases
      Stage 2: GLiNER classifies each phrase as person/org/place/unknown

    Better than KeyBERT alone because GLiNER's zero-shot NER handles
    rare entities and phrases that spaCy's rule-based NER would miss.
    Better than GLiNER alone because KeyBERT provides better candidate
    phrases than running GLiNER directly on raw article text.
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
        nr_candidates=30,
        top_n=top_n,
    )
    raw_cats = [
        ExtractedCategory(name=kw.title(), score=round(score, 3))
        for kw, score in keywords if len(kw.strip()) > 2
    ]

    cats = quality_filter_and_classify(raw_cats, nlp)
    cats = gliner_classify(cats, gliner)

    elapsed = time.monotonic() - t0
    ram_d   = proc.memory_info().rss / 1024 / 1024 - ram0

    return ApproachResult(
        name="KeyBERT + GLiNER",
        categories=cats,
        elapsed_s=round(elapsed, 3),
        ram_delta_mb=round(ram_d, 1),
        notes="KeyBERT extracts semantic phrases, GLiNER classifies entity type. "
              "Best accuracy, highest resource cost.",
    )

# ────────────────────────────────────────────────────────────────────────────
# UI rendering
# ────────────────────────────────────────────────────────────────────────────

ENTITY_COLORS = {
    "person":       ("#388bfd", "#1c3a6b"),
    "organization": ("#3fb950", "#1a3a22"),
    "place":        ("#d2a8ff", "#2d1f5e"),
    "unknown":      ("#8b949e", "#1e2128"),
    "low_quality":  ("#f85149", "#3a1a1a"),
}

def render_entity_badge(entity_type: str) -> str:
    color, bg = ENTITY_COLORS.get(entity_type, ENTITY_COLORS["unknown"])
    icons = {"person":"👤","organization":"🏢","place":"📍",
             "unknown":"❓","low_quality":"⚠"}
    icon  = icons.get(entity_type, "❓")
    label = entity_type.replace("_"," ").title()
    return (f'<span style="background:{bg};color:{color};border:1px solid {color};'
            f'padding:2px 8px;border-radius:4px;font-size:11px;font-weight:700;'
            f'font-family:monospace;text-transform:uppercase;letter-spacing:0.06em">'
            f'{icon} {label}</span>')

def render_score_bar(score: float, color: str = "#58a6ff") -> str:
    pct = int(score * 100)
    return (f'<div style="display:flex;align-items:center;gap:8px">'
            f'<div style="flex:1;background:#21262d;border-radius:3px;height:6px">'
            f'<div style="width:{pct}%;background:{color};height:6px;border-radius:3px"></div>'
            f'</div>'
            f'<span style="font-family:monospace;font-size:11px;color:#8b949e;min-width:36px">'
            f'{score:.2f}</span></div>')

def render_approach_results(result: ApproachResult):
    """Render one approach's results as an expandable card."""
    n_clean = sum(1 for c in result.categories if c.is_clean)
    n_lq    = sum(1 for c in result.categories if not c.is_clean)

    # Header metrics row
    c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 1])
    with c1:
        st.markdown(f"**{result.name}**")
        st.caption(result.notes)
    with c2:
        st.metric("Time", f"{result.elapsed_s}s")
    with c3:
        st.metric("RAM Δ", f"{result.ram_delta_mb:.0f} MB")
    with c4:
        st.metric("Clean", n_clean)
    with c5:
        st.metric("Filtered", n_lq)

    # Category grid
    if not result.categories:
        st.markdown(
            '<div style="color:#8b949e;font-size:13px;padding:8px">No categories extracted</div>',
            unsafe_allow_html=True
        )
        return

    # Split clean vs low-quality
    clean = [c for c in result.categories if c.is_clean]
    dirty = [c for c in result.categories if not c.is_clean]

    if clean:
        # Build HTML table for clean categories
        rows_html = ""
        for cat in clean:
            badge    = render_entity_badge(cat.entity_type)
            bar      = render_score_bar(cat.score,
                           ENTITY_COLORS.get(cat.entity_type, ("",""))[0] or "#58a6ff")
            rows_html += f"""
            <tr>
              <td style="padding:7px 12px;font-weight:500">{cat.name}</td>
              <td style="padding:7px 12px">{badge}</td>
              <td style="padding:7px 12px;min-width:140px">{bar}</td>
            </tr>"""

        st.markdown(f"""
        <table style="width:100%;border-collapse:collapse;
                      background:#161b22;border-radius:8px;overflow:hidden">
          <thead>
            <tr style="background:#21262d">
              <th style="padding:8px 12px;text-align:left;font-size:11px;
                         color:#8b949e;font-family:monospace;text-transform:uppercase">
                Category</th>
              <th style="padding:8px 12px;text-align:left;font-size:11px;
                         color:#8b949e;font-family:monospace;text-transform:uppercase">
                Entity Type</th>
              <th style="padding:8px 12px;text-align:left;font-size:11px;
                         color:#8b949e;font-family:monospace;text-transform:uppercase">
                Relevance</th>
            </tr>
          </thead>
          <tbody>{rows_html}</tbody>
        </table>
        """, unsafe_allow_html=True)

    if dirty:
        with st.expander(f"⚠ {len(dirty)} low-quality categories filtered out"):
            for cat in dirty:
                reasons = " · ".join(cat.lq_reasons)
                st.markdown(
                    f'<span style="color:#8b949e;font-size:12px;font-family:monospace">'
                    f'**{cat.name}** — {reasons}</span>',
                    unsafe_allow_html=True
                )

# ────────────────────────────────────────────────────────────────────────────
# Benchmark comparison table
# ────────────────────────────────────────────────────────────────────────────

def render_benchmark_table(results: list[ApproachResult]):
    import pandas as pd

    rows = []
    for r in results:
        n_clean = sum(1 for c in r.categories if c.is_clean)
        n_person = sum(1 for c in r.categories if c.is_clean and c.entity_type == "person")
        n_org    = sum(1 for c in r.categories if c.is_clean and c.entity_type == "organization")
        n_place  = sum(1 for c in r.categories if c.is_clean and c.entity_type == "place")
        n_unk    = sum(1 for c in r.categories if c.is_clean and c.entity_type == "unknown")
        n_lq     = sum(1 for c in r.categories if not c.is_clean)
        rows.append({
            "Approach":       r.name,
            "Time (s)":       r.elapsed_s,
            "RAM Δ (MB)":     r.ram_delta_mb,
            "Clean cats":     n_clean,
            "👤 Person":      n_person,
            "🏢 Org":         n_org,
            "📍 Place":       n_place,
            "❓ Unknown":     n_unk,
            "⚠ Filtered":    n_lq,
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True, use_container_width=True)

    # Fastest approach
    if results:
        fastest = min(results, key=lambda r: r.elapsed_s)
        most    = max(results, key=lambda r: sum(1 for c in r.categories if c.is_clean))
        st.markdown(
            f'<div style="background:#1c2d40;border-left:3px solid #58a6ff;'
            f'border-radius:0 6px 6px 0;padding:10px 14px;font-size:13px;margin-top:8px">'
            f'⚡ <b>Fastest:</b> {fastest.name} ({fastest.elapsed_s}s) &nbsp;·&nbsp; '
            f'📦 <b>Most categories:</b> {most.name} '
            f'({sum(1 for c in most.categories if c.is_clean)} clean)'
            f'</div>',
            unsafe_allow_html=True
        )

# ────────────────────────────────────────────────────────────────────────────
# Main render function — called from main app or standalone
# ────────────────────────────────────────────────────────────────────────────

def render_category_extractor(article_text: Optional[str] = None):
    """
    Main entry point.
    article_text: pre-extracted article body (passed from main app).
                  If None, shows a text area for manual input.
    """
    st.markdown("## 🏷️ Category Extraction & NER Tagging")
    st.markdown(
        "Extract what topics/categories an article belongs to — "
        "even when the exact category words don't appear in the text. "
        "Four approaches benchmarked side-by-side."
    )

    # ── Approach selector ─────────────────────────────────────────────────────
    with st.expander("⚙ Approach Settings", expanded=True):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Select approaches to run:**")
            run_keybert    = st.checkbox("KeyBERT (semantic keyphrases)",      value=True)
            run_spacy      = st.checkbox("spaCy NER (named entities)",         value=True)
            run_zeroshot   = st.checkbox("Zero-Shot BART (abstract topics)",   value=False,
                help="Needs ~1.6 GB download on first run")
            run_kb_gliner  = st.checkbox("KeyBERT + GLiNER (combined)",       value=False,
                help="Needs GLiNER (~1.5 GB) — most accurate, slowest")

        with col_b:
            top_n        = st.slider("Max categories per approach", 5, 30, 15)
            use_gliner_ner = st.checkbox(
                "Use GLiNER for NER classification (all approaches)",
                value=False,
                help="Overrides spaCy NER with GLiNER for all approaches that support it"
            )

            st.markdown("**Zero-shot candidate categories:**")
            custom_labels_txt = st.text_area(
                "One per line (leave blank for defaults)",
                placeholder="Football\nCricket\nPolitics\nTechnology\n...",
                height=100,
            )
            custom_labels = (
                [l.strip() for l in custom_labels_txt.strip().splitlines() if l.strip()]
                if custom_labels_txt.strip() else None
            )

    # ── Text input ────────────────────────────────────────────────────────────
    if article_text:
        text = article_text
        st.success(f"Using article from URL — {len(text.split()):,} words")
    else:
        text = st.text_area(
            "Or paste article text directly:",
            height=200,
            placeholder="Paste article content here if not using a URL above...",
        )

    if not text or not text.strip():
        st.info("Extract an article using the URL field above, or paste text here.")
        return

    run_any = run_keybert or run_spacy or run_zeroshot or run_kb_gliner
    if not run_any:
        st.warning("Select at least one approach.")
        return

    if not st.button("▶ Extract Categories", use_container_width=True, type="primary"):
        return

    # ── Load base models ──────────────────────────────────────────────────────
    with st.spinner("Loading spaCy quality model..."):
        nlp = load_spacy_model()

    gliner = None
    if use_gliner_ner or run_kb_gliner:
        with st.spinner("Loading GLiNER..."):
            gliner = load_gliner_model()

    # ── Run approaches ────────────────────────────────────────────────────────
    all_results: list[ApproachResult] = []

    if run_keybert:
        with st.spinner("Running KeyBERT..."):
            r = run_keybert(text, nlp, gliner, top_n=top_n, use_gliner=use_gliner_ner)
            all_results.append(r)

    if run_spacy:
        with st.spinner("Running spaCy NER..."):
            r = run_spacy_ner(text, nlp)
            all_results.append(r)

    if run_zeroshot:
        with st.spinner("Running Zero-Shot BART (this takes ~10-30s)..."):
            r = run_zero_shot(text, nlp, top_n=top_n, custom_labels=custom_labels)
            all_results.append(r)

    if run_kb_gliner:
        with st.spinner("Running KeyBERT + GLiNER..."):
            r = run_keybert_gliner(text, nlp, top_n=top_n)
            all_results.append(r)

    if not all_results:
        return

    # ── Benchmark comparison ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Benchmark Comparison")
    render_benchmark_table(all_results)

    # ── Per-approach results ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📋 Results by Approach")

    for result in all_results:
        with st.expander(
            f"**{result.name}** — {sum(1 for c in result.categories if c.is_clean)} clean categories"
            f" · {result.elapsed_s}s",
            expanded=True,
        ):
            render_approach_results(result)

    # ── Download all results ──────────────────────────────────────────────────
    st.markdown("---")
    import pandas as pd, io
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
        df  = pd.DataFrame(all_rows)
        csv = df.to_csv(index=False).encode()
        st.download_button(
            "⬇ Download all extracted categories (CSV)",
            csv,
            file_name="extracted_categories.csv",
            mime="text/csv",
        )


# ── Standalone mode ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    st.set_page_config(
        page_title="Category Extractor",
        page_icon="🏷️",
        layout="wide",
    )
    st.title("🏷️ Category Extraction — Standalone")
    render_category_extractor()