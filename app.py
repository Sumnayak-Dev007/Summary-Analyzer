import re
import os
import time
import nltk
import psutil
import streamlit as st
import trafilatura
import spacy
import pytextrank

st.set_page_config(
    page_title="Article Analyzer",
    page_icon="",
    layout="wide"
)

nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
if nltk_data_path not in nltk.data.path:
    nltk.data.path.append(nltk_data_path)

for resource, path in [
    ("punkt",                         "tokenizers/punkt"),
    ("punkt_tab",                     "tokenizers/punkt_tab"),
    ("averaged_perceptron_tagger",    "taggers/averaged_perceptron_tagger"),
    ("averaged_perceptron_tagger_eng","taggers/averaged_perceptron_tagger_eng"),
]:
    try:
        nltk.data.find(path)
    except (LookupError, OSError):
        nltk.download(resource, download_dir=nltk_data_path)

NOISE_PATTERNS = re.compile(
    r"(click here|follow us|subscribe|telegram|whatsapp|"
    r"read more|advertisement|also read|watch video|"
    r"watch live|breaking news|sign up|newsletter|"
    r"cookie|privacy policy|terms of use|all rights reserved|"
    r"share this|related articles|trending now|"
    r"download app|follow on|connect with us|"
    r"ifsc code|pin code finder|emi calculator|"
    r"petrol price|diesel price|gold price|silver price|"
    r"loan calculator|bmi calculator|age calculator|aqi|"
    r"home loan|personal loan|car loan|education loan|"
    r"senior copy editor|copy editor|contributing writer|"
    r"for any tips and queries|reach out to|master's degree|"
    r"@abpnetwork|@gmail|@yahoo)",
    re.IGNORECASE
)

BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
SPACY_LG_PATH    = os.path.join(BASE_DIR, "local-models", "en_core_web_lg")


# ── Model loading ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_spacy_lg():
    # en_core_web_lg has better NER and word vectors than sm/md
    # installed via requirements.txt wheel URL
    try:
        nlp = spacy.load("en_core_web_lg")
        nlp.add_pipe("textrank")
        return nlp
    except OSError:
        try:
            nlp = spacy.load("en_core_web_md")
            nlp.add_pipe("textrank")
            return nlp
        except OSError:
            st.error(
                "spaCy model not found. Add to requirements.txt:\n"
                "en-core-web-lg @ https://github.com/explosion/spacy-models/"
                "releases/download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl"
            )
            st.stop()


# ── Article fetching ──────────────────────────────────────────────────────────

def fetch_and_extract(url: str) -> str | None:
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return None
    raw_text = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=False,
        favor_precision=True,      # reduces boilerplate vs recall tradeoff
        deduplicate=True,          # removes duplicate paragraphs
    )
    if not raw_text:
        return None
    lines = []
    for line in raw_text.split("\n"):
        line = line.strip()
        if len(line) < 40:
            continue
        if NOISE_PATTERNS.search(line):
            continue
        lines.append(line)
    return " ".join(lines) if lines else None


def clean_text(text: str) -> str:
    text = NOISE_PATTERNS.sub("", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


# ── TextRank summarization (tuned) ────────────────────────────────────────────

def textrank_summarize(
    text: str,
    nlp,
    n_sentences: int = 5,
    min_sentence_len: int = 40,
    focus_phrases: list[str] | None = None,
) -> dict:
    """
    TextRank via spaCy + PyTextRank.

    TextRank builds a graph where sentences are nodes and edges are
    weighted by how many important phrases they share. Sentences that
    share many high-ranked phrases get high scores.

    Tuning applied here vs default:
    - min_sentence_len filters fragments before scoring
    - focus_phrases biases the graph toward sentences containing
      user-specified terms (e.g. article subject, key person name)
    - Sentences returned in original article order (not score order)
      so the summary reads naturally
    - Overlapping/near-duplicate sentences are deduplicated by
      checking token overlap ratio before adding to output

    Returns a dict with summary text + per-sentence scores for display.
    """
    proc     = psutil.Process(os.getpid())
    ram_before = proc.memory_info().rss / 1024 / 1024
    t0       = time.monotonic()

    # spaCy processes up to 100k chars to avoid memory issues on long articles
    doc = nlp(text[:100_000])

    # Collect all sentences with their textrank scores
    # doc._.textrank.calc_textgraph() already ran via the pipe
    sent_scores: dict[str, float] = {}
    for phrase in doc._.phrases:
        for sent in doc.sents:
            if phrase.text.lower() in sent.text.lower():
                sent_text = sent.text.strip()
                sent_scores[sent_text] = sent_scores.get(sent_text, 0) + phrase.rank

    # Boost sentences containing focus phrases if provided
    if focus_phrases:
        for sent_text, score in sent_scores.items():
            for fp in focus_phrases:
                if fp.lower() in sent_text.lower():
                    sent_scores[sent_text] = score * 1.5

    # Filter by minimum length
    sent_scores = {
        s: sc for s, sc in sent_scores.items()
        if len(s) >= min_sentence_len
    }

    if not sent_scores:
        # Fallback: return first n sentences if textrank found nothing
        fallback = [
            s.text.strip() for s in doc.sents
            if len(s.text.strip()) >= min_sentence_len
        ][:n_sentences]
        return {
            "summary":    " ".join(fallback),
            "sentences":  [(s, 0.0) for s in fallback],
            "elapsed_s":  round(time.monotonic() - t0, 3),
            "ram_mb":     round(proc.memory_info().rss / 1024 / 1024 - ram_before, 1),
            "method":     "fallback (first-n)",
        }

    # Pick top-n by score
    top_sents = sorted(sent_scores.items(), key=lambda x: -x[1])[:n_sentences * 2]

    # Deduplicate by token overlap — don't include near-duplicate sentences
    def _overlap(a: str, b: str) -> float:
        ta = set(a.lower().split())
        tb = set(b.lower().split())
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / min(len(ta), len(tb))

    selected: list[tuple[str, float]] = []
    for sent_text, score in top_sents:
        if len(selected) >= n_sentences:
            break
        if any(_overlap(sent_text, s) > 0.6 for s, _ in selected):
            continue
        selected.append((sent_text, score))

    # Restore original article order
    all_sents_ordered = [s.text.strip() for s in doc.sents]
    selected_ordered  = sorted(
        selected,
        key=lambda x: all_sents_ordered.index(x[0])
        if x[0] in all_sents_ordered else 9999
    )

    summary = " ".join(s for s, _ in selected_ordered)
    elapsed = time.monotonic() - t0
    ram_d   = proc.memory_info().rss / 1024 / 1024 - ram_before

    return {
        "summary":   summary,
        "sentences": selected_ordered,
        "elapsed_s": round(elapsed, 3),
        "ram_mb":    round(ram_d, 1),
        "method":    "textrank",
    }


def measure(func, *args):
    process    = psutil.Process(os.getpid())
    ram_before = process.memory_info().rss / 1024 / 1024
    start      = time.time()
    result     = func(*args)
    elapsed    = round(time.time() - start, 3)
    ram_used   = round(process.memory_info().rss / 1024 / 1024 - ram_before, 2)
    return result, elapsed, ram_used


# ── UI ────────────────────────────────────────────────────────────────────────

st.title("Article Analyzer")
st.markdown("TextRank summarization and category extraction with NER tagging.")

with st.sidebar:
    st.header("Settings")

    url = st.text_input(
        "Article URL",
        placeholder="https://www.hindustantimes.com/..."
    )

    st.divider()
    st.subheader("Summarization")

    n_sentences = st.slider(
        "Number of sentences in summary",
        min_value=2, max_value=15, value=5
    )

    min_sent_len = st.slider(
        "Minimum sentence length (chars)",
        min_value=20, max_value=100, value=40,
        help="Shorter sentences are filtered before scoring"
    )

    focus_phrases_input = st.text_input(
        "Focus phrases (comma separated)",
        placeholder="e.g. Narendra Modi, Budget 2025",
        help="Sentences containing these phrases get boosted scores"
    )
    focus_phrases = (
        [p.strip() for p in focus_phrases_input.split(",") if p.strip()]
        if focus_phrases_input.strip() else None
    )

    run_button = st.button("Analyze Article", type="primary", use_container_width=True)


# ── Main ──────────────────────────────────────────────────────────────────────

if run_button and not url:
    st.warning("Please enter a URL.")

if run_button and url:
    with st.spinner("Fetching article..."):
        raw_text = fetch_and_extract(url)

    if not raw_text:
        st.error("Could not extract content from this URL. The site may block scrapers.")
    else:
        cleaned = clean_text(raw_text)
        st.session_state["article_text"] = cleaned
        st.session_state["article_url"]  = url.strip()

        st.success(f"Extracted {len(cleaned.split()):,} words from article")

        with st.expander("Full article text"):
            st.write(cleaned)

        st.divider()

        # Load model and run TextRank
        with st.spinner("Loading spaCy model..."):
            nlp = load_spacy_lg()

        with st.spinner("Running TextRank..."):
            result = textrank_summarize(
                cleaned, nlp,
                n_sentences=n_sentences,
                min_sentence_len=min_sent_len,
                focus_phrases=focus_phrases,
            )

        st.header("Summary")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Time", f"{result['elapsed_s']}s")
        with col2:
            st.metric("RAM delta", f"{result['ram_mb']} MB")
        with col3:
            st.metric("Sentences", len(result["sentences"]))

        st.markdown(result["summary"])

        with st.expander("Sentence scores (TextRank ranking)"):
            for sent_text, score in sorted(result["sentences"], key=lambda x: -x[1]):
                bar_w = int(min(score / max(s for _, s in result["sentences"]) * 100, 100)) if result["sentences"] else 0
                st.markdown(
                    f'<div style="margin-bottom:10px">'
                    f'<div style="font-size:13px;margin-bottom:4px">{sent_text}</div>'
                    f'<div style="display:flex;align-items:center;gap:8px">'
                    f'<div style="flex:1;background:#e0e0e0;border-radius:3px;height:5px">'
                    f'<div style="width:{bar_w}%;background:#1f77b4;height:5px;border-radius:3px"></div>'
                    f'</div>'
                    f'<span style="font-family:monospace;font-size:11px;color:#666">{score:.4f}</span>'
                    f'</div></div>',
                    unsafe_allow_html=True
                )


# ── Category extraction — always visible, uses same URL ───────────────────────

st.divider()
st.header("Category Extraction and NER Tagging")

from category_extractor import render_category_extractor

cat_text = None
if url and url.strip():
    if (
        "article_text" in st.session_state
        and st.session_state.get("article_url") == url.strip()
    ):
        cat_text = st.session_state["article_text"]
    elif (
        "cat_article_url" in st.session_state
        and st.session_state["cat_article_url"] == url.strip()
    ):
        cat_text = st.session_state["cat_article_text"]

render_category_extractor(article_text=cat_text, url=url.strip() if url else None)