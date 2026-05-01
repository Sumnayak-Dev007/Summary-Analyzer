import re
import os
import time
import nltk
import psutil
import streamlit as st
import trafilatura
import spacy
import pytextrank
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

st.set_page_config(
    page_title="Article Analyzer",
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


# ── Model loading ─────────────────────────────────────────────────────────────

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
SPACY_LG_PATH  = os.path.join(BASE_DIR, "local-models", "en_core_web_lg")


@st.cache_resource
def load_spacy_lg():
    """Load spaCy large model with TextRank pipeline"""
    try:
        nlp = spacy.load("en_core_web_lg")
        # Check if textrank already exists to avoid duplicate
        if "textrank" not in nlp.pipe_names:
            nlp.add_pipe("textrank")
        return nlp
    except OSError:
        st.error(
            "en_core_web_lg model not found.\n\n"
            "Please install it using one of these methods:\n\n"
            "1. Run: python -m spacy download en_core_web_lg\n\n"
            "2. Or add to requirements.txt:\n"
            "   en-core-web-lg @ https://github.com/explosion/spacy-models/releases/"
            "download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl"
        )
        return None


def fetch_article(url: str) -> tuple[str | None, str | None]:
    """
    Returns (cleaned_text, raw_html) — raw_html kept so trafilatura
    keyword extraction can use the full document structure.
    """
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return None, None
    raw_text = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=False,
        favor_precision=True,
        deduplicate=True,
    )
    if not raw_text:
        return None, None
    lines = []
    for line in raw_text.split("\n"):
        line = line.strip()
        if len(line) < 40:
            continue
        if NOISE_PATTERNS.search(line):
            continue
        lines.append(line)
    cleaned = " ".join(lines) if lines else None
    return cleaned, downloaded


def clean_text(text: str) -> str:
    text = NOISE_PATTERNS.sub("", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def sumy_textrank_summarize(text: str, n_sentences: int = 5) -> tuple[str, dict]:
    """Sumy's TextRank implementation (keyword-based)"""
    process = psutil.Process(os.getpid())
    ram_before = process.memory_info().rss / 1024 / 1024
    start = time.time()
    
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = TextRankSummarizer()
        summary_sentences = summarizer(parser.document, n_sentences)
        
        # Get all sentences in original order
        original_sentences = [str(sent) for sent in parser.document.sentences]
        
        # Extract summary sentences
        summary_texts = [str(s) for s in summary_sentences if len(str(s)) > 40]
        
        # Preserve original order
        ordered = []
        for sent in summary_texts:
            try:
                idx = original_sentences.index(sent)
                ordered.append((idx, sent))
            except ValueError:
                continue
        
        ordered.sort(key=lambda x: x[0])
        summary = " ".join([s[1] for s in ordered]).strip()
        
    except Exception as e:
        summary = f"Error: {e}"
    
    elapsed = round(time.time() - start, 3)
    ram_used = round(process.memory_info().rss / 1024 / 1024 - ram_before, 2)
    
    return summary, {"elapsed": elapsed, "ram": ram_used, "words": len(summary.split())}


def textrank_summarize(
    text: str,
    nlp,
    n_sentences: int = 7,
    min_sentence_len: int = 32,
    focus_phrases: list[str] | None = None,
) -> dict:
    proc = psutil.Process(os.getpid())
    ram0 = proc.memory_info().rss / 1024 / 1024
    t0   = time.monotonic()

    doc = nlp(text[:100_000])

    sent_scores: dict[str, float] = {}
    for phrase in doc._.phrases:
        for sent in doc.sents:
            if phrase.text.lower() in sent.text.lower():
                key = sent.text.strip()
                sent_scores[key] = sent_scores.get(key, 0) + phrase.rank

    if focus_phrases:
        for key in sent_scores:
            for fp in focus_phrases:
                if fp.lower() in key.lower():
                    sent_scores[key] *= 1.5

    sent_scores = {s: sc for s, sc in sent_scores.items() if len(s) >= min_sentence_len}

    if not sent_scores:
        fallback = [
            s.text.strip() for s in doc.sents
            if len(s.text.strip()) >= min_sentence_len
        ][:n_sentences]
        return {
            "summary":   " ".join(fallback),
            "sentences": [(s, 0.0) for s in fallback],
            "elapsed_s": round(time.monotonic() - t0, 3),
            "ram_mb":    round(proc.memory_info().rss / 1024 / 1024 - ram0, 1),
            "method":    "fallback",
        }

    top = sorted(sent_scores.items(), key=lambda x: -x[1])[:n_sentences * 2]

    def _overlap(a: str, b: str) -> float:
        ta, tb = set(a.lower().split()), set(b.lower().split())
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / min(len(ta), len(tb))

    selected: list[tuple[str, float]] = []
    for sent_text, score in top:
        if len(selected) >= n_sentences:
            break
        if any(_overlap(sent_text, s) > 0.6 for s, _ in selected):
            continue
        selected.append((sent_text, score))

    all_ordered = [s.text.strip() for s in doc.sents]
    selected    = sorted(
        selected,
        key=lambda x: all_ordered.index(x[0]) if x[0] in all_ordered else 9999
    )

    return {
        "summary":   " ".join(s for s, _ in selected),
        "sentences": selected,
        "elapsed_s": round(time.monotonic() - t0, 3),
        "ram_mb":    round(proc.memory_info().rss / 1024 / 1024 - ram0, 1),
        "method":    "textrank",
    }


# ── UI ────────────────────────────────────────────────────────────────────────

st.title("Article Analyzer")

with st.sidebar:
    st.header("Settings")

    url = st.text_input(
        "Article URL",
        placeholder="https://..."
    )

    st.divider()
    st.subheader("Summarization")
    n_sentences   = st.slider("Summary sentences", 2, 15, 7)
    min_sent_len  = st.slider("Min sentence length (chars)", 20, 100, 32)
    focus_input   = st.text_input(
        "Focus phrases (comma separated)",
        placeholder="e.g. Narendra Modi, Budget 2025"
    )
    focus_phrases = (
        [p.strip() for p in focus_input.split(",") if p.strip()]
        if focus_input.strip() else None
    )

    btn_summarize  = st.button("Analyze Article",      type="primary",   width="stretch")

    st.divider()
    st.subheader("Category Extraction")
    btn_categories = st.button("Extract Categories",   type="secondary", width="stretch")


# ── Fetch article (shared between both tasks) ─────────────────────────────────

def get_article(url: str) -> tuple[str | None, str | None]:
    """
    Returns cached (cleaned_text, raw_html) if URL already fetched,
    otherwise fetches and caches it.
    """
    if (
        "article_url"  in st.session_state
        and st.session_state["article_url"] == url.strip()
        and "article_text" in st.session_state
    ):
        return st.session_state["article_text"], st.session_state.get("article_html")

    with st.spinner("Fetching article..."):
        cleaned, html = fetch_article(url)

    if cleaned:
        st.session_state["article_text"] = cleaned
        st.session_state["article_html"] = html
        st.session_state["article_url"]  = url.strip()

    return cleaned, html


# ── Summarization task ────────────────────────────────────────────────────────

if btn_summarize:
    if not url:
        st.warning("Enter a URL first.")
    else:
        cleaned, _ = get_article(url)
        if not cleaned:
            st.error("Could not extract content from this URL.")
        else:
            # Run Sumy TextRank
            with st.spinner("Running Sumy TextRank..."):
                sumy_summary, sumy_metrics = sumy_textrank_summarize(cleaned, n_sentences)
            
            # Run spaCy PyTextRank
            with st.spinner("Loading spaCy model..."):
                nlp = load_spacy_lg()
            
            if nlp is None:
                st.error("spaCy model could not be loaded. Check requirements.txt.")
            else:
                with st.spinner("Running spaCy PyTextRank..."):
                    result = textrank_summarize(
                        cleaned, nlp,
                        n_sentences=n_sentences,
                        min_sentence_len=min_sent_len,
                        focus_phrases=focus_phrases,
                    )
                
                # Store both results
                st.session_state["sumy_summary"] = sumy_summary
                st.session_state["sumy_metrics"] = sumy_metrics
                st.session_state["summary_result"] = result
                st.session_state["summary_text"] = cleaned


# ── Render summary result (persists independently) ────────────────────────────

if "summary_result" in st.session_state:
    result = st.session_state["summary_result"]
    sumy_summary = st.session_state.get("sumy_summary", "")
    sumy_metrics = st.session_state.get("sumy_metrics", {})
    cleaned = st.session_state["summary_text"]

    st.header("TextRank Method Comparison")
    
    # Display side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Sumy TextRank")
        st.markdown("*Keyword-based TextRank*")
        
        if sumy_metrics:
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Time", f"{sumy_metrics.get('elapsed', 0)}s")
            with m2:
                st.metric("RAM", f"{sumy_metrics.get('ram', 0)} MB")
            with m3:
                st.metric("Words", sumy_metrics.get('words', 0))
        
        st.markdown("**Summary:**")
        st.markdown(sumy_summary if sumy_summary else "No summary generated")
    
    with col2:
        st.markdown("### spaCy PyTextRank")
        st.markdown("*Phrase-based TextRank*")
        
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Time", f"{result['elapsed_s']}s")
        with m2:
            st.metric("RAM delta", f"{result['ram_mb']} MB")
        with m3:
            st.metric("Sentences", len(result["sentences"]))
        
        st.markdown("**Summary:**")
        st.markdown(result["summary"])
    
    st.divider()
    
    with st.expander("Full article text"):
        st.write(cleaned)
    
    # Show sentence scores for spaCy
    if result["sentences"] and any(s > 0 for _, s in result["sentences"]):
        with st.expander("spaCy PyTextRank - Sentence Scores"):
            max_score = max(s for _, s in result["sentences"]) or 1
            for sent_text, score in sorted(result["sentences"], key=lambda x: -x[1]):
                bar_w = int(score / max_score * 100)
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
    
    # Key differences explanation
    with st.expander("Key Differences Between Methods"):
        st.markdown("""
        **Sumy TextRank:**
        - Builds graph where nodes are **words**
        - Connects words that appear near each other
        - Good for: General content, faster processing
        - May miss named entities
        
        **spaCy PyTextRank:**
        - Builds graph where nodes are **noun phrases**
        - Uses grammatical dependencies (subject, object, etc.)
        - Good for: Articles with many named entities (people, places, organizations)
        - Better captures quotes and important names
        """)


# ── Category extraction task ──────────────────────────────────────────────────

if btn_categories:
    if not url:
        st.warning("Enter a URL first.")
    else:
        cleaned, html = get_article(url)
        if not cleaned:
            st.error("Could not extract content from this URL.")
        else:
            # Import here to avoid circular imports
            from category_extractor import run_extraction, render_cat_results
            st.divider()
            st.header("Category Extraction and NER Tagging")
            
            # Run extraction and get result
            with st.spinner("Extracting categories and entities..."):
                cat_result = run_extraction(cleaned, html)
            
            # Store and render
            st.session_state["cat_result"] = cat_result
            st.session_state["cat_text"] = cleaned
            render_cat_results(cat_result)


# ── Render category result (persists independently) ───────────────────────────

if "cat_result" in st.session_state and not btn_categories:
    from category_extractor import render_cat_results
    st.divider()
    st.header("Category Extraction and NER Tagging")
    render_cat_results(st.session_state["cat_result"])