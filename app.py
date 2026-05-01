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
        if "textrank" not in nlp.pipe_names:
            nlp.add_pipe("textrank")
        return nlp
    except OSError:
        st.error(
            "en_core_web_lg model not found.\n\n"
            "Please install it using: python -m spacy download en_core_web_lg"
        )
        return None


def fetch_article(url: str) -> tuple[str | None, str | None]:
    """Returns (cleaned_text, raw_html)"""
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


def auto_detect_focus_phrases(text: str, nlp) -> list[str]:
    """
    Automatically detect important focus phrases from the article.
    Returns list of important named entities and key phrases.
    """
    doc = nlp(text[:50000])
    
    # Collect named entities (people, organizations, places, products, events)
    entities = []
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "GPE", "PRODUCT", "EVENT", "NORP", "LOC"]:
            text_clean = ent.text.strip()
            if 3 < len(text_clean) < 40:
                entities.append(text_clean)
    
    # Count frequency of each entity
    from collections import Counter
    entity_counts = Counter(entities)
    
    # Get top entities by frequency (at least 2 mentions)
    top_entities = [entity for entity, count in entity_counts.most_common(10) if count >= 2]
    
    # If no repeated entities, take unique ones
    if not top_entities:
        top_entities = list(dict.fromkeys(entities))[:7]
    
    return top_entities[:7]  # Return top 7 focus phrases


def textrank_summarize(
    text: str,
    nlp,
    n_sentences: int = 7,
    min_sentence_len: int = 32,
    focus_phrases: list[str] | None = None,
    phrase_boost: float = 1.5,
    diversity_threshold: float = 0.6,
    bullet_points: bool = False,
) -> dict:
    """
    spaCy + PyTextRank summarization with improved lead sentence detection.
    """
    proc = psutil.Process(os.getpid())
    ram0 = proc.memory_info().rss / 1024 / 1024
    t0 = time.monotonic()

    doc = nlp(text[:100_000])
    
    # Get all sentences
    all_sentences = [sent.text.strip() for sent in doc.sents]
    
    # ALWAYS include the first sentence if it's substantial (for news articles)
    first_sentence = all_sentences[0] if all_sentences else ""
    has_lead = len(first_sentence) > 40

    sent_scores: dict[str, float] = {}
    for phrase in doc._.phrases:
        for sent in doc.sents:
            if phrase.text.lower() in sent.text.lower():
                key = sent.text.strip()
                sent_scores[key] = sent_scores.get(key, 0) + phrase.rank

    # Apply focus phrase boosting
    if focus_phrases:
        for key in sent_scores:
            for fp in focus_phrases:
                if fp.lower() in key.lower():
                    sent_scores[key] *= phrase_boost

    sent_scores = {s: sc for s, sc in sent_scores.items() if len(s) >= min_sentence_len}

    if not sent_scores:
        fallback = [
            s.text.strip() for s in doc.sents
            if len(s.text.strip()) >= min_sentence_len
        ][:n_sentences]
        summary = " ".join(fallback)
        if bullet_points:
            sentences_list = [s.strip() for s in summary.split('. ') if s.strip()]
            bullet_summary = "\n".join([f"• {s}." for s in sentences_list])
            summary = bullet_summary
        
        return {
            "summary": summary,
            "sentences": [(s, 0.0) for s in fallback],
            "elapsed_s": round(time.monotonic() - t0, 3),
            "ram_mb": round(proc.memory_info().rss / 1024 / 1024 - ram0, 1),
            "method": "fallback",
        }

    top = sorted(sent_scores.items(), key=lambda x: -x[1])[:n_sentences * 2]

    def _overlap(a: str, b: str) -> float:
        ta, tb = set(a.lower().split()), set(b.lower().split())
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / min(len(ta), len(tb))

    selected: list[tuple[str, float]] = []
    
    # PRIORITY 1: Add the first sentence if it's a good lead
    if has_lead and first_sentence in sent_scores:
        selected.append((first_sentence, sent_scores[first_sentence]))
    elif has_lead:
        # First sentence not in scores but important - add it anyway
        selected.append((first_sentence, 1.0))
    
    # PRIORITY 2: Add other important sentences
    for sent_text, score in top:
        if len(selected) >= n_sentences:
            break
        # Skip if already selected
        if sent_text == first_sentence:
            continue
        if any(_overlap(sent_text, s) > diversity_threshold for s, _ in selected):
            continue
        selected.append((sent_text, score))

    # Reorder to original article sequence
    selected = sorted(
        selected,
        key=lambda x: all_sentences.index(x[0]) if x[0] in all_sentences else 9999
    )

    # Build summary
    summary_sentences = [s for s, _ in selected]
    summary = " ".join(summary_sentences)
    
    # Clean up the summary (remove duplicate periods, fix spacing)
    summary = re.sub(r'\s+', ' ', summary)
    summary = re.sub(r'\.\s+\.', '.', summary)
    summary = re.sub(r'\s+\.', '.', summary)
    
    # Ensure first sentence is complete (ends with period)
    if summary and not summary[0].isupper():
        summary = summary[0].upper() + summary[1:]
    
    # Convert to bullet points if requested
    if bullet_points:
        sentences_list = []
        for sent in summary_sentences:
            sent = sent.strip()
            if sent:
                # Ensure sentence ends with period
                if not sent.endswith(('.', '!', '?')):
                    sent = sent + '.'
                sentences_list.append(f"• {sent}")
        summary = "\n".join(sentences_list)

    return {
        "summary": summary,
        "sentences": selected,
        "elapsed_s": round(time.monotonic() - t0, 3),
        "ram_mb": round(proc.memory_info().rss / 1024 / 1024 - ram0, 1),
        "method": "textrank",
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
    st.subheader("Summarization Settings")
    
    n_sentences = st.slider("Number of sentences", 2, 15, 7,
                            help="More sentences = more detailed summary")
    
    # Remove min_sentence_len slider when bullet points is checked
    min_sent_len = st.slider("Min sentence length (chars)", 20, 100, 32,
                             help="Shorter = more sentences included")
    
    st.divider()
    
    # Bullet points option
    bullet_points = st.checkbox("Show summary as bullet points", value=False,
                                help="Convert summary into easy-to-read bullet points")
    
    st.divider()
    
    # Focus phrases section - only show if article is loaded
    st.subheader("Focus Phrases")
    st.markdown("*Automatically detected from the article*")
    
    # Placeholder for focus phrases - will be populated after article fetch
    focus_phrases_selected = []
    
    # These buttons will be enabled after article is fetched
    btn_summarize = st.button("Analyze Article", type="primary", width="stretch")

    st.divider()
    st.subheader("Category Extraction")
    btn_categories = st.button("Extract Categories", type="secondary", width="stretch")


# ── Fetch article (shared between both tasks) ─────────────────────────────────

def get_article(url: str) -> tuple[str | None, str | None]:
    if (
        "article_url" in st.session_state
        and st.session_state["article_url"] == url.strip()
        and "article_text" in st.session_state
    ):
        return st.session_state["article_text"], st.session_state.get("article_html")

    with st.spinner("Fetching article..."):
        cleaned, html = fetch_article(url)

    if cleaned:
        st.session_state["article_text"] = cleaned
        st.session_state["article_html"] = html
        st.session_state["article_url"] = url.strip()

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
            # Load spaCy model
            with st.spinner("Loading spaCy model..."):
                nlp = load_spacy_lg()
            
            if nlp is None:
                st.error("spaCy model could not be loaded.")
            else:
                # Auto-detect focus phrases
                with st.spinner("Analyzing article for key topics..."):
                    auto_focus_phrases = auto_detect_focus_phrases(cleaned, nlp)
                
                # Store for later use
                st.session_state["auto_focus_phrases"] = auto_focus_phrases
                
                # Show detected focus phrases and let user select
                st.info(f"📌 **Detected key topics in this article:**")
                
                # Create multiselect for focus phrases
                focus_phrases_selected = st.multiselect(
                    "Select phrases to focus on (minimum 1 recommended):",
                    options=auto_focus_phrases,
                    default=auto_focus_phrases[:3] if len(auto_focus_phrases) >= 3 else auto_focus_phrases,
                    help="Selecting phrases makes the summary emphasize these topics"
                )
                
                # Validate minimum selection
                if len(focus_phrases_selected) == 0:
                    st.warning("⚠️ Please select at least one focus phrase for better results. Using default focus.")
                    focus_phrases_selected = auto_focus_phrases[:2] if auto_focus_phrases else None
                
                # Option to boost phrase importance
                phrase_boost = st.slider(
                    "How much to emphasize selected phrases?",
                    1.0, 2.5, 1.5, 0.1,
                    help="Higher = more weight on sentences containing selected phrases"
                )
                
                st.divider()
                
                # Run summarization
                with st.spinner("Generating summary..."):
                    result = textrank_summarize(
                        cleaned, nlp,
                        n_sentences=n_sentences,
                        min_sentence_len=min_sent_len,
                        focus_phrases=focus_phrases_selected if focus_phrases_selected else None,
                        phrase_boost=phrase_boost,
                        diversity_threshold=0.6,
                        bullet_points=bullet_points,
                    )
                
                st.session_state["summary_result"] = result
                st.session_state["summary_text"] = cleaned
                st.session_state["focus_phrases_used"] = focus_phrases_selected


# ── Render summary result (persists independently) ────────────────────────────

if "summary_result" in st.session_state:
    result = st.session_state["summary_result"]
    cleaned = st.session_state["summary_text"]
    focus_phrases_used = st.session_state.get("focus_phrases_used", [])

    st.header("Summary")

    # Show metrics
    c1, c2, c3 = st.columns(3)
    with c1: 
        st.metric("Time", f"{result['elapsed_s']}s")
    with c2: 
        st.metric("RAM delta", f"{result['ram_mb']} MB")
    with c3: 
        st.metric("Total items", len(result["sentences"]) if not bullet_points else len(result["summary"].split('\n')))

    # Show focus phrases used
    if focus_phrases_used:
        st.info(f"🎯 **Focusing on:** {', '.join(focus_phrases_used)}")

    # Display summary
    if bullet_points:
        st.markdown("### Summary (Bullet Points)")
        st.markdown(result["summary"])
    else:
        st.markdown("### Summary")
        st.markdown(result["summary"])

    with st.expander("Full article text"):
        st.write(cleaned)

    # Show sentence scores for debugging
    if result["sentences"] and any(s > 0 for _, s in result["sentences"]):
        with st.expander("Sentence Scores (Debug)"):
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


# ── Category extraction task ──────────────────────────────────────────────────

if btn_categories:
    if not url:
        st.warning("Enter a URL first.")
    else:
        cleaned, html = get_article(url)
        if not cleaned:
            st.error("Could not extract content from this URL.")
        else:
            from category_extractor import run_extraction, render_cat_results
            st.divider()
            st.header("Category Extraction and NER Tagging")
            
            with st.spinner("Extracting categories and entities..."):
                cat_result = run_extraction(cleaned, html)
            
            st.session_state["cat_result"] = cat_result
            st.session_state["cat_text"] = cleaned
            render_cat_results(cat_result)


# ── Render category result (persists independently) ───────────────────────────

if "cat_result" in st.session_state and not btn_categories:
    from category_extractor import render_cat_results
    st.divider()
    st.header("Category Extraction and NER Tagging")
    render_cat_results(st.session_state["cat_result"])