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


def smart_article_cleaning(text: str) -> tuple[str, str | None]:
    """
    Separates title from article body for better summarization.
    Returns (full_text_for_context, title_for_display)
    """
    lines = text.split('. ')
    
    # Try to detect title (first segment that's shorter and likely a headline)
    title = None
    body = text
    
    # Check if first part looks like a title (quoted, short, no verb)
    first_part = lines[0].strip() if lines else ""
    
    if first_part and (first_part.startswith("'") or first_part.startswith('"') or first_part.startswith('‘')):
        # Extract title (remove quotes)
        title = first_part.strip("'\"‘’")
        # Remove title from body
        body = text[len(first_part):].strip()
        if body.startswith(('.', '!', '?')):
            body = body[1:].strip()
    
    # Also check for title pattern: short line (< 80 chars) without ending punctuation
    elif len(first_part) < 80 and not first_part.endswith(('.', '!', '?')):
        title = first_part
        body = '. '.join(lines[1:]) if len(lines) > 1 else text
    
    # If title found, prepend it with proper separation for context
    if title:
        full_text = f"{title}. {body}"
    else:
        full_text = body
    
    return full_text, title


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
    
    # Get top entities by frequency
    top_entities = [entity for entity, count in entity_counts.most_common(10)]
    
    # If no entities found, extract important noun phrases
    if not top_entities:
        noun_phrases = []
        for chunk in doc.noun_chunks:
            if 2 <= len(chunk.text.split()) <= 4 and len(chunk.text) > 5:
                noun_phrases.append(chunk.text.strip())
        top_entities = list(dict.fromkeys(noun_phrases))[:7]
    
    return top_entities[:7]


def textrank_summarize(
    text: str,
    nlp,
    n_sentences: int = 6,
    min_sentence_len: int = 32,
    focus_phrases: list[str] | None = None,
    phrase_boost: float = 1.5,
    diversity_threshold: float = 0.6,
    bullet_points: bool = False,
    title: str | None = None,
) -> dict:
    """
    spaCy + PyTextRank summarization with intelligent lead sentence detection.
    Focus phrases only applied if provided.
    """
    proc = psutil.Process(os.getpid())
    ram0 = proc.memory_info().rss / 1024 / 1024
    t0 = time.monotonic()

    # Process with title for better context
    doc = nlp(text[:100_000])
    
    # Get all sentences
    all_sentences = [sent.text.strip() for sent in doc.sents]
    
    # Find the proper lead sentence (skip title if embedded)
    first_proper_sentence = all_sentences[0] if all_sentences else ""
    
    # If first sentence contains the title as prefix, extract the actual lead
    if title and title in first_proper_sentence:
        lead_start = first_proper_sentence.find(title) + len(title)
        if lead_start > 0 and lead_start < len(first_proper_sentence):
            actual_lead = first_proper_sentence[lead_start:].strip()
            actual_lead = actual_lead.lstrip('.!? ')
            if actual_lead and len(actual_lead) > 30:
                first_proper_sentence = actual_lead

    sent_scores: dict[str, float] = {}
    for phrase in doc._.phrases:
        for sent in doc.sents:
            if phrase.text.lower() in sent.text.lower():
                key = sent.text.strip()
                sent_scores[key] = sent_scores.get(key, 0) + phrase.rank

    # Apply focus phrase boosting ONLY if focus_phrases provided
    if focus_phrases and len(focus_phrases) > 0:
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
        
        if bullet_points:
            sentences_list = []
            for sent in fallback:
                sent = sent.strip()
                if sent:
                    if not sent.endswith(('.', '!', '?')):
                        sent = sent + '.'
                    sentences_list.append(f"• {sent}")
            summary = "\n\n".join(sentences_list)
        else:
            summary = " ".join(fallback)
        
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
    
    # Intelligent lead sentence selection
    lead_candidates = []
    
    # Candidate 1: First proper sentence
    if len(first_proper_sentence) >= min_sentence_len:
        lead_candidates.append((first_proper_sentence, sent_scores.get(first_proper_sentence, 1.2)))
    
    # Candidate 2: Highest scoring sentence
    if top:
        lead_candidates.append(top[0])
    
    # Pick best lead (highest score)
    if lead_candidates:
        best_lead = max(lead_candidates, key=lambda x: x[1])
        selected.append(best_lead)
    
    # Add other important sentences
    for sent_text, score in top:
        if len(selected) >= n_sentences:
            break
        if any(sent_text == s for s, _ in selected):
            continue
        if any(_overlap(sent_text, s) > diversity_threshold for s, _ in selected):
            continue
        selected.append((sent_text, score))

    # Reorder to original article sequence
    selected = sorted(
        selected,
        key=lambda x: all_sentences.index(x[0]) if x[0] in all_sentences else 9999
    )

    # Build summary sentences
    summary_sentences = [s for s, _ in selected]
    
    # Clean each sentence
    cleaned_sentences = []
    for sent in summary_sentences:
        sent = re.sub(r'\s+', ' ', sent)
        if not sent.endswith(('.', '!', '?')):
            sent = sent + '.'
        sent = re.sub(r'\.\.+', '.', sent)
        cleaned_sentences.append(sent)
    
    # Format based on bullet_points preference
    if bullet_points:
        bullet_list = [f"• {sent}" for sent in cleaned_sentences]
        summary = "\n\n".join(bullet_list)
    else:
        summary = " ".join(cleaned_sentences)
    
    # Ensure first letter is capitalized
    if summary and not summary[0].isupper():
        summary = summary[0].upper() + summary[1:]

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
    
    n_sentences = st.slider("Number of sentences", 2, 15, 6,
                            help="More sentences = more detailed summary")
    
    min_sent_len = st.slider("Min sentence length (chars)", 20, 100, 32,
                             help="Shorter = more sentences included")
    
    st.divider()
    
    bullet_points = st.checkbox("Show summary as bullet points", value=False,
                                help="Convert summary into easy-to-read bullet points")
    
    st.divider()
    
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


if btn_summarize:
    if not url:
        st.warning("Enter a URL first.")
    else:
        cleaned, _ = get_article(url)
        if not cleaned:
            st.error("Could not extract content from this URL.")
        else:
            # Clean article and separate title
            full_text, article_title = smart_article_cleaning(cleaned)
            
            # Show title if detected
            if article_title:
                st.info(f"Article: {article_title}")
            
            # Load spaCy model
            with st.spinner("Loading spaCy model..."):
                nlp = load_spacy_lg()
            
            if nlp is None:
                st.error("spaCy model could not be loaded.")
            else:
                # Auto-detect focus phrases
                with st.spinner("Analyzing article for key topics..."):
                    auto_focus_phrases = auto_detect_focus_phrases(full_text, nlp)
                
                # Initialize session state for focus phrases if not exists
                if "focus_phrases_selected" not in st.session_state:
                    st.session_state.focus_phrases_selected = []
                if "phrase_boost" not in st.session_state:
                    st.session_state.phrase_boost = 1.5
                if "apply_focus" not in st.session_state:
                    st.session_state.apply_focus = False
                
                # Variables to store what to use
                focus_to_use = None
                boost_to_use = 1.5
                
                # Show detected focus phrases
                if auto_focus_phrases:
                    st.info(f"Detected key topics: {', '.join(auto_focus_phrases[:5])}")
                    
                    # Focus phrase selection
                    selected = st.multiselect(
                        "Select phrases to focus on (optional):",
                        options=auto_focus_phrases,
                        default=st.session_state.focus_phrases_selected,
                        help="Selecting phrases makes the summary emphasize these topics"
                    )
                    
                    # Ensure boost value is within valid range
                    current_boost = st.session_state.phrase_boost
                    if current_boost < 1.0 or current_boost > 2.5:
                        current_boost = 1.5
                    
                    boost = st.slider(
                        "Emphasis strength for selected phrases:",
                        1.0, 2.5, current_boost, 0.1,
                        help="Higher = more weight on sentences containing selected phrases"
                    )
                    
                    # Apply button for focus phrases
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        apply_btn = st.button("Apply Focus Phrases", type="secondary")
                    
                    if apply_btn:
                        st.session_state.focus_phrases_selected = selected
                        st.session_state.phrase_boost = boost
                        st.session_state.apply_focus = True
                        st.rerun()
                    
                    # Determine which focus phrases to use
                    if st.session_state.apply_focus and st.session_state.focus_phrases_selected:
                        focus_to_use = st.session_state.focus_phrases_selected
                        boost_to_use = st.session_state.phrase_boost
                
                st.divider()
                
                # Run summarization
                with st.spinner("Generating summary..."):
                    result = textrank_summarize(
                        full_text, nlp,
                        n_sentences=n_sentences,
                        min_sentence_len=min_sent_len,
                        focus_phrases=focus_to_use,
                        phrase_boost=boost_to_use,
                        diversity_threshold=0.6,
                        bullet_points=bullet_points,
                        title=article_title,
                    )
                
                st.session_state["summary_result"] = result
                st.session_state["summary_text"] = cleaned
                st.session_state["article_title"] = article_title
                st.session_state["focus_phrases_used"] = focus_to_use if focus_to_use else []

# ── Render summary result (persists independently) ────────────────────────────

if "summary_result" in st.session_state:
    result = st.session_state["summary_result"]
    cleaned = st.session_state["summary_text"]
    article_title = st.session_state.get("article_title", "")
    focus_phrases_used = st.session_state.get("focus_phrases_used", [])

    st.header("Summary")

    # Show metrics
    c1, c2, c3 = st.columns(3)
    with c1: 
        st.metric("Time", f"{result['elapsed_s']}s")
    with c2: 
        st.metric("RAM delta", f"{result['ram_mb']} MB")
    with c3: 
        if bullet_points:
            st.metric("Bullet points", len(result["summary"].split('\n')))
        else:
            st.metric("Sentences", len(result["sentences"]))

    # Show focus phrases used
    if focus_phrases_used:
        st.info(f"Focusing on: {', '.join(focus_phrases_used)} (with {st.session_state.get('phrase_boost', 1.5)}x boost)")

    # Display summary (ONLY ONCE)
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