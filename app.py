import re
import os
import time
import nltk
import psutil
import streamlit as st
import trafilatura
import spacy
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
def load_spacy_for_ner():
    """Load spaCy large model only for NER (entity detection for focus phrases)"""
    try:
        nlp = spacy.load("en_core_web_lg")
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
    lines = text.split('\n')
    
    title = None
    body = text
    
    # Method 1: Check if first line is short and likely a title
    if lines and len(lines[0].strip()) < 100 and not lines[0].strip().endswith(('.', '!', '?')):
        title = lines[0].strip()
        body = ' '.join(lines[1:]) if len(lines) > 1 else text
    
    # Method 2: Look for quoted title
    first_part = text.split('. ')[0] if '. ' in text else text[:100]
    if first_part and (first_part.startswith("'") or first_part.startswith('"') or first_part.startswith('‘')):
        title = first_part.strip("'\"‘’")
        body = text[len(first_part):].strip()
        if body.startswith(('.', '!', '?')):
            body = body[1:].strip()
    
    # Method 3: Look for short first sentence (less than 80 chars without ending punctuation)
    elif len(first_part) < 80 and not first_part.endswith(('.', '!', '?')):
        title = first_part
        body = '. '.join(text.split('. ')[1:]) if '. ' in text else text
    
    if title:
        full_text = f"{title}. {body}"
    else:
        full_text = body
    
    return full_text, title



def remove_title_from_text(text: str, title: str | None = None) -> str:
    """
    Aggressively remove title/headline from article text.
    """
    if not title:
        # Try to auto-detect title if not provided
        lines = text.split('\n')
        if lines and len(lines[0].strip()) < 100 and not lines[0].strip().endswith(('.', '!', '?')):
            title = lines[0].strip()
    
    clean_text = text
    
    if title:
        # Method 1: Remove title as a line
        clean_text = clean_text.replace(title, '', 1)
        
        # Method 2: Remove title followed by newline
        clean_text = clean_text.replace(f"{title}\n", '', 1)
        
        # Method 3: Remove title followed by period and space
        clean_text = clean_text.replace(f"{title}. ", '', 1)
        
        # Method 4: Remove quoted title
        title_clean = title.strip("'\"‘’")
        clean_text = clean_text.replace(f"'{title_clean}' ", '', 1)
        clean_text = clean_text.replace(f'"{title_clean}" ', '', 1)
        clean_text = clean_text.replace(f"‘{title_clean}’ ", '', 1)
    
    # Method 5: Remove first line if it's short and doesn't end with punctuation
    lines = clean_text.split('\n')
    if lines and len(lines[0].strip()) < 100 and not lines[0].strip().endswith(('.', '!', '?')):
        clean_text = ' '.join(lines[1:]) if len(lines) > 1 else clean_text
    
    # Method 6: Remove first sentence if it's very short and likely a subtitle
    sentences = re.split(r'(?<=[.!?])\s+', clean_text)
    if sentences and len(sentences[0]) < 80 and not sentences[0].endswith(('.', '!', '?')):
        clean_text = ' '.join(sentences[1:]) if len(sentences) > 1 else clean_text
    
    # Method 7: Remove any line that looks like a headline (all caps or short with no period)
    lines = clean_text.split('. ')
    if lines and len(lines[0]) < 80 and ' ' in lines[0] and not any(c.islower() for c in lines[0][:10]):
        clean_text = '. '.join(lines[1:]) if len(lines) > 1 else clean_text
    
    # Clean up extra spaces
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    
    return clean_text



def fix_quote_balance(text: str) -> str:
    """Fix unbalanced quotes in the summary."""
    # Count quotes
    double_quotes = text.count('"')
    single_quotes = text.count("'")
    smart_quotes_open = text.count('“')
    smart_quotes_close = text.count('”')
    
    # Add missing closing quotes
    if double_quotes % 2 != 0:
        text = text + '"'
    if smart_quotes_open > smart_quotes_close:
        text = text + '”'
    
    # Ensure quotes are properly attributed
    # If a quote opens but no attribution word nearby, add attribution
    quote_starts = [m.start() for m in re.finditer(r'["“]', text)]
    for start in quote_starts:
        # Check if attribution word exists before the quote
        preceding = text[max(0, start-50):start]
        attribution_words = ['said', 'added', 'emphasised', 'stated', 'told', 'explained', 'noted', 'called']
        if not any(word in preceding.lower() for word in attribution_words):
            # Try to add attribution from context
            pass
    
    return text


def improve_quote_structure(text: str) -> str:
    """Improve quote structure in summary."""
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Track if we're inside a quote
    in_quote = False
    fixed_sentences = []
    
    for sent in sentences:
        # Check quote balance in this sentence
        quote_count = sent.count('"') + sent.count('“')
        if quote_count % 2 != 0:
            in_quote = not in_quote
            if not in_quote:
                # Quote closes here, ensure proper punctuation
                if not sent.endswith(('"', '”')):
                    sent = sent + '"'
        fixed_sentences.append(sent)
    
    return ' '.join(fixed_sentences)



def remove_dateline(text: str) -> str:
    """Remove datelines like 'New Delhi:', 'Washington:', etc."""
    # Remove patterns like "City Name:" at beginning
    text = re.sub(r'^[A-Z][a-z]+(?: [A-Z][a-z]+)?:\s*', '', text)
    # Remove "New Delhi:" anywhere
    text = re.sub(r'\bNew Delhi:\s*', '', text)
    # Remove "City, Country:" patterns
    text = re.sub(r'^[A-Z][a-z]+, [A-Z][a-z]+:\s*', '', text)
    return text


def is_complete_quote(sent_text: str) -> bool:
    """Check if a quote in the sentence is complete."""
    # Count quote characters
    quote_count = sent_text.count('"') + sent_text.count("'") + sent_text.count('“') + sent_text.count('”')
    
    # Even number means balanced quotes
    if quote_count > 0 and quote_count % 2 == 0:
        return True
    return False



def auto_detect_focus_phrases(text: str, nlp) -> list[str]:
    """Automatically detect important focus phrases from the article using spaCy NER."""
    doc = nlp(text[:50000])
    
    entities = []
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "GPE", "PRODUCT", "EVENT", "NORP", "LOC"]:
            text_clean = ent.text.strip()
            if 3 < len(text_clean) < 40:
                entities.append(text_clean)
    
    from collections import Counter
    entity_counts = Counter(entities)
    top_entities = [entity for entity, count in entity_counts.most_common(10)]
    
    if not top_entities:
        noun_phrases = []
        for chunk in doc.noun_chunks:
            if 2 <= len(chunk.text.split()) <= 4 and len(chunk.text) > 5:
                noun_phrases.append(chunk.text.strip())
        top_entities = list(dict.fromkeys(noun_phrases))[:7]
    
    return top_entities[:7]


def sumy_textrank_summarize(
    text: str,
    n_sentences: int = 6,
    min_sentence_len: int = 32,
    focus_phrases: list[str] | None = None,
    phrase_boost: float = 1.5,
    diversity_threshold: float = 0.6,
    bullet_points: bool = False,
    title: str | None = None,
) -> dict:
    """Sumy TextRank summarization with title removal and quote handling."""
    proc = psutil.Process(os.getpid())
    ram0 = proc.memory_info().rss / 1024 / 1024
    t0 = time.monotonic()

    # Remove datelines first
    clean_text = text
    dateline_patterns = [
        r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?:\s*',
        r'^[A-Z][a-z]+, [A-Z][a-z]+:\s*',
        r'\b[A-Z][a-z]+:\s+(?=[A-Z])',
    ]
    for pattern in dateline_patterns:
        clean_text = re.sub(pattern, '', clean_text)
    
    # Aggressively remove title
    clean_text_for_summary = remove_title_from_text(clean_text, title)
    
    # Remove any lingering datelines
    clean_text_for_summary = remove_dateline(clean_text_for_summary)
    
    # If the first sentence still looks like a title, remove it
    first_sent = clean_text_for_summary.split('. ')[0] if '. ' in clean_text_for_summary else clean_text_for_summary[:100]
    if len(first_sent) < 80 and not first_sent.endswith(('.', '!', '?')):
        parts = clean_text_for_summary.split('. ', 1)
        if len(parts) > 1:
            clean_text_for_summary = parts[1]
    
    parser = PlaintextParser.from_string(clean_text_for_summary, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    
    all_sentences = [str(sent).strip() for sent in parser.document.sentences]
    first_proper_sentence = all_sentences[0] if all_sentences else ""

    # Get summary sentences (get more for ranking)
    summary_sentences = summarizer(parser.document, n_sentences * 2)
    
    # Build sentence scores
    sent_scores: dict[str, float] = {}
    for sent in summary_sentences:
        sent_text = str(sent).strip()
        if len(sent_text) >= min_sentence_len:
            sent_scores[sent_text] = sent_scores.get(sent_text, 0) + 1
    
    # Apply focus phrase boosting
    if focus_phrases and len(focus_phrases) > 0:
        for key in sent_scores:
            for fp in focus_phrases:
                if fp.lower() in key.lower():
                    sent_scores[key] *= phrase_boost

    if not sent_scores:
        fallback = [s for s in all_sentences if len(s) >= min_sentence_len][:n_sentences]
        
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

    # Sort by score and select top sentences
    top = sorted(sent_scores.items(), key=lambda x: -x[1])[:n_sentences * 2]

    def _overlap(a: str, b: str) -> float:
        ta, tb = set(a.lower().split()), set(b.lower().split())
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / min(len(ta), len(tb))

    selected: list[tuple[str, float]] = []
    
    # Intelligent lead sentence selection
    lead_candidates = []
    
    if len(first_proper_sentence) >= min_sentence_len:
        lead_candidates.append((first_proper_sentence, sent_scores.get(first_proper_sentence, 1.2)))
    
    if top:
        lead_candidates.append(top[0])
    
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

    summary_sentences_final = [s for s, _ in selected]
    
    # Clean each sentence
    cleaned_sentences = []
    for sent in summary_sentences_final:
        sent = re.sub(r'\s+', ' ', sent)
        # Remove any lingering datelines
        sent = re.sub(r'\bNew Delhi:\s*', '', sent)
        sent = re.sub(r'\b[A-Z][a-z]+:\s*$', '', sent)
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
    if summary:
        summary = fix_quote_balance(summary)
        summary = improve_quote_structure(summary)
        if not summary[0].isupper():
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


# Initialize session state variables
if "article_loaded" not in st.session_state:
    st.session_state.article_loaded = False
if "full_text" not in st.session_state:
    st.session_state.full_text = None
if "article_title" not in st.session_state:
    st.session_state.article_title = None
if "spacy_model" not in st.session_state:
    st.session_state.spacy_model = None
if "auto_focus_phrases" not in st.session_state:
    st.session_state.auto_focus_phrases = []
if "focus_phrases_selected" not in st.session_state:
    st.session_state.focus_phrases_selected = []
if "phrase_boost" not in st.session_state:
    st.session_state.phrase_boost = 1.5
if "apply_focus" not in st.session_state:
    st.session_state.apply_focus = False
if "initial_summary_generated" not in st.session_state:
    st.session_state.initial_summary_generated = False


# Load article and generate initial summary when button is clicked
if btn_summarize and url:
    with st.spinner("Fetching and analyzing article..."):
        cleaned, _ = get_article(url)
        if cleaned:
            full_text, article_title = smart_article_cleaning(cleaned)
            spacy_nlp = load_spacy_for_ner()
            
            if spacy_nlp:
                auto_phrases = auto_detect_focus_phrases(full_text, spacy_nlp)
                
                st.session_state.full_text = full_text
                st.session_state.article_title = article_title
                st.session_state.auto_focus_phrases = auto_phrases
                st.session_state.spacy_model = spacy_nlp
                st.session_state.article_loaded = True
                st.session_state.apply_focus = False
                st.session_state.initial_summary_generated = False
                
                # Generate initial summary immediately
                with st.spinner("Generating initial summary..."):
                    initial_result = sumy_textrank_summarize(
                        full_text,
                        n_sentences=n_sentences,
                        min_sentence_len=min_sent_len,
                        focus_phrases=None,
                        phrase_boost=1.5,
                        diversity_threshold=0.6,
                        bullet_points=bullet_points,
                        title=article_title,
                    )
                
                st.session_state["summary_result"] = initial_result
                st.session_state["summary_text"] = full_text
                st.session_state["focus_phrases_used"] = []
                st.session_state.initial_summary_generated = True
                
                st.rerun()
        else:
            st.error("Could not extract content from this URL.")

# Display focus phrase selection UI (only after article is loaded)
if st.session_state.article_loaded and st.session_state.full_text:
    
    if st.session_state.article_title:
        st.info(f"Article: {st.session_state.article_title}")
    
    if st.session_state.auto_focus_phrases:
        st.info(f"Detected key topics: {', '.join(st.session_state.auto_focus_phrases[:5])}")
        
        selected = st.multiselect(
            "Select phrases to focus on (optional):",
            options=st.session_state.auto_focus_phrases,
            default=st.session_state.focus_phrases_selected,
            help="Selecting phrases makes the summary emphasize these topics",
            key="focus_phrases_multiselect"
        )
        
        current_boost = st.session_state.phrase_boost
        if current_boost < 1.0 or current_boost > 2.5:
            current_boost = 1.5
        
        boost = st.slider(
            "Emphasis strength for selected phrases:",
            1.0, 2.5, current_boost, 0.1,
            help="Higher = more weight on sentences containing selected phrases",
            key="phrase_boost_slider"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            apply_btn = st.button("Apply Focus Phrases & Regenerate", key="apply_focus_btn")
        
        if apply_btn:
            st.session_state.focus_phrases_selected = selected
            st.session_state.phrase_boost = boost
            st.session_state.apply_focus = True
            
            with st.spinner("Regenerating summary with focus phrases..."):
                if st.session_state.apply_focus and st.session_state.focus_phrases_selected:
                    focus_to_use = st.session_state.focus_phrases_selected
                    boost_to_use = st.session_state.phrase_boost
                else:
                    focus_to_use = None
                    boost_to_use = 1.5
                
                new_result = sumy_textrank_summarize(
                    st.session_state.full_text,
                    n_sentences=n_sentences,
                    min_sentence_len=min_sent_len,
                    focus_phrases=focus_to_use,
                    phrase_boost=boost_to_use,
                    diversity_threshold=0.6,
                    bullet_points=bullet_points,
                    title=st.session_state.article_title,
                )
            
            st.session_state["summary_result"] = new_result
            st.session_state["focus_phrases_used"] = focus_to_use if focus_to_use else []
            st.rerun()


# ── Render summary result ─────────────────────────────────────────────────────

if "summary_result" in st.session_state:
    result = st.session_state["summary_result"]
    cleaned = st.session_state["summary_text"]
    focus_phrases_used = st.session_state.get("focus_phrases_used", [])

    st.header("Summary")

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

    if focus_phrases_used:
        st.info(f"Focusing on: {', '.join(focus_phrases_used)} (with {st.session_state.get('phrase_boost', 1.5)}x boost)")

    if bullet_points:
        st.markdown("### Summary (Bullet Points)")
        st.markdown(result["summary"])
    else:
        st.markdown("### Summary")
        st.markdown(result["summary"])

    with st.expander("Full article text"):
        st.write(cleaned)

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