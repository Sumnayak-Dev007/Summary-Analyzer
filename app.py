import re
import os
import time
import nltk
import psutil
import requests
import streamlit as st
import trafilatura
import spacy
import pytextrank
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sumy.parsers.plaintext import PlaintextParser
from nltk.tokenize import sent_tokenize
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.sum_basic import SumBasicSummarizer
from sumy.summarizers.kl import KLSummarizer
from transformers import (
    BartForConditionalGeneration, BartTokenizer,
    T5ForConditionalGeneration, T5Tokenizer
)
from category_extractor import render_category_extractor

st.set_page_config(
    page_title  = "Summary Comparator",
    page_icon   = "🔬",
    layout      = "wide"
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


BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(BASE_DIR, "local-models", "bart-large-cnn")
T5_MODEL_PATH = os.path.join(BASE_DIR, "local-models", "t5-small")
ST_MODEL_PATH = os.path.join(BASE_DIR, "local-models", "all-MiniLM-L6-v2")
SPACY_MODEL_PATH = os.path.join(BASE_DIR, "local-models", "en_core_web_sm")


@st.cache_resource
def load_sentence_transformer():
    """Load sentence transformer from local folder, download if not present"""
    if not os.path.exists(ST_MODEL_PATH):
        print("Sentence Transformer not found locally, downloading...")
        os.makedirs(ST_MODEL_PATH, exist_ok=True)
        model = SentenceTransformer("all-MiniLM-L6-v2")
        model.save(ST_MODEL_PATH)
        print("Sentence Transformer saved!")
    else:
        print("Sentence Transformer found locally, loading...")

    return SentenceTransformer(ST_MODEL_PATH)


@st.cache_resource
def load_spacy():
    try:
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe("textrank")
        return nlp

    except OSError:
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe("textrank")
        return nlp


@st.cache_resource
def load_bart():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading BART model (~1.6GB)..."):
            os.makedirs(MODEL_PATH, exist_ok=True)
            BartForConditionalGeneration.from_pretrained(
                "facebook/bart-large-cnn").save_pretrained(MODEL_PATH)
            BartTokenizer.from_pretrained(
                "facebook/bart-large-cnn").save_pretrained(MODEL_PATH)
    model     = BartForConditionalGeneration.from_pretrained(MODEL_PATH)
    tokenizer = BartTokenizer.from_pretrained(MODEL_PATH)
    return model, tokenizer


@st.cache_resource
def load_t5():
    if not os.path.exists(T5_MODEL_PATH):
        with st.spinner("Downloading T5 model (~60MB)..."):
            os.makedirs(T5_MODEL_PATH, exist_ok=True)
            T5ForConditionalGeneration.from_pretrained(
                "t5-small").save_pretrained(T5_MODEL_PATH)
            T5Tokenizer.from_pretrained(
                "t5-small").save_pretrained(T5_MODEL_PATH)
    model     = T5ForConditionalGeneration.from_pretrained(T5_MODEL_PATH)
    tokenizer = T5Tokenizer.from_pretrained(T5_MODEL_PATH)
    return model, tokenizer



def fetch_and_extract(url: str) -> str | None:
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return None

    raw_text = trafilatura.extract(
        downloaded,
        include_comments = False,
        include_tables   = False,
    )

    if not raw_text:
        return None

    lines = []
    for line in raw_text.split("\n"):
        line = line.strip()
        if len(line) < 50:
            continue
        if NOISE_PATTERNS.search(line):
            continue
        lines.append(line)

    return " ".join(lines) if lines else None


def clean_text(text: str) -> str:
    text = NOISE_PATTERNS.sub("", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def sumy_summarize(text, summarizer_class, n):
    try:
        parser            = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer        = summarizer_class()
        summary_sentences = summarizer(parser.document, n)
        original          = [str(s) for s in parser.document.sentences]
        ordered           = []
        for s in summary_sentences:
            s_str = str(s)
            if len(s_str) <= 40:
                continue
            try:
                idx = original.index(s_str)
                ordered.append((idx, s_str))
            except ValueError:
                continue
        ordered.sort(key=lambda x: x[0])
        return " ".join([i[1] for i in ordered]).strip()
    except Exception as e:
        return f"Failed: {e}"




def pytextrank_summarize(text, n, nlp):
    """
    Uses spaCy + PyTextRank to extract key sentences.
    PyTextRank scores sentences based on the importance
    of the key phrases they contain.
    """
    try:
        doc       = nlp(text)
        sentences = [
            sent.text.strip()
            for sent in doc._.textrank.summary(limit_sentences=n)
            if len(sent.text.strip()) > 40
        ]

        # Preserve original article order
        all_sentences = [sent.text.strip() for sent in doc.sents]
        ordered       = []
        for sent in sentences:
            try:
                idx = all_sentences.index(sent)
                ordered.append((idx, sent))
            except ValueError:
                continue

        ordered.sort(key=lambda x: x[0])
        return " ".join([s[1] for s in ordered]).strip()

    except Exception as e:
        return f"Failed: {e}"


def semantic_summarize(text, n, st_model):
    """
    Uses Sentence Transformers to find sentences that are
    most semantically similar to the overall document meaning.
    Unlike keyword methods — understands meaning not just words.
    """
    try:
        sentences = sent_tokenize(text)
        sentences = [s for s in sentences if len(s.strip()) > 40]

        if len(sentences) <= n:
            return " ".join(sentences)

        # Encode all sentences into vectors
        embeddings = st_model.encode(sentences)

        # Encode full document as average of all sentence vectors
        doc_embedding = embeddings.mean(axis=0, keepdims=True)

        # Score each sentence by similarity to document
        scores      = cosine_similarity(embeddings, doc_embedding).flatten()

        # Pick top N indices
        top_indices = scores.argsort()[-n:][::-1]

        # Sort back into original article order
        top_indices = sorted(top_indices)

        return " ".join([sentences[i] for i in top_indices])

    except Exception as e:
        return f"Failed: {e}"




def bart_summarize(text, bart):
    try:
        model, tokenizer = bart
        inputs           = tokenizer(
            text, max_length=1024, truncation=True, return_tensors="pt"
        )
        ids = model.generate(
            inputs["input_ids"],
            max_length=130, min_length=30,
            length_penalty=2.0, num_beams=4, early_stopping=True
        )
        return tokenizer.decode(ids[0], skip_special_tokens=True)
    except Exception as e:
        return f"Failed: {e}"


def t5_summarize(text, t5):
    try:
        model, tokenizer = t5
        inputs           = tokenizer(
            "summarize: " + text,
            max_length=512, truncation=True, return_tensors="pt"
        )
        ids     = model.generate(
            inputs["input_ids"],
            max_length=130, min_length=30,
            length_penalty=2.0, num_beams=4, early_stopping=True
        )
        summary   = tokenizer.decode(ids[0], skip_special_tokens=True)
        sentences = re.split(r'(?<=[.!?])\s+', summary)
        return " ".join([s.capitalize() for s in sentences])
    except Exception as e:
        return f"Failed: {e}"


def first_sentences(text, n):
    sentences = [
        s.strip() for s in re.split(r"(?<=[.!?])\s+", text)
        if len(s.strip()) > 35
    ]
    return " ".join(sentences[:n])


def measure(func, *args):
    process    = psutil.Process(os.getpid())
    ram_before = process.memory_info().rss / 1024 / 1024
    start      = time.time()
    result     = func(*args)
    elapsed    = round(time.time() - start, 3)
    ram_used   = round(process.memory_info().rss / 1024 / 1024 - ram_before, 2)
    return result, elapsed, ram_used


st.title("Summary Method Comparator")
st.markdown("Compare extractive and abstractive summarization methods side by side.")


with st.sidebar:
    st.header("Settings")

    url = st.text_input(
        "Article URL",
        placeholder="https://www.hindustantimes.com/..."
    )

    num_sentences = st.slider(
        "Sentences to be picked by extractive methods",
        min_value = 2,
        max_value = 10,
        value     = 5
    )

    st.divider()
    st.subheader("Methods to Run")

    st.divider()
    st.subheader("Extractive Methods")
    run_lsa       = st.checkbox("LSA",           value=True)
    run_lexrank   = st.checkbox("LexRank",        value=True)
    run_textrank  = st.checkbox("TextRank",       value=True)
    run_luhn      = st.checkbox("Luhn",           value=True)
    run_sumbasic  = st.checkbox("SumBasic",       value=True)
    run_kl        = st.checkbox("KL-Divergence",  value=True)
    run_pytextrank = st.checkbox("PyTextRank (spaCy)",      value=True)
    run_semantic  = st.checkbox("Sentence Transformers",    value=True)

    st.divider()
    st.subheader("Abstractive Methods")
    run_t5        = st.checkbox("T5-Small",       value=True)
    run_bart      = st.checkbox("BART",           value=False)  

    run_button = st.button("Run Comparison", type="primary", use_container_width=True)


if run_button and url:


    with st.spinner("Fetching and extracting article..."):
        raw_text = fetch_and_extract(url)

    if not raw_text:
        st.error("Could not extract content from URL")
        st.stop()

    cleaned  = clean_text(raw_text)
    baseline = first_sentences(cleaned, num_sentences)

    st.success(f"Extracted {len(cleaned.split())} words")


    with st.expander("Extracted Article Body"):
        st.write(cleaned)

    st.divider()


    sumy_methods = []
    if run_lsa:      sumy_methods.append(("LSA",           lambda t: sumy_summarize(t, LsaSummarizer,     num_sentences)))
    if run_lexrank:  sumy_methods.append(("LexRank",       lambda t: sumy_summarize(t, LexRankSummarizer,  num_sentences)))
    if run_textrank: sumy_methods.append(("TextRank",      lambda t: sumy_summarize(t, TextRankSummarizer, num_sentences)))
    if run_luhn:     sumy_methods.append(("Luhn",          lambda t: sumy_summarize(t, LuhnSummarizer,     num_sentences)))
    if run_sumbasic: sumy_methods.append(("SumBasic",      lambda t: sumy_summarize(t, SumBasicSummarizer, num_sentences)))
    if run_kl:       sumy_methods.append(("KL-Divergence", lambda t: sumy_summarize(t, KLSummarizer,       num_sentences)))


    st.header("Extractive Methods")

    for name, func in sumy_methods:
        with st.spinner(f"Running {name}..."):
            summary, elapsed, ram = measure(func, cleaned)

        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        with col1:
            st.markdown(f"### {name}")
        with col2:
            st.metric("Time", f"{elapsed}s")
        with col3:
            st.metric("RAM", f"{ram}MB")
        with col4:
            st.metric("Words", len(summary.split()))

        left, right = st.columns(2)
        with left:
            st.markdown("**Summary**")
            st.markdown(summary)
        with right:
            st.markdown("**Baseline (First N Sentences)**")
            st.markdown(baseline)

        st.divider()

    # PyTextRank
    if run_pytextrank:
        with st.spinner("Loading spaCy model..."):
            nlp_model = load_spacy()
        with st.spinner("Running PyTextRank..."):
            summary, elapsed, ram = measure(
                pytextrank_summarize, cleaned, num_sentences, nlp_model
            )

        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        with col1:
            st.markdown("### PyTextRank (spaCy)")
        with col2:
            st.metric("Time", f"{elapsed}s")
        with col3:
            st.metric("RAM", f"{ram}MB")
        with col4:
            st.metric("Words", len(summary.split()))

        left, right = st.columns(2)
        with left:
            st.markdown("**Summary**")
            st.markdown(summary)
        with right:
            st.markdown("**Baseline (First N Sentences)**")
            st.markdown(baseline)

        st.divider()

    # Sentence Transformers
    if run_semantic:
        with st.spinner("Loading Sentence Transformer model..."):
            st_model = load_sentence_transformer()
        with st.spinner("Running Semantic Summarization..."):
            summary, elapsed, ram = measure(
                semantic_summarize, cleaned, num_sentences, st_model
            )

        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        with col1:
            st.markdown("### Sentence Transformers")
        with col2:
            st.metric("Time", f"{elapsed}s")
        with col3:
            st.metric("RAM", f"{ram}MB")
        with col4:
            st.metric("Words", len(summary.split()))

        left, right = st.columns(2)
        with left:
            st.markdown("**Summary**")
            st.markdown(summary)
        with right:
            st.markdown("**Baseline (First N Sentences)**")
            st.markdown(baseline)

        st.divider()


    st.header("Abstractive Methods")

    if run_t5:
        with st.spinner("Loading T5 model..."):
            t5_model = load_t5()
        with st.spinner("Generating T5 summary..."):
            summary, elapsed, ram = measure(t5_summarize, cleaned, t5_model)

        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        with col1:
            st.markdown("### T5-Small")
        with col2:
            st.metric("Time", f"{elapsed}s")
        with col3:
            st.metric("RAM", f"{ram}MB")
        with col4:
            st.metric("Words", len(summary.split()))

        left, right = st.columns(2)
        with left:
            st.markdown("**Summary**")
            st.markdown(summary)
        with right:
            st.markdown("**Baseline**")
            st.markdown(baseline)

        st.divider()

    if run_bart:
        with st.spinner("Loading BART model (this takes a moment)..."):
            bart_model = load_bart()
        with st.spinner("Generating BART summary..."):
            summary, elapsed, ram = measure(bart_summarize, cleaned, bart_model)

        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        with col1:
            st.markdown("### BART")
        with col2:
            st.metric("Time", f"{elapsed}s")
        with col3:
            st.metric("RAM", f"{ram}MB")
        with col4:
            st.metric("Words", len(summary.split()))

        left, right = st.columns(2)
        with left:
            st.markdown("**Summary**")
            st.markdown(summary)
        with right:
            st.markdown("**Baseline**")
            st.markdown(baseline)

elif run_button and not url:
    st.warning("Please enter a URL first")

cleaned  = clean_text(raw_text)
st.session_state["article_text"] = cleaned  

if "article_text" in st.session_state:
    st.divider()
    st.header("🏷️ Category Extraction & NER Tagging")
    render_category_extractor(article_text=st.session_state["article_text"])