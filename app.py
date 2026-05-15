import re
import os
import time
import math
import logging
from collections import Counter
from pathlib import Path

import nltk
import psutil
import streamlit as st
import trafilatura
from langdetect import DetectorFactory, LangDetectException, detect

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Article Summarizer",
    layout="wide",
)

# Make langdetect deterministic — same input always detects the same language.
DetectorFactory.seed = 0


# ── Paths ─────────────────────────────────────────────────────────────────────

_BASE_DIR = Path(__file__).resolve().parent
_DEFAULT_NLTK_DATA = _BASE_DIR / ".nltk_data"
_DEFAULT_LOCAL_MODELS = _BASE_DIR / "local-models"


# ── NLTK setup ────────────────────────────────────────────────────────────────

_NLTK_DATA_DIR = os.environ.get("NLTK_DATA", str(_DEFAULT_NLTK_DATA))
os.makedirs(_NLTK_DATA_DIR, exist_ok=True)
if _NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.insert(0, _NLTK_DATA_DIR)


def _ensure_nltk_data():
    """Ensure Punkt and stopwords are cached locally. Idempotent."""
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            try:
                nltk.download("punkt_tab", download_dir=_NLTK_DATA_DIR, quiet=True)
            except Exception:
                nltk.download("punkt", download_dir=_NLTK_DATA_DIR, quiet=True)

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        try:
            nltk.download("stopwords", download_dir=_NLTK_DATA_DIR, quiet=True)
        except Exception:
            pass


_ensure_nltk_data()


# ── Article fetching / cleaning ───────────────────────────────────────────────

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
    r"@abpnetwork|@gmail|@yahoo|"
    r"list of \d+ items|list \d+ of \d+|"
    r"^image id:|^title:)",
    re.IGNORECASE,
)

# Characters to strip from the start of a body after removing the title.
_LEADING_BODY_NOISE = ".!?:;,-—–·| \n\t\r\xa0'\"‘’“”"

_BYLINE_PATTERN = re.compile(
    r"\b[A-Z][a-z]+ [A-Z][a-z]+\d{1,2} \w{3,9} \d{4}",
    re.UNICODE,
)


def is_aggregation_page(body):
    """
    Detect topic/tag pages, RSS feeds, and category listings. These
    produce nonsense summaries because the input is multiple unrelated
    articles, not a single coherent narrative.
    """
    if not body or len(body) < 100:
        return False
    try:
        sentences = nltk.sent_tokenize(body, language="english")
    except LookupError:
        sentences = re.split(r"(?<=[.!?])\s+", body)
    if len(sentences) < 10:
        return False

    backref_words = {
        "he", "she", "it", "they", "his", "her", "their", "them",
        "this", "that", "these", "those", "such", "however", "but",
        "additionally", "moreover", "meanwhile", "later", "also",
    }
    backrefs = 0
    for sent in sentences:
        toks = sent.lower().split()[:1]
        if toks and toks[0].rstrip(",.;:") in backref_words:
            backrefs += 1
    if backrefs / len(sentences) < 0.15:
        return True

    if len(_BYLINE_PATTERN.findall(body)) >= 3:
        return True

    return False


def fetch_article(url):
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
        if _BYLINE_PATTERN.search(line):
            continue
        lines.append(line)
    cleaned = " ".join(lines) if lines else None
    return cleaned, downloaded


def smart_article_cleaning(text):
    """
    Split the extracted article into (body, title).
    The body is what every summarizer will consume — title is excluded.
    """
    text = (text or "").strip()
    if not text:
        return "", None

    title = None
    body = text

    # Strategy 1: title is on its own line at the start.
    if "\n" in text:
        first_line, rest = text.split("\n", 1)
        first_line = first_line.strip()
        if first_line and len(first_line) < 200 and not first_line.endswith((".", "!", "?")):
            title = first_line.strip("'\"‘’")
            body = rest.strip().lstrip(_LEADING_BODY_NOISE)
            return body, title

    # Strategy 2: title is a short prefix ending at first ". "
    first_period = text.find(". ")
    if 0 < first_period < 150:
        candidate_title = text[:first_period].strip()
        if (
            candidate_title
            and "." not in candidate_title
            and "!" not in candidate_title
            and "?" not in candidate_title
            and len(candidate_title) < 150
        ):
            title = candidate_title.strip("'\"‘’")
            body = text[first_period + 2:].strip().lstrip(_LEADING_BODY_NOISE)
            return body, title

    # Strategy 3: ALL CAPS heading followed by a colon.
    first_colon = text.find(":")
    if 0 < first_colon < 200:
        candidate = text[:first_colon].strip()
        letters = [c for c in candidate if c.isalpha()]
        if letters:
            upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
            if upper_ratio > 0.6 and len(candidate) < 200:
                title = candidate.strip("'\"‘’")
                body = text[first_colon + 1:].strip().lstrip(_LEADING_BODY_NOISE)
                return body, title

    # Strategy 4: quoted title at the start.
    if text.startswith(("'", '"', "‘", "“")):
        for closer in ["'", '"', "’", "”"]:
            end = text.find(closer, 1)
            if 0 < end < 200:
                title = text[1:end].strip()
                body = text[end + 1:].strip().lstrip(_LEADING_BODY_NOISE)
                return body, title

    return body, title


def _strip_title_from_body(body, title):
    """Safety net: strip the title if it leaked into the body."""
    if not body:
        return body
    body = body.lstrip(_LEADING_BODY_NOISE)
    if not title:
        return body
    title_clean = title.strip().strip("'\"‘’")
    if not title_clean:
        return body
    if body.lower().startswith(title_clean.lower()):
        body = body[len(title_clean):]
        body = body.lstrip(_LEADING_BODY_NOISE)
    return body


# ── Multilingual language support ─────────────────────────────────────────────

# ISO 639-1 → (punkt_language, stopwords_language).
# - punkt_language=None: use the multilingual regex sentence splitter.
# - stopwords_language=None: no stopword filtering (algorithm still works).
_LANGUAGE_MAP = {
    "en": ("english", "english"),
    "es": ("spanish", "spanish"),
    "fr": ("french", "french"),
    "de": ("german", "german"),
    "it": ("italian", "italian"),
    "pt": ("portuguese", "portuguese"),
    "nl": ("dutch", "dutch"),
    "ru": ("russian", "russian"),
    "sv": ("swedish", "swedish"),
    "no": ("norwegian", "norwegian"),
    "da": ("danish", "danish"),
    "fi": ("finnish", "finnish"),
    "tr": ("turkish", "turkish"),
    "el": ("greek", "greek"),
    "pl": ("polish", None),
    "cs": ("czech", None),
    "et": ("estonian", None),
    "sl": ("slovene", "slovene"),
    "hi": (None, "hindi"),
    "bn": (None, "bengali"),
    "ar": (None, "arabic"),
    "ne": (None, "nepali"),
    "id": (None, "indonesian"),
    "he": (None, "hebrew"),
    "ro": (None, "romanian"),
    "hu": (None, "hungarian"),
    "ca": (None, "catalan"),
    "zh-cn": (None, "chinese"),
    "zh-tw": (None, "chinese"),
}

_NO_SPACE_LANGUAGES = {"zh-cn", "zh-tw", "ja", "th", "km", "lo", "my"}

# Sentence terminators across major writing systems.
_MULTILINGUAL_SENTENCE_END = re.compile(r"(?<=[.!?।。!?؟።။])\s+")


def _fallback_sentence_split(text):
    """Generic regex sentence splitter for languages without Punkt models."""
    if not text:
        return []
    raw = _MULTILINGUAL_SENTENCE_END.split(text.strip())
    return [s.strip() for s in raw if s.strip()]


# ── Stopwords ─────────────────────────────────────────────────────────────────

_ENGLISH_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "of", "in", "on", "at", "to", "for", "with", "by", "from", "as",
    "this", "that", "these", "those", "it", "its", "they", "them",
    "i", "you", "he", "she", "we", "us", "him", "her", "their", "our",
    "not", "no", "so", "if", "then", "than", "also", "just", "only",
    "will", "would", "should", "could", "may", "might", "can", "shall",
    "about", "into", "out", "up", "down", "over", "under", "again",
    "any", "all", "some", "such", "what", "which", "who", "whom",
    "when", "where", "why", "how", "there", "here", "more", "most",
    "very", "much", "many", "one", "two", "said", "says", "say",
})

# Lazy cache: nltk_language_name -> frozenset of stopwords.
_STOPWORD_CACHE = {"english": _ENGLISH_STOPWORDS}


def _get_stopwords(nltk_language):
    """Return a frozenset of stopwords for the given language, or empty set."""
    if nltk_language is None:
        return frozenset()
    if nltk_language in _STOPWORD_CACHE:
        return _STOPWORD_CACHE[nltk_language]
    try:
        from nltk.corpus import stopwords
        words = frozenset(stopwords.words(nltk_language))
        _STOPWORD_CACHE[nltk_language] = words
        return words
    except Exception:
        _STOPWORD_CACHE[nltk_language] = frozenset()
        return _STOPWORD_CACHE[nltk_language]


def _detect_language(text):
    """
    Detect language and map to (iso_code, punkt_lang, stopwords_lang).
    Falls back to English on detection failure.
    """
    sample = (text or "")[:2000].strip()
    if len(sample) < 50:
        return "en", "english", "english"
    try:
        iso = detect(sample)
    except LangDetectException:
        return "en", "english", "english"

    punkt_lang, stopwords_lang = _LANGUAGE_MAP.get(iso, (None, None))
    return iso, punkt_lang, stopwords_lang


# ── Algorithm 1: PlainTextRankSummarizer (multilingual + MMR) ─────────────────

class PlainTextRankSummarizer:
    """TextRank with NLTK preprocessing, MMR diversity selection, position bias, and multilingual support."""

    VERSION = "plain-textrank-v3-multilingual"

    def __init__(
        self,
        *,
        damping=0.85,
        max_iterations=40,
        min_delta=1e-4,
        sentence_limit=6,
        min_sentence_words=5,
        max_summary_chars=700,
        mmr_lambda=0.50,
        position_boost=(1.5, 1.3, 1.15),
    ):
        self.damping = damping
        self.max_iterations = max_iterations
        self.min_delta = min_delta
        self.sentence_limit = sentence_limit
        self.min_sentence_words = min_sentence_words
        self.max_summary_chars = max_summary_chars
        self.mmr_lambda = mmr_lambda
        self.position_boost = position_boost

    def summarize(self, text):
        iso, punkt_lang, stopwords_lang = _detect_language(text)
        stopwords_set = _get_stopwords(stopwords_lang)

        sentences = self._split_sentences(text, iso, punkt_lang)
        if not sentences:
            return ""
        if len(sentences) == 1:
            return sentences[0][: self.max_summary_chars].strip()

        tokens = [self._tokenize(s, stopwords_set) for s in sentences]
        graph = self._build_similarity_matrix(tokens)
        ranks = self._page_rank(graph)

        # Position bias: news ledes carry the core facts.
        for i in range(min(len(self.position_boost), len(ranks))):
            ranks[i] *= self.position_boost[i]

        # MMR selection: balances rank against diversity from already-chosen sentences.
        selected_indices = self._mmr_select(ranks, graph)

        # Restore document order for readability.
        selected_indices.sort()
        selected_sentences = [sentences[idx] for idx in selected_indices]
        return self._truncate(selected_sentences)

    def _mmr_select(self, ranks, sim_matrix):
        """Greedy Maximal Marginal Relevance selection."""
        n = len(ranks)
        if n == 0:
            return []
        if self.sentence_limit >= n:
            return list(range(n))

        first = max(range(n), key=lambda i: ranks[i])
        selected = [first]
        remaining = set(range(n)) - {first}

        while remaining and len(selected) < self.sentence_limit:
            def mmr_score(i):
                relevance = ranks[i]
                max_sim = max(sim_matrix[i][j] for j in selected)
                return self.mmr_lambda * relevance - (1.0 - self.mmr_lambda) * max_sim

            best = max(remaining, key=mmr_score)
            selected.append(best)
            remaining.remove(best)

        return selected

    def _truncate(self, sentences):
        parts = []
        used = 0
        for sent in sentences:
            add_len = len(sent) + (1 if parts else 0)
            if used + add_len <= self.max_summary_chars:
                parts.append(sent)
                used += add_len

        if parts:
            return " ".join(parts).strip()
        if sentences:
            return min(sentences, key=len).strip()
        return ""

    def _split_sentences(self, text, iso, punkt_lang):
        if not text:
            return []

        if punkt_lang is not None:
            try:
                raw = nltk.sent_tokenize(text.strip(), language=punkt_lang)
            except LookupError:
                raw = _fallback_sentence_split(text)
        else:
            raw = _fallback_sentence_split(text)

        out = []
        for sent in raw:
            normalized = re.sub(r"\s+", " ", sent).strip()
            if self._passes_length_filter(normalized, iso):
                out.append(normalized)
        return out

    def _passes_length_filter(self, sentence, iso):
        if iso in _NO_SPACE_LANGUAGES:
            return len(sentence) >= self.min_sentence_words * 5
        return len(sentence.split()) >= self.min_sentence_words

    def _tokenize(self, sentence, stopwords_set):
        text = sentence.lower().replace("\u2019", "'").replace("\u2018", "'")
        raw = re.findall(r"\w+(?:[-'.]\w+)*", text, flags=re.UNICODE)
        cleaned = []
        for tok in raw:
            tok = tok.strip("'-.")
            if tok and tok not in stopwords_set:
                cleaned.append(tok)
        return cleaned

    def _build_similarity_matrix(self, tokenized_sentences):
        count = len(tokenized_sentences)
        graph = [[0.0 for _ in range(count)] for _ in range(count)]
        for i in range(count):
            for j in range(count):
                if i == j:
                    continue
                graph[i][j] = self._sentence_similarity(tokenized_sentences[i], tokenized_sentences[j])
        return graph

    def _sentence_similarity(self, left, right):
        if not left or not right:
            return 0.0
        left_counter = Counter(left)
        right_counter = Counter(right)
        common = set(left_counter.keys()) & set(right_counter.keys())
        if not common:
            return 0.0
        numerator = sum(left_counter[token] * right_counter[token] for token in common)
        denominator = math.sqrt(sum(v * v for v in left_counter.values())) * math.sqrt(
            sum(v * v for v in right_counter.values())
        )
        if denominator == 0:
            return 0.0
        return numerator / denominator

    def _page_rank(self, graph):
        count = len(graph)
        ranks = [1.0 / count for _ in range(count)]
        outbound = [sum(graph[i]) for i in range(count)]

        for _ in range(self.max_iterations):
            new_ranks = [(1.0 - self.damping) / count for _ in range(count)]
            for j in range(count):
                if outbound[j] == 0:
                    continue
                contribution = ranks[j] / outbound[j]
                for i in range(count):
                    if graph[j][i] > 0:
                        new_ranks[i] += self.damping * graph[j][i] * contribution
            delta = sum(abs(new_ranks[i] - ranks[i]) for i in range(count))
            ranks = new_ranks
            if delta < self.min_delta:
                break
        return ranks


# ── Algorithm 2: NltkTextRankSummarizer ───────────────────────────────────────

class NltkTextRankSummarizer:
    """TextRank with NLTK sentence splitting, hyphen-aware tokenizer, English stopwords."""

    VERSION = "nltk-textrank-v1"

    def __init__(
        self,
        *,
        damping=0.85,
        max_iterations=40,
        min_delta=1e-4,
        sentence_limit=3,
        min_sentence_words=5,
        max_summary_chars=600,
    ):
        self.damping = damping
        self.max_iterations = max_iterations
        self.min_delta = min_delta
        self.sentence_limit = sentence_limit
        self.min_sentence_words = min_sentence_words
        self.max_summary_chars = max_summary_chars

    def summarize(self, text):
        sentences = self._split_sentences(text)
        if not sentences:
            return ""
        if len(sentences) == 1:
            return self._truncate(sentences)

        tokens = [self._tokenize(s) for s in sentences]
        valid_indices = [i for i, t in enumerate(tokens) if t]

        if len(valid_indices) <= 1:
            return self._truncate(sentences[: self.sentence_limit])

        valid_tokens = [tokens[i] for i in valid_indices]
        graph = self._build_similarity_matrix(valid_tokens)
        ranks = self._page_rank(graph)

        ranked_pairs = sorted(zip(valid_indices, ranks), key=lambda p: p[1], reverse=True)
        chosen_indices = sorted(idx for idx, _ in ranked_pairs[: self.sentence_limit])

        selected_sentences = [sentences[idx] for idx in chosen_indices]
        return self._truncate(selected_sentences)

    def _truncate(self, sentences):
        parts = []
        used = 0
        for sent in sentences:
            add_len = len(sent) + (1 if parts else 0)
            if used + add_len <= self.max_summary_chars:
                parts.append(sent)
                used += add_len

        if parts:
            return " ".join(parts).strip()
        if sentences:
            return min(sentences, key=len).strip()
        return ""

    def _split_sentences(self, text):
        if not text:
            return []
        raw = nltk.sent_tokenize(text.strip(), language="english")
        out = []
        for sent in raw:
            normalized = re.sub(r"\s+", " ", sent).strip()
            if len(normalized.split()) >= self.min_sentence_words:
                out.append(normalized)
        return out

    def _tokenize(self, sentence):
        text = sentence.lower().replace("\u2019", "'").replace("\u2018", "'")
        raw = re.findall(r"\w+(?:[-'.]\w+)*", text, flags=re.UNICODE)
        cleaned = []
        for tok in raw:
            tok = tok.strip("'-.")
            if tok and tok not in _ENGLISH_STOPWORDS:
                cleaned.append(tok)
        return cleaned

    def _build_similarity_matrix(self, tokens):
        n = len(tokens)
        graph = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                sim = self._sentence_similarity(tokens[i], tokens[j])
                graph[i][j] = sim
                graph[j][i] = sim
        return graph

    def _sentence_similarity(self, left, right):
        if not left or not right:
            return 0.0
        lc, rc = Counter(left), Counter(right)
        common = set(lc) & set(rc)
        if not common:
            return 0.0
        numerator = sum(lc[t] * rc[t] for t in common)
        ln = math.sqrt(sum(v * v for v in lc.values()))
        rn = math.sqrt(sum(v * v for v in rc.values()))
        denom = ln * rn
        if denom == 0:
            return 0.0
        return numerator / denom

    def _page_rank(self, graph):
        n = len(graph)
        ranks = [1.0 / n] * n
        outbound = [sum(graph[i]) for i in range(n)]
        for _ in range(self.max_iterations):
            new_ranks = [(1.0 - self.damping) / n] * n
            for j in range(n):
                if outbound[j] == 0:
                    continue
                contribution = ranks[j] / outbound[j]
                for i in range(n):
                    if graph[j][i] > 0:
                        new_ranks[i] += self.damping * graph[j][i] * contribution
            delta = sum(abs(new_ranks[i] - ranks[i]) for i in range(n))
            ranks = new_ranks
            if delta < self.min_delta:
                break
        return ranks


# ── Algorithm 3: MiniLmTextRankSummarizer ─────────────────────────────────────

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_LOCAL_MODELS_DIR = Path(os.environ.get("LOCAL_MODELS_DIR", str(_DEFAULT_LOCAL_MODELS)))
_LOCAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)


@st.cache_resource
def _get_minilm_model():
    """
    Load MiniLM once per Streamlit session. On first run, downloads from
    HuggingFace and saves a local snapshot under ./local-models/. Subsequent
    runs (and subsequent app starts) load from the local snapshot — no
    network calls, no re-downloads.
    """
    from sentence_transformers import SentenceTransformer

    local_path = _LOCAL_MODELS_DIR / "all-MiniLM-L6-v2"

    if local_path.exists() and any(local_path.iterdir()):
        logger.info("Loading MiniLM from local snapshot at %s", local_path)
        return SentenceTransformer(str(local_path), device="cpu")

    logger.warning(
        "No local MiniLM snapshot at %s; downloading %s from HuggingFace hub "
        "and saving to local snapshot for future runs.",
        local_path, _MODEL_NAME,
    )
    model = SentenceTransformer(_MODEL_NAME, device="cpu")
    try:
        model.save(str(local_path))
        logger.info("Saved MiniLM snapshot to %s", local_path)
    except Exception as exc:
        logger.warning("Failed to save MiniLM snapshot to %s: %s", local_path, exc)
    return model


class MiniLmTextRankSummarizer:
    """TextRank with MiniLM sentence embeddings for semantic similarity."""

    VERSION = "minilm-textrank-v1"

    def __init__(
        self,
        *,
        damping=0.85,
        max_iterations=40,
        min_delta=1e-4,
        sentence_limit=3,
        min_sentence_words=5,
        max_summary_chars=600,
        similarity_floor=0.1,
    ):
        self.damping = damping
        self.max_iterations = max_iterations
        self.min_delta = min_delta
        self.sentence_limit = sentence_limit
        self.min_sentence_words = min_sentence_words
        self.max_summary_chars = max_summary_chars
        self.similarity_floor = similarity_floor

    def summarize(self, text):
        sentences = self._split_sentences(text)
        if not sentences:
            return ""
        if len(sentences) == 1:
            return sentences[0].strip()
        if len(sentences) <= self.sentence_limit:
            return self._truncate(sentences)

        graph = self._build_similarity_matrix(sentences)
        ranks = self._page_rank(graph)
        ranked_indices = sorted(range(len(sentences)), key=lambda idx: ranks[idx], reverse=True)
        selected_indices = sorted(ranked_indices[:self.sentence_limit])

        selected_sentences = [sentences[i] for i in selected_indices]
        return self._truncate(selected_sentences)

    def _truncate(self, sentences):
        parts = []
        used = 0
        for sent in sentences:
            add_len = len(sent) + (1 if parts else 0)
            if used + add_len <= self.max_summary_chars:
                parts.append(sent)
                used += add_len

        if parts:
            return " ".join(parts).strip()
        if sentences:
            return min(sentences, key=len).strip()
        return ""

    def _split_sentences(self, text):
        if not text:
            return []
        raw = nltk.sent_tokenize(text.strip(), language="english")
        out = []
        for sent in raw:
            normalized = re.sub(r"\s+", " ", sent).strip()
            if len(normalized.split()) >= self.min_sentence_words:
                out.append(normalized)
        return out

    def _build_similarity_matrix(self, sentences):
        import numpy as np

        model = _get_minilm_model()
        embeddings = model.encode(
            sentences,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        sim_matrix = embeddings @ embeddings.T
        np.fill_diagonal(sim_matrix, 0.0)
        sim_matrix = np.where(sim_matrix < self.similarity_floor, 0.0, sim_matrix)
        return sim_matrix.tolist()

    def _page_rank(self, graph):
        n = len(graph)
        ranks = [1.0 / n] * n
        outbound = [sum(graph[i]) for i in range(n)]
        for _ in range(self.max_iterations):
            new_ranks = [(1.0 - self.damping) / n] * n
            for j in range(n):
                if outbound[j] == 0:
                    continue
                contribution = ranks[j] / outbound[j]
                for i in range(n):
                    if graph[j][i] > 0:
                        new_ranks[i] += self.damping * graph[j][i] * contribution
            delta = sum(abs(new_ranks[i] - ranks[i]) for i in range(n))
            ranks = new_ranks
            if delta < self.min_delta:
                break
        return ranks


# ── Algorithm registry ────────────────────────────────────────────────────────

ALGORITHMS = {
    "Plain TextRank (multilingual + MMR)": PlainTextRankSummarizer,
    "NLTK TextRank (Punkt + stopwords)": NltkTextRankSummarizer,
    "MiniLM TextRank (semantic embeddings)": MiniLmTextRankSummarizer,
}


def run_summarizer(
    algorithm_name,
    body,
    *,
    sentence_limit,
    min_sentence_words,
    max_summary_chars,
):
    """Run a single summarizer and capture benchmarks."""
    summarizer_cls = ALGORITHMS[algorithm_name]
    summarizer = summarizer_cls(
        sentence_limit=sentence_limit,
        min_sentence_words=min_sentence_words,
        max_summary_chars=max_summary_chars,
    )

    proc = psutil.Process(os.getpid())
    ram_before = proc.memory_info().rss / 1024 / 1024
    t0 = time.monotonic()
    try:
        summary = summarizer.summarize(body)
        error = None
    except Exception as exc:
        logger.exception("Summarizer %s failed", algorithm_name)
        summary = ""
        error = str(exc)
    elapsed = time.monotonic() - t0
    ram_after = proc.memory_info().rss / 1024 / 1024
    ram_delta = ram_after - ram_before

    return {
        "algorithm": algorithm_name,
        "version": summarizer.VERSION,
        "summary": summary,
        "summary_chars": len(summary),
        "summary_words": len(summary.split()) if summary else 0,
        "elapsed_s": round(elapsed, 3),
        "ram_before_mb": round(ram_before, 1),
        "ram_after_mb": round(ram_after, 1),
        "ram_delta_mb": round(ram_delta, 1),
        "error": error,
    }


# ── UI ────────────────────────────────────────────────────────────────────────

st.title("Article Summarizer — Algorithm Comparison")

with st.sidebar:
    st.header("Settings")

    url = st.text_input("Article URL", placeholder="https://...")

    st.divider()
    st.subheader("Algorithms to run")

    selected_algorithms = []
    if st.checkbox("Plain TextRank (multilingual + MMR)", value=True, key="alg_plain"):
        selected_algorithms.append("Plain TextRank (multilingual + MMR)")
    if st.checkbox("NLTK TextRank (Punkt + stopwords)", value=True, key="alg_nltk"):
        selected_algorithms.append("NLTK TextRank (Punkt + stopwords)")
    if st.checkbox("MiniLM TextRank (semantic embeddings)", value=True, key="alg_minilm"):
        selected_algorithms.append("MiniLM TextRank (semantic embeddings)")

    st.divider()
    st.subheader("Summarization Settings")

    sentence_limit = st.slider(
        "Number of sentences", 2, 15, 6,
        help="How many sentences each summarizer should pick.",
    )
    min_sent_words = st.slider(
        "Min sentence length (words)", 3, 20, 5,
        help="Sentences shorter than this are skipped during ranking.",
    )
    max_summary_chars = st.slider(
        "Max summary length (chars)", 200, 2000, 700, step=50,
        help="Soft cap on the final summary length.",
    )

    st.divider()
    btn_summarize = st.button("Summarize Article", type="primary", width="stretch")


# ── Fetch article (cached in session_state) ───────────────────────────────────

def get_article(url):
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


# ── Run summarization on click ────────────────────────────────────────────────

if btn_summarize:
    if not url:
        st.warning("Enter a URL first.")
    elif not selected_algorithms:
        st.warning("Select at least one algorithm to run.")
    else:
        cleaned, _ = get_article(url)
        if not cleaned:
            st.error("Could not extract content from this URL.")
        else:
            body_text, article_title = smart_article_cleaning(cleaned)
            body_text = _strip_title_from_body(body_text, article_title)

            results = []
            for algo_name in selected_algorithms:
                with st.spinner(f"Running {algo_name}..."):
                    result = run_summarizer(
                        algo_name,
                        body_text,
                        sentence_limit=sentence_limit,
                        min_sentence_words=min_sent_words,
                        max_summary_chars=max_summary_chars,
                    )
                results.append(result)

            st.session_state["results"] = results
            st.session_state["body_text"] = body_text
            st.session_state["article_title"] = article_title


# ── Render results ────────────────────────────────────────────────────────────

if "results" in st.session_state:
    results = st.session_state["results"]
    body_text = st.session_state["body_text"]
    article_title = st.session_state.get("article_title")

    if article_title:
        st.info(f"**Article title:** {article_title}")

    # Show detected language alongside body stats.
    try:
        detected_iso, _, _ = _detect_language(body_text)
    except Exception:
        detected_iso = "?"
    st.caption(
        f"Body length: {len(body_text):,} chars / {len(body_text.split()):,} words. "
        f"Detected language: `{detected_iso}`. "
        "Every algorithm receives this body (title excluded)."
    )

    st.header("Results")

    for i, r in enumerate(results):
        if i > 0:
            st.divider()

        st.subheader(f"{r['algorithm']}")
        st.caption(f"version: `{r['version']}`")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Time", f"{r['elapsed_s']}s")
        c2.metric("RAM delta", f"{r['ram_delta_mb']} MB")
        c3.metric("Chars", r["summary_chars"])
        c4.metric("Words", r["summary_words"])

        if r["error"]:
            st.error(r["error"])
        elif not r["summary"]:
            st.warning("Empty summary produced.")
        else:
            st.markdown("**Summary:**")
            st.markdown(r["summary"])

    with st.expander("Show full article body (input to every summarizer)"):
        st.write(body_text)