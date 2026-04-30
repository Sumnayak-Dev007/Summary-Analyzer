#!/usr/bin/env python3
"""
category_extractor.py
─────────────────────
Category extraction and NER classification using spaCy only.
Based on the original enhancement pipeline but simplified for Streamlit.
"""

import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import psutil
import streamlit as st
import spacy


# ── Quality filter constants (from original) ──────────────────────────────────

JUNK_STANDALONE = {
    "news", "look", "first", "things", "it", "this", "there", "after", "to",
    "in", "on", "at", "if", "the", "a", "an", "how", "why", "when", "what",
    "who", "where", "said", "says", "latest", "update", "updates", "new",
    "old", "big", "small", "good", "bad", "best", "top", "all", "more",
    "less", "own", "out", "up", "down", "off", "over",
}

NOISE_VERBS = {
    "say", "see", "tell", "know", "think", "go", "come", "be", "exit",
    "push", "visit", "want", "walk", "look", "rescind", "talk", "approve",
    "make", "do", "using", "taking", "getting", "giving", "having", "saying",
    "knowing", "thinking", "finding", "asking", "trying", "leaving",
    "following", "showing", "keeping", "calling", "working", "running",
    "moving", "building", "writing", "becoming", "opening", "cutting",
}

HONORIFIC_PREFIXES = frozenset({
    "dr", "dr.", "prof", "prof.", "professor",
    "mr", "mr.", "mrs", "mrs.", "ms", "ms.", "miss",
    "sir", "rev", "rev.", "gen", "gen.",
    "col", "col.", "lt", "lt.", "capt", "capt.",
    "sgt", "sgt.", "cpl", "cpl.",
})

PERSON_TITLE_ONLY_TOKENS = frozenset({
    "president", "vice", "prime", "minister", "senator", "governor",
    "mayor", "chief", "justice", "judge", "secretary", "chairman",
    "chairwoman", "director", "commissioner", "chancellor", "ambassador",
    "consul", "sheriff", "superintendent", "commander", "admiral",
    "general", "colonel", "lieutenant", "captain", "sergeant", "corporal",
    "private", "representative", "delegate", "councillor", "councilor",
    "alderman", "speaker", "treasurer", "comptroller", "auditor",
})

PERSON_HEADLINE_JUNK = frozenset({
    "news", "update", "updates", "video", "interview", "press",
    "photos", "photo", "latest", "breaking", "exclusive", "report",
    "statement", "conference", "briefing", "speech", "remarks",
    "announces", "says", "said", "advertisement", "live", "watch",
    "read", "today", "roundup", "recap", "preview", "profile",
})

PERSON_BLACKLIST_TOKENS = {"news", "update", "video", "photos", "advertisement", "live"}

ORG_HINT_WORDS = {
    "inc", "llc", "ltd", "plc", "corp", "company", "co", "group", "bank",
    "university", "ministry", "department", "agency", "government", "council",
    "committee", "authority",
}


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class Category:
    name: str
    score: float = 0.0
    source: str = ""  # trafilatura | spacy_ner
    entity_type: str = "unknown"
    lq_reasons: list[str] = field(default_factory=list)
    is_clean: bool = True


# ── Quality filter helpers (from original) ────────────────────────────────────

def _is_non_ascii(name: str) -> bool:
    if not name:
        return False
    return sum(1 for c in name if ord(c) > 127) / max(len(name), 1) > 0.50


def _is_verb_heavy(doc) -> bool:
    free_verbs = sum(1 for t in doc if t.pos_ in {"VERB", "AUX"} and t.ent_type_ == "")
    return (free_verbs / max(len(doc), 1)) > 0.40


def _format_reasons(name: str) -> list[str]:
    reasons: list[str] = []
    s = name.strip()
    if not s:
        reasons.append("empty_name")
        return reasons
    if len(s) <= 2:
        reasons.append("too_short")
    if re.fullmatch(r"[\W_]+", s):
        reasons.append("only_symbols")
    if re.fullmatch(r"\d+", s):
        reasons.append("only_digits")
    if re.search(r"(.)\1{4,}", s.lower()):
        reasons.append("repeated_chars")
    if len(re.findall(r"[^A-Za-z0-9\s]", s)) / max(len(s), 1) > 0.4:
        reasons.append("high_symbol_ratio")
    return reasons


def _quality_reasons(name: str, doc) -> list[str]:
    reasons = _format_reasons(name)
    if _is_non_ascii(name):
        reasons.append("non_english")
    if name.strip().lower() in JUNK_STANDALONE:
        reasons.append("junk_standalone")
    tokens = [t.text.lower() for t in doc if not t.is_punct and not t.is_space]
    if tokens and all(t in NOISE_VERBS for t in tokens):
        reasons.append("all_verb_tokens")
    if _is_verb_heavy(doc):
        reasons.append("verb_heavy")
    if re.match(r"^(how|why|when|what|who|where)\b", name.strip(), re.I):
        reasons.append("question_prefix")
    return reasons


def _clean_and_validate_person_name(raw: str) -> Optional[str]:
    """Validate if a name is actually a person name (from original)"""
    tokens = raw.strip().split()
    while tokens and tokens[0].rstrip(".").lower() in HONORIFIC_PREFIXES:
        tokens = tokens[1:]
    if not tokens or len(tokens) < 2 or len(tokens) > 4:
        return None
    lo = {t.lower().rstrip(".,") for t in tokens}
    if (lo - {"the", "a", "an"}) <= PERSON_TITLE_ONLY_TOKENS:
        return None
    if lo & PERSON_HEADLINE_JUNK or lo & PERSON_BLACKLIST_TOKENS:
        return None
    return " ".join(tokens)


def _looks_like_org_name(name: str) -> bool:
    """Check if name looks like an organization (from original)"""
    lo = re.sub(r"[^a-z0-9 ]+", " ", name.strip().lower()).strip()
    tokens = lo.split()
    if not tokens:
        return False
    return any(t in ORG_HINT_WORDS for t in tokens) or bool(re.fullmatch(r"[A-Z]{2,8}", name.strip()))


def _classify_with_spacy(name: str, nlp) -> tuple[str, float]:
    """
    Use spaCy NER to classify entity type.
    Returns (entity_type, confidence_score)
    """
    doc = nlp(name)
    
    # Check for named entities
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            # Validate person name
            if _clean_and_validate_person_name(ent.text):
                return ("person", 0.9)
        elif ent.label_ in ["ORG", "NORP"]:
            if _looks_like_org_name(ent.text):
                return ("organization", 0.85)
        elif ent.label_ in ["GPE", "LOC", "FAC"]:
            return ("place", 0.8)
        elif ent.label_ == "PRODUCT":
            return ("product", 0.75)
        elif ent.label_ == "EVENT":
            return ("event", 0.7)
    
    # If no entity found, try heuristic classification
    if _clean_and_validate_person_name(name):
        return ("person", 0.6)
    elif _looks_like_org_name(name):
        return ("organization", 0.6)
    
    return ("unknown", 0.0)


# ── Model loader ──────────────────────────────────────────────────────────────

@st.cache_resource
def _load_spacy():
    """Load spaCy model with caching"""
    for model_id in ["en_core_web_lg", "en_core_web_md", "en_core_web_sm"]:
        try:
            nlp = spacy.load(model_id)
            return nlp
        except OSError:
            continue
    st.error("No spaCy model found. Please ensure en_core_web_lg is installed.")
    return None


# ── Category extraction ───────────────────────────────────────────────────────

def _extract_from_trafilatura(raw_html: Optional[str], cleaned_text: str, nlp) -> list[Category]:
    """
    Extract keywords using trafilatura and classify with spaCy
    """
    candidates: list[Category] = []
    
    # Try trafilatura's keyword extraction
    if raw_html:
        try:
            from trafilatura import bare_extraction
            meta = bare_extraction(raw_html, include_comments=False, include_tables=False)
            kws = meta.get("keywords") or meta.get("tags") or [] if meta else []
            
            if kws:
                for i, term in enumerate(kws[:30]):
                    if not isinstance(term, str) or not term.strip():
                        continue
                    
                    term = term.strip()
                    
                    # Skip very short terms
                    if len(term) < 3:
                        continue
                    
                    # Classify with spaCy
                    entity_type, confidence = _classify_with_spacy(term, nlp)
                    
                    # Only keep if it's a meaningful entity type
                    if entity_type != "unknown" or confidence > 0:
                        candidates.append(Category(
                            name=term.title(),
                            score=round(1.0 - i * 0.03, 3),
                            source="trafilatura",
                            entity_type=entity_type,
                        ))
                if candidates:
                    return candidates
        except Exception:
            pass
    
    # Fallback: extract noun phrases using spaCy
    doc = nlp(cleaned_text[:50000])
    noun_phrases = []
    
    for chunk in doc.noun_chunks:
        phrase = chunk.text.strip()
        if 3 <= len(phrase) <= 50:
            noun_phrases.append(phrase)
    
    # Count frequencies
    freq = defaultdict(int)
    for phrase in noun_phrases:
        freq[phrase.lower()] += 1
    
    max_freq = max(freq.values()) if freq else 1
    
    for phrase, count in sorted(freq.items(), key=lambda x: -x[1])[:30]:
        entity_type, _ = _classify_with_spacy(phrase, nlp)
        
        # Only keep meaningful entities
        if entity_type != "unknown":
            candidates.append(Category(
                name=phrase.title(),
                score=round(count / max_freq, 3),
                source="frequency",
                entity_type=entity_type,
            ))
    
    return candidates


def _extract_spacy_ner(cleaned_text: str, nlp) -> list[Category]:
    """
    Run spaCy NER directly on the article text
    """
    doc = nlp(cleaned_text[:100000])
    
    label_map = {
        "PERSON": "person",
        "ORG": "organization",
        "GPE": "place",
        "LOC": "place",
        "FAC": "place",
        "NORP": "organization",
        "PRODUCT": "product",
        "EVENT": "event",
    }
    
    seen: dict[str, dict] = {}
    
    for ent in doc.ents:
        if ent.label_ not in label_map:
            continue
        
        text = ent.text.strip()
        
        # Basic filtering
        if len(text) < 3:
            continue
        
        # Skip if it fails validation for its type
        if ent.label_ == "PERSON" and not _clean_and_validate_person_name(text):
            continue
        elif ent.label_ in ["ORG", "NORP"] and not _looks_like_org_name(text):
            # Still keep if it's a recognized organization by spaCy
            if len(text) < 4:
                continue
        
        key = text.lower()
        entity_type = label_map[ent.label_]
        
        if key not in seen:
            seen[key] = {"name": text, "count": 1, "type": entity_type}
        else:
            seen[key]["count"] += 1
    
    # Convert to Category objects
    categories = []
    max_count = max([v["count"] for v in seen.values()]) if seen else 1
    
    for data in sorted(seen.values(), key=lambda x: -x["count"])[:25]:
        categories.append(Category(
            name=data["name"],
            score=round(data["count"] / max_count, 3),
            source="spacy_ner",
            entity_type=data["type"],
        ))
    
    return categories


def _apply_quality_filter(category: Category, nlp) -> bool:
    """
    Apply quality filter to a category (from original logic)
    Returns True if category should be kept, False if filtered out
    """
    doc = nlp(category.name)
    reasons = _quality_reasons(category.name, doc)
    
    if reasons:
        category.is_clean = False
        category.lq_reasons = reasons
        return False
    
    return True


# ── Main extraction function ─────────────────────────────────────────────────

def run_extraction(cleaned_text: str, raw_html: Optional[str]) -> dict:
    """
    Main extraction function - simplified version using only spaCy
    """
    proc = psutil.Process(os.getpid())
    ram0 = proc.memory_info().rss / 1024 / 1024
    t0 = time.monotonic()
    
    # Load spaCy
    nlp = _load_spacy()
    if nlp is None:
        st.error("spaCy model could not be loaded.")
        return {"categories": [], "elapsed_s": 0, "ram_mb": 0,
                "n_clean": 0, "n_lq": 0, "n_person": 0,
                "n_org": 0, "n_place": 0, "n_unknown": 0}
    
    # Extract from both sources
    spacy_cats = _extract_spacy_ner(cleaned_text, nlp)
    traf_cats = _extract_from_trafilatura(raw_html, cleaned_text, nlp)
    
    # Merge and deduplicate (prioritize spaCy)
    seen_names: set[str] = set()
    merged: list[Category] = []
    
    # Add spaCy categories first (they have higher quality)
    for cat in spacy_cats:
        key = cat.name.strip().lower()
        if key and key not in seen_names:
            seen_names.add(key)
            merged.append(cat)
    
    # Add trafilatura categories that aren't duplicates and pass quality filter
    for cat in traf_cats:
        key = cat.name.strip().lower()
        if key and key not in seen_names:
            seen_names.add(key)
            # Check quality before adding
            if _apply_quality_filter(cat, nlp):
                merged.append(cat)
            else:
                # Still add but mark as low quality for transparency
                merged.append(cat)
    
    # Apply final quality filter to all categories
    final_categories = []
    for cat in merged:
        if cat.is_clean:  # Already marked by filter
            final_categories.append(cat)
        else:
            # Double-check if it should be filtered
            if _apply_quality_filter(cat, nlp):
                final_categories.append(cat)
            else:
                final_categories.append(cat)  # Keep but marked as dirty
    
    elapsed = time.monotonic() - t0
    ram_used = proc.memory_info().rss / 1024 / 1024 - ram0
    
    return _build_result(final_categories, round(elapsed, 3), round(ram_used, 1))


def _build_result(categories: list, elapsed_s: float, ram_mb: float) -> dict:
    """Build result dictionary with statistics"""
    n_clean = sum(1 for c in categories if c.is_clean)
    n_lq = sum(1 for c in categories if not c.is_clean)
    
    return {
        "categories": categories,
        "elapsed_s": elapsed_s,
        "ram_mb": ram_mb,
        "n_clean": n_clean,
        "n_lq": n_lq,
        "n_person": sum(1 for c in categories if c.is_clean and c.entity_type == "person"),
        "n_org": sum(1 for c in categories if c.is_clean and c.entity_type == "organization"),
        "n_place": sum(1 for c in categories if c.is_clean and c.entity_type == "place"),
        "n_product": sum(1 for c in categories if c.is_clean and c.entity_type == "product"),
        "n_event": sum(1 for c in categories if c.is_clean and c.entity_type == "event"),
        "n_unknown": sum(1 for c in categories if c.is_clean and c.entity_type == "unknown"),
    }


# ── Rendering functions ───────────────────────────────────────────────────────

ENTITY_STYLE = {
    "person": "color:#388bfd;background:#1c3a6b;border:1px solid #388bfd",
    "organization": "color:#3fb950;background:#1a3a22;border:1px solid #3fb950",
    "place": "color:#d2a8ff;background:#2d1f5e;border:1px solid #d2a8ff",
    "product": "color:#f0883e;background:#3b2a1a;border:1px solid #f0883e",
    "event": "color:#f85149;background:#3b1a1a;border:1px solid #f85149",
    "unknown": "color:#8b949e;background:#1e2128;border:1px solid #30363d",
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
    """Render category results in Streamlit UI"""
    if not result or not result.get("categories"):
        st.info("No categories were extracted.")
        return
    
    # Display metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("⏱️ Time", f"{result['elapsed_s']}s")
    with col2:
        st.metric("💾 RAM", f"{result['ram_mb']:.0f} MB")
    with col3:
        st.metric("✅ Clean", result["n_clean"])
    with col4:
        st.metric("🚫 Filtered", result["n_lq"])
    with col5:
        st.metric("📊 Total", len(result["categories"]))
    
    st.markdown("---")
    
    # Display only clean categories
    clean = [c for c in result["categories"] if c.is_clean]
    dirty = [c for c in result["categories"] if not c.is_clean]
    
    if clean:
        st.markdown("### 🏷️ Extracted Categories")
        
        # Entity type breakdown
        type_counts = {
            "Person": result["n_person"],
            "Organization": result["n_org"],
            "Place": result["n_place"],
            "Product": result.get("n_product", 0),
            "Event": result.get("n_event", 0),
            "Unknown": result["n_unknown"],
        }
        
        # Show breakdown if there are results
        if any(type_counts.values()):
            with st.expander("📈 Entity Type Breakdown", expanded=False):
                import pandas as pd
                df_counts = pd.DataFrame([
                    {"Type": k, "Count": v}
                    for k, v in type_counts.items()
                    if v > 0
                ])
                st.dataframe(df_counts, hide_index=True, use_container_width=True)
        
        # Display categories table
        entity_color_map = {
            "person": "#388bfd",
            "organization": "#3fb950",
            "place": "#d2a8ff",
            "product": "#f0883e",
            "event": "#f85149",
            "unknown": "#8b949e",
        }
        
        rows = ""
        for cat in sorted(clean, key=lambda c: -c.score):
            clr = entity_color_map.get(cat.entity_type, "#8b949e")
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
            f' hilab</thead><tbody>{rows}</tbody></table>',
            unsafe_allow_html=True,
        )
        
        # Download button
        if clean:
            import pandas as pd
            csv = pd.DataFrame([{
                "name": c.name,
                "entity_type": c.entity_type,
                "score": c.score,
                "source": c.source,
            } for c in clean]).to_csv(index=False).encode()
            st.download_button(
                "📥 Download categories (CSV)",
                csv,
                file_name="categories.csv",
                mime="text/csv",
                use_container_width=False,
            )
    
    # Show filtered out categories
    if dirty:
        with st.expander(f"🚫 {len(dirty)} categories filtered out"):
            for cat in dirty:
                reason_text = " | ".join(cat.lq_reasons) if cat.lq_reasons else "quality_filter"
                st.markdown(
                    f'<span style="font-family:monospace;font-size:12px;color:#8b949e">'
                    f'<b>{cat.name}</b> — {reason_text}</span>',
                    unsafe_allow_html=True,
                )