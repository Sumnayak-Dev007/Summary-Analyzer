#!/usr/bin/env python3
"""
category_extractor.py
─────────────────────
Category extraction using spaCy NER only - no frequency-based garbage.
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

PERSON_BLACKLIST = {"news", "update", "video", "photos", "advertisement", "live"}

ORG_HINT_WORDS = {
    "inc", "llc", "ltd", "plc", "corp", "company", "co", "group", "bank",
    "university", "ministry", "department", "agency", "government", "council",
    "committee", "authority", "technologies", "systems", "solutions", "labs",
}

# Product-related terms to filter out
PRODUCT_NOISE = {
    "battery", "mah", "display", "screen", "pixel", "refresh rate", "hz",
    "gb", "ram", "rom", "storage", "processor", "snapdragon", "dimensity",
    "camera", "megapixel", "mp", "ultra wide", "telephoto", "zoom",
    "charging", "watts", "w", "fast charging", "wireless",
    "bluetooth", "wifi", "5g", "lte", "nfc", "usb", "type-c",
    "inch", "nit", "color", "hdr", "dolby", "speaker", "audio",
    "price", "discount", "offer", "sale", "emi", "rupees", "rs",
}


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class Category:
    name: str
    score: float = 0.0
    source: str = "spacy_ner"
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
    return reasons


def _is_product_noise(name: str) -> bool:
    """Check if the entity is actually a product spec/feature rather than a brand"""
    name_lower = name.lower()
    for noise_term in PRODUCT_NOISE:
        if noise_term in name_lower:
            return True
    # Check for measurements (numbers with units)
    if re.search(r'\d+\s*(gb|tb|mb|kb|ghz|mhz|hz|mah|w|mm|cm|inch|"%")', name_lower):
        return True
    # Check for specs with numbers
    if re.search(r'\d+(k|p|x)\s*(display|screen|resolution)?', name_lower):
        return True
    return False


def _quality_reasons(name: str, doc) -> list[str]:
    reasons = _format_reasons(name)
    if _is_non_ascii(name):
        reasons.append("non_english")
    if name.strip().lower() in JUNK_STANDALONE:
        reasons.append("junk_standalone")
    if _is_product_noise(name):
        reasons.append("product_spec")
    tokens = [t.text.lower() for t in doc if not t.is_punct and not t.is_space]
    if tokens and all(t in NOISE_VERBS for t in tokens):
        reasons.append("all_verb_tokens")
    if _is_verb_heavy(doc):
        reasons.append("verb_heavy")
    return reasons


def _clean_and_validate_person_name(raw: str) -> Optional[str]:
    """Validate if a name is actually a person name"""
    # Skip if it contains numbers
    if re.search(r'\d', raw):
        return None
    
    # Skip if it looks like a product spec
    if _is_product_noise(raw):
        return None
    
    tokens = raw.strip().split()
    while tokens and tokens[0].rstrip(".").lower() in HONORIFIC_PREFIXES:
        tokens = tokens[1:]
    if not tokens or len(tokens) < 2 or len(tokens) > 4:
        return None
    lo = {t.lower().rstrip(".,") for t in tokens}
    if (lo - {"the", "a", "an"}) <= PERSON_TITLE_ONLY_TOKENS:
        return None
    if lo & PERSON_HEADLINE_JUNK or lo & PERSON_BLACKLIST:
        return None
    return " ".join(tokens)


def _looks_like_org_name(name: str) -> bool:
    """Check if name looks like an organization"""
    # Skip if it contains numbers or looks like a product
    if re.search(r'\d', name) or _is_product_noise(name):
        return False
    
    lo = re.sub(r"[^a-z0-9 ]+", " ", name.strip().lower()).strip()
    tokens = lo.split()
    if not tokens:
        return False
    
    # Check for organization indicators
    org_indicators = ORG_HINT_WORDS
    has_org_indicator = any(t in org_indicators for t in tokens)
    
    # Check for all-caps acronym (e.g., "NASA", "IBM")
    is_acronym = bool(re.fullmatch(r"[A-Z]{2,8}", name.strip()))
    
    # Check for proper capitalization pattern (e.g., "Apple Inc.")
    has_proper_case = all(w[0].isupper() for w in tokens if len(w) > 1)
    
    return has_org_indicator or is_acronym or has_proper_case


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


# ── NER Extraction (the ONLY source - no frequency fallback) ─────────────────

def extract_entities_with_spacy(cleaned_text: str, nlp) -> list[Category]:
    """
    Extract named entities using spaCy NER only.
    This is the ONLY source of categories - no frequency-based garbage.
    """
    doc = nlp(cleaned_text[:100000])  # Limit text length
    
    label_map = {
        "PERSON": "person",
        "ORG": "organization",
        "GPE": "place",
        "LOC": "place",
        "FAC": "place",
        "NORP": "organization",
        "PRODUCT": "product",
        "EVENT": "event",
        "WORK_OF_ART": "product",
        "LAW": "unknown",
        "DATE": "unknown",
        "TIME": "unknown",
        "PERCENT": "unknown",
        "MONEY": "unknown",
        "QUANTITY": "unknown",
        "ORDINAL": "unknown",
        "CARDINAL": "unknown",
    }
    
    seen: dict[str, dict] = {}
    
    for ent in doc.ents:
        # Skip entity types we don't care about
        if ent.label_ not in label_map:
            continue
        
        # Skip low-confidence entity types
        if label_map[ent.label_] == "unknown":
            continue
        
        text = ent.text.strip()
        
        # Basic filtering
        if len(text) < 3:
            continue
        
        # Skip product specs and noise
        if _is_product_noise(text):
            continue
        
        # Validate based on entity type
        if ent.label_ == "PERSON":
            validated = _clean_and_validate_person_name(text)
            if not validated:
                continue
            text = validated
            entity_type = "person"
        elif ent.label_ in ["ORG", "NORP"]:
            if not _looks_like_org_name(text):
                # Only keep if it has organization characteristics
                if len(text) < 4:
                    continue
            entity_type = "organization"
        elif ent.label_ in ["GPE", "LOC", "FAC"]:
            # Skip if it looks like a product spec
            if _is_product_noise(text):
                continue
            entity_type = "place"
        elif ent.label_ == "PRODUCT":
            # Only keep product names that look like brands, not specs
            if _is_product_noise(text):
                continue
            # Skip generic product names
            if text.lower() in ["phone", "tablet", "laptop", "device", "gadget"]:
                continue
            entity_type = "product"
        elif ent.label_ == "EVENT":
            entity_type = "event"
        else:
            continue
        
        key = text.lower()
        
        if key not in seen:
            seen[key] = {"name": text, "count": 1, "type": entity_type}
        else:
            seen[key]["count"] += 1
    
    # Convert to Category objects
    categories = []
    max_count = max([v["count"] for v in seen.values()]) if seen else 1
    
    for data in sorted(seen.values(), key=lambda x: -x["count"])[:50]:
        categories.append(Category(
            name=data["name"],
            score=round(data["count"] / max_count, 3),
            source="spacy_ner",
            entity_type=data["type"],
            is_clean=True,
        ))
    
    return categories


# ── Main extraction function ─────────────────────────────────────────────────

def run_extraction(cleaned_text: str, raw_html: Optional[str] = None) -> dict:
    """
    Main extraction function - uses ONLY spaCy NER, no frequency fallback.
    raw_html parameter is kept for compatibility but not used.
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
    
    # Extract entities using spaCy NER only
    categories = extract_entities_with_spacy(cleaned_text, nlp)
    
    # Apply quality filter to all categories
    final_categories = []
    for cat in categories:
        doc = nlp(cat.name)
        reasons = _quality_reasons(cat.name, doc)
        if reasons:
            cat.is_clean = False
            cat.lq_reasons = reasons
        final_categories.append(cat)
    
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
        st.info("No named entities were found in the article.")
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
        st.markdown("### 🏷️ Extracted Named Entities")
        
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
                f'<td style="padding:7px 12px;min-width:130px">{_bar(cat.score, clr)}</td>'
                f'</tr>'
            )
        
        st.markdown(
            f'<table style="width:100%;border-collapse:collapse;border-radius:8px;overflow:hidden">'
            f'<thead><tr style="background:#21262d">'
            f'<th style="padding:8px 12px;text-align:left;font-size:11px;'
            f'color:#8b949e;font-family:monospace;text-transform:uppercase">Entity</th>'
            f'<th style="padding:8px 12px;text-align:left;font-size:11px;'
            f'color:#8b949e;font-family:monospace;text-transform:uppercase">Type</th>'
            f'<th style="padding:8px 12px;text-align:left;font-size:11px;'
            f'color:#8b949e;font-family:monospace;text-transform:uppercase">Frequency Score</th>'
            f'<tr></thead><tbody>{rows}</tbody></table>',
            unsafe_allow_html=True,
        )
        
        # Download button
        if clean:
            import pandas as pd
            csv = pd.DataFrame([{
                "name": c.name,
                "entity_type": c.entity_type,
                "frequency_score": c.score,
            } for c in clean]).to_csv(index=False).encode()
            st.download_button(
                "📥 Download entities (CSV)",
                csv,
                file_name="named_entities.csv",
                mime="text/csv",
                use_container_width=False,
            )
    
    # Show filtered out categories
    if dirty:
        with st.expander(f"🚫 {len(dirty)} entities filtered out (product specs, noise, etc.)"):
            for cat in dirty:
                reason_text = " | ".join(cat.lq_reasons) if cat.lq_reasons else "quality_filter"
                st.markdown(
                    f'<span style="font-family:monospace;font-size:12px;color:#8b949e">'
                    f'<b>{cat.name}</b> — {reason_text}</span>',
                    unsafe_allow_html=True,
                )