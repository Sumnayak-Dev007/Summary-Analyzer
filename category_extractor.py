#!/usr/bin/env python3
"""
category_extractor.py
─────────────────────
Category extraction using spaCy NER with aggressive filtering.
Shows raw extracted categories, cleaned categories, and discarded categories.
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


# ── Quality filter constants ──────────────────────────────────────────────────

# Words that indicate junk/non-entities
JUNK_WORDS = {
    "news", "look", "first", "things", "it", "this", "there", "after", 
    "said", "says", "latest", "update", "updates", "new", "old", "big", 
    "small", "good", "bad", "best", "top", "all", "more", "less",
    "plus", "ultra", "slim", "metal", "glass", "plastic", "lite", "pro", "max",
}

# Product specification patterns to filter out
SPEC_PATTERNS = [
    # Battery specs
    r'\d+\s*(mah|mAh|MAH|battery|Battery)',
    r'\d+\s*(w|W|watts?|Watts?)',
    # Display specs
    r'\d+(\.\d+)?\s*(inch|"|″|inches?)',
    r'\d+(k|K)\s*(display|screen)?',
    r'\d+(\.\d+)?\s*(hdr|HDR|oled|OLED|lcd|LCD)',
    # Storage/RAM
    r'\d+\s*(gb|GB|tb|TB|mb|MB|ram|RAM|rom|ROM|storage)',
    # Performance
    r'\d+\s*(hz|Hz|ghz|GHz|mhz|MHz|fps|FPS)',
    # Camera specs
    r'\d+\s*(mp|MP|megapixel|Megapixel)',
    # Units
    r'\d+\s*(nits|Nits|nit|Nit|ppp|PPP)',
    # Colors with numbers
    r'^\d+\s*bits?',
    r'\d+\s*[\-]\s*bit',
    # Generic numbers
    r'^\d+\s*(hrs?|Hrs?)',
    r'^\d+\s*(%|percent)',
    # WiFi/Bluetooth generations
    r'wi[-\s]*fi\s*\d+',
    r'bluetooth\s*\d+',
]

# Organization noise patterns
ORG_NOISE = {
    "adaptive refresh rate", "refresh rate", "display", "processor", "chipset",
    "speaker", "camera", "battery", "charging", "wireless", "bluetooth",
    "rating", "certified", "display", "screen", "audio", "sound",
    "lte", "5g", "4g", "wifi", "wi-fi",
}

# Person noise patterns
PERSON_NOISE = {
    "battery", "display", "screen", "processor", "camera", "speaker",
    "wireless", "bluetooth", "charging", "rating", "discount", "purchase",
    "offer", "sale", "price", "rs", "rupees",
}


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class Category:
    name: str
    score: float = 0.0
    source: str = "spacy_ner"
    entity_type: str = "unknown"
    is_clean: bool = True
    filter_reason: str = ""


# ── Filtering functions ───────────────────────────────────────────────────────

def is_product_spec(text: str) -> bool:
    """Check if text looks like a product specification"""
    text_lower = text.lower()
    
    # Check against spec patterns
    for pattern in SPEC_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    
    # Check for pure numbers or very short texts
    if re.match(r'^[\d\s\+\-\(\)\|]+$', text):
        return True
    
    # Check for spec indicators
    spec_indicators = ['mah', 'gb', 'mb', 'hz', 'fps', 'mp', 'inch', 'nits', 
                       'ram', 'rom', 'cpu', 'gpu', 'display', 'screen', 
                       'battery', 'charging', 'processor', 'camera']
    
    words = text_lower.split()
    if len(words) <= 3:
        for indicator in spec_indicators:
            if indicator in text_lower:
                return True
    
    return False


def is_too_generic(text: str) -> bool:
    """Check if entity name is too generic to be useful"""
    text_lower = text.lower()
    
    # Single word generic terms
    if text_lower in JUNK_WORDS:
        return True
    
    # Very short (likely not meaningful)
    if len(text) <= 2:
        return True
    
    # All lowercase with no spaces (likely not a proper noun)
    if text.islower() and ' ' not in text:
        if len(text) <= 4:
            return True
    
    return False


def is_valid_person_name(name: str) -> bool:
    """Validate person name (should be real name, not product or spec)"""
    # Skip if it contains numbers or units
    if re.search(r'\d', name) or is_product_spec(name):
        return False
    
    # Skip if it's product noise
    name_lower = name.lower()
    if any(noise in name_lower for noise in PERSON_NOISE):
        return False
    
    # Person names should have at least first and last name (2+ words)
    words = name.split()
    if len(words) < 2:
        return False
    
    # Check if words are proper case (not all caps or all lowercase)
    if not any(w[0].isupper() for w in words):
        return False
    
    # Skip if it contains measurement units
    if re.search(r'\d+\s*(%|percent|mah|gb|hz)', name_lower):
        return False
    
    return True


def is_valid_organization(name: str) -> bool:
    """Validate organization name"""
    # Skip specs and noise
    if is_product_spec(name):
        return False
    
    name_lower = name.lower()
    
    # Skip if it's common org noise
    if name_lower in ORG_NOISE:
        return False
    
    # Check for organization indicators
    org_indicators = ['inc', 'llc', 'ltd', 'corp', 'company', 'group', 'labs', 
                     'technologies', 'systems', 'solutions', 'corporation']
    
    # Organizations often have multiple words or proper case
    words = name.split()
    
    # Single word organizations should be significant length
    if len(words) == 1:
        if len(name) < 4:
            return False
        # Must be proper case or acronym
        if not (name.isupper() or name[0].isupper()):
            return False
    
    return True


def is_valid_place(name: str) -> bool:
    """Validate place/location name"""
    # Skip specs
    if is_product_spec(name):
        return False
    
    # Places should be proper nouns
    if not name[0].isupper():
        return False
    
    # Skip very short
    if len(name) < 3:
        return False
    
    return True


def is_valid_product(name: str) -> bool:
    """Validate product name"""
    # Skip if it's a spec
    if is_product_spec(name):
        return False
    
    # Product names should have proper capitalization
    words = name.split()
    
    # Skip generic product names
    generic_products = {'phone', 'tablet', 'laptop', 'device', 'gadget', 
                       'earbuds', 'speaker', 'watch', 'band', 'charger'}
    if name.lower() in generic_products:
        return False
    
    return True


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


# ── NER Extraction with strict filtering ─────────────────────────────────────

def extract_entities_with_spacy(cleaned_text: str, nlp) -> tuple[list[Category], list[Category], list[Category]]:
    """
    Extract named entities using spaCy NER with aggressive filtering.
    Returns: (raw_categories, clean_categories, discarded_categories)
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
    }
    
    seen: dict[str, dict] = {}
    raw_categories: list[Category] = []
    
    # First pass: collect all entities (raw extraction)
    for ent in doc.ents:
        if ent.label_ not in label_map:
            continue
        
        text = ent.text.strip()
        
        # Skip very short texts
        if len(text) < 3:
            continue
        
        entity_type = label_map[ent.label_]
        
        # Count frequencies
        key = text.lower()
        
        if key not in seen:
            seen[key] = {"name": text, "count": 1, "type": entity_type}
        else:
            seen[key]["count"] += 1
    
    # Convert to Category objects (raw)
    max_count = max([v["count"] for v in seen.values()]) if seen else 1
    
    for data in seen.values():
        raw_categories.append(Category(
            name=data["name"],
            score=round(data["count"] / max_count, 3),
            source="spacy_ner",
            entity_type=data["type"],
            is_clean=True,
        ))
    
    # Second pass: filter for quality
    clean_categories: list[Category] = []
    discarded_categories: list[Category] = []
    
    for cat in raw_categories:
        filter_reason = None
        
        # Check if it's a product spec
        if is_product_spec(cat.name):
            filter_reason = "product_specification"
        # Check if too generic
        elif is_too_generic(cat.name):
            filter_reason = "too_generic"
        # Validate based on entity type
        elif cat.entity_type == "person":
            if not is_valid_person_name(cat.name):
                filter_reason = "invalid_person_name"
        elif cat.entity_type == "organization":
            if not is_valid_organization(cat.name):
                filter_reason = "invalid_organization"
        elif cat.entity_type == "place":
            if not is_valid_place(cat.name):
                filter_reason = "invalid_place"
        elif cat.entity_type == "product":
            if not is_valid_product(cat.name):
                filter_reason = "invalid_product"
        
        if filter_reason:
            cat.is_clean = False
            cat.filter_reason = filter_reason
            discarded_categories.append(cat)
        else:
            # Only keep person, organization, and place for final display
            if cat.entity_type in ["person", "organization", "place"]:
                clean_categories.append(cat)
            else:
                cat.is_clean = False
                cat.filter_reason = f"entity_type_{cat.entity_type}_not_in_required_types"
                discarded_categories.append(cat)
    
    return raw_categories, clean_categories, discarded_categories


# ── Main extraction function ─────────────────────────────────────────────────

def run_extraction(cleaned_text: str, raw_html: Optional[str] = None) -> dict:
    """
    Main extraction function - uses ONLY spaCy NER with aggressive filtering
    """
    proc = psutil.Process(os.getpid())
    ram0 = proc.memory_info().rss / 1024 / 1024
    t0 = time.monotonic()
    
    # Load spaCy
    nlp = _load_spacy()
    if nlp is None:
        st.error("spaCy model could not be loaded.")
        return {
            "raw_categories": [], 
            "clean_categories": [],
            "discarded_categories": [],
            "elapsed_s": 0, 
            "ram_mb": 0,
            "n_person": 0,
            "n_org": 0, 
            "n_place": 0
        }
    
    # Extract entities using spaCy NER only
    raw_cats, clean_cats, discarded_cats = extract_entities_with_spacy(cleaned_text, nlp)
    
    elapsed = time.monotonic() - t0
    ram_used = proc.memory_info().rss / 1024 / 1024 - ram0
    
    return {
        "raw_categories": raw_cats,
        "clean_categories": clean_cats,
        "discarded_categories": discarded_cats,
        "elapsed_s": round(elapsed, 3),
        "ram_mb": round(ram_used, 1),
        "n_person": sum(1 for c in clean_cats if c.entity_type == "person"),
        "n_org": sum(1 for c in clean_cats if c.entity_type == "organization"),
        "n_place": sum(1 for c in clean_cats if c.entity_type == "place"),
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


def render_table(categories: list[Category], title: str, show_filter_reason: bool = False):
    """Render a table of categories"""
    if not categories:
        st.info(f"No {title.lower()} to display.")
        return
    
    st.markdown(f"### {title}")
    
    entity_color_map = {
        "person": "#388bfd",
        "organization": "#3fb950",
        "place": "#d2a8ff",
        "product": "#f0883e",
        "event": "#f85149",
        "unknown": "#8b949e",
    }
    
    rows = ""
    for cat in sorted(categories, key=lambda c: -c.score):
        clr = entity_color_map.get(cat.entity_type, "#8b949e")
        
        if show_filter_reason and cat.filter_reason:
            rows += (
                f'<tr>'
                f'<td style="padding:7px 12px;font-size:13px;font-weight:500">{cat.name}</td>'
                f'<td style="padding:7px 12px">{_badge(cat.entity_type)}</td>'
                f'<td style="padding:7px 12px;color:#8b949e;font-size:12px;'
                f'font-family:monospace">{cat.source}</td>'
                f'<td style="padding:7px 12px;min-width:130px">{_bar(cat.score, clr)}</td>'
                f'<td style="padding:7px 12px;color:#f85149;font-size:11px">{cat.filter_reason}</td>'
                f'</tr>'
            )
        else:
            rows += (
                f'<tr>'
                f'<td style="padding:7px 12px;font-size:13px;font-weight:500">{cat.name}</td>'
                f'<td style="padding:7px 12px">{_badge(cat.entity_type)}</td>'
                f'<td style="padding:7px 12px;color:#8b949e;font-size:12px;'
                f'font-family:monospace">{cat.source}</td>'
                f'<td style="padding:7px 12px;min-width:130px">{_bar(cat.score, clr)}</td>'
                f'</tr>'
            )
    
    # Build table headers
    if show_filter_reason:
        headers = ["Entity", "Type", "Source", "Frequency Score", "Filter Reason"]
    else:
        headers = ["Entity", "Type", "Source", "Frequency Score"]
    
    header_html = "".join([
        f'<th style="padding:8px 12px;text-align:left;font-size:11px;'
        f'color:#8b949e;font-family:monospace;text-transform:uppercase">{h}</th>'
        for h in headers
    ])
    
    st.markdown(
        f'<table style="width:100%;border-collapse:collapse;border-radius:8px;overflow:hidden">'
        f'<thead><tr style="background:#21262d">{header_html}</tr></thead>'
        f'<tbody>{rows}</tbody></table>',
        unsafe_allow_html=True,
    )


def render_cat_results(result: dict):
    """Render category results in Streamlit UI"""
    if not result or not result.get("raw_categories"):
        st.info("No named entities were found in the article.")
        return
    
    # Display metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("⏱️ Time", f"{result['elapsed_s']}s")
    with col2:
        st.metric("💾 RAM", f"{result['ram_mb']:.0f} MB")
    with col3:
        st.metric("👤 Persons", result["n_person"])
    with col4:
        st.metric("🏢 Organizations", result["n_org"])
    with col5:
        st.metric("📍 Places", result["n_place"])
    
    st.markdown("---")
    
    # Show raw extracted categories
    with st.expander("📋 Raw Extracted Categories (All Entities)", expanded=False):
        render_table(result["raw_categories"], "Raw Entities from spaCy NER")
    
    st.markdown("---")
    
    # Show cleaned categories (only person, organization, place)
    st.markdown("### ✅ Cleaned & Validated Categories")
    st.caption("Showing only validated Person, Organization, and Place entities")
    render_table(result["clean_categories"], "")
    
    # Show discarded categories
    if result["discarded_categories"]:
        st.markdown("---")
        with st.expander(f"🚫 Discarded Categories ({len(result['discarded_categories'])} filtered out)", expanded=False):
            render_table(result["discarded_categories"], "Filtered Out Entities", show_filter_reason=True)
    
    # Download buttons
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if result["clean_categories"]:
            import pandas as pd
            csv_clean = pd.DataFrame([{
                "name": c.name,
                "entity_type": c.entity_type,
                "source": c.source,
                "frequency_score": c.score,
            } for c in result["clean_categories"]]).to_csv(index=False).encode()
            st.download_button(
                "📥 Download Clean Categories (CSV)",
                csv_clean,
                file_name="clean_categories.csv",
                mime="text/csv",
                use_container_width=True,
            )
    
    with col2:
        if result["raw_categories"]:
            import pandas as pd
            csv_raw = pd.DataFrame([{
                "name": c.name,
                "entity_type": c.entity_type,
                "source": c.source,
                "frequency_score": c.score,
            } for c in result["raw_categories"]]).to_csv(index=False).encode()
            st.download_button(
                "📥 Download Raw Categories (CSV)",
                csv_raw,
                file_name="raw_categories.csv",
                mime="text/csv",
                use_container_width=True,
            )
    
    with col3:
        if result["discarded_categories"]:
            import pandas as pd
            csv_discarded = pd.DataFrame([{
                "name": c.name,
                "entity_type": c.entity_type,
                "source": c.source,
                "frequency_score": c.score,
                "filter_reason": c.filter_reason,
            } for c in result["discarded_categories"]]).to_csv(index=False).encode()
            st.download_button(
                "📥 Download Discarded Categories (CSV)",
                csv_discarded,
                file_name="discarded_categories.csv",
                mime="text/csv",
                use_container_width=True,
            )