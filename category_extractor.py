#!/usr/bin/env python3
"""
category_extractor.py
─────────────────────
Category extraction using spaCy NER + Topic Keyword Matching.
Shows raw extracted categories, cleaned categories (including topic categories), and discarded categories.
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


# ── Topic Categories with Keywords ──────────────────────────────────────────

TOPIC_CATEGORIES = {
    "Sports": {
        "keywords": [
            "sport", "sports", "football", "soccer", "cricket", "basketball", 
            "tennis", "baseball", "olympics", "world cup", "championship", 
            "tournament", "league", "match", "game", "athlete", "player", 
            "coach", "team", "goal", "score", "win", "loss", "victory", 
            "defeat", "champion", "fifa", "uefa", "nba", "nfl", "ipl", 
            "bcci", "worldcup", "women's world cup", "olympic", "paralympic",
            "sport and rights alliance"
        ]
    },
    "Politics": {
        "keywords": [
            "politics", "political", "government", "election", "vote", 
            "president", "prime minister", "minister", "parliament", "congress", 
            "democracy", "republican", "democrat", "party", "policy", "law", 
            "bill", "act", "constitution", "supreme court", "judge", "campaign",
            "diplomacy", "foreign policy", "treaty", "alliance", "sanction"
        ]
    },
    "Technology": {
        "keywords": [
            "technology", "tech", "software", "hardware", "app", "application",
            "digital", "ai", "artificial intelligence", "machine learning",
            "data", "algorithm", "computer", "smartphone", "laptop", "tablet",
            "processor", "chip", "gpu", "cpu", "ram", "storage", "display",
            "camera", "battery", "charging", "wireless", "bluetooth", "wifi",
            "5g", "internet", "cloud", "cyber", "security"
        ]
    },
    "Business": {
        "keywords": [
            "business", "company", "corporate", "enterprise", "startup",
            "market", "stock", "trading", "investment", "finance", "financial",
            "economy", "economic", "revenue", "profit", "loss", "growth",
            "merger", "acquisition", "deal", "contract", "partnership",
            "ceo", "executive", "management"
        ]
    },
    "Entertainment": {
        "keywords": [
            "entertainment", "movie", "film", "cinema", "hollywood", "bollywood",
            "actor", "actress", "director", "producer", "celebrity", "star",
            "music", "song", "album", "concert", "tour", "performance",
            "tv", "television", "show", "series", "netflix", "amazon prime",
            "award", "oscar", "grammy", "emmy"
        ]
    },
    "Health": {
        "keywords": [
            "health", "medical", "medicine", "doctor", "hospital", "clinic",
            "disease", "illness", "treatment", "therapy", "surgery",
            "vaccine", "covid", "pandemic", "epidemic", "virus", "bacteria",
            "fitness", "exercise", "wellness", "nutrition", "diet",
            "mental health", "wellbeing", "care", "patient"
        ]
    },
    "Science": {
        "keywords": [
            "science", "research", "study", "scientist", "laboratory", "lab",
            "discovery", "experiment", "data", "analysis", "finding",
            "space", "astronomy", "physics", "chemistry", "biology",
            "genetics", "dna", "evolution", "climate", "environment",
            "sustainability", "renewable", "energy", "nuclear", "quantum"
        ]
    },
    "Environment": {
        "keywords": [
            "environment", "climate", "climate change", "global warming",
            "sustainability", "renewable", "green", "eco", "ecological",
            "pollution", "carbon", "emissions", "fossil fuel", "solar",
            "wind", "hydro", "electric", "conservation", "wildlife",
            "forest", "ocean", "plastic", "recycling", "waste"
        ]
    }
}


# ── Quality filter constants ──────────────────────────────────────────────────

# Generic junk words (but not for valid entities)
JUNK_WORDS = {
    "news", "look", "first", "things", "it", "this", "there", "after", 
    "said", "says", "latest", "update", "updates", "new", "old", "big", 
    "small", "good", "bad", "best", "top", "all", "more", "less",
    "plus", "ultra", "slim", "metal", "glass", "plastic", "lite", "pro", "max",
}

# Product specification patterns - ONLY for technical specs
SPEC_PATTERNS = [
    r'^\d+\s*(mah|mAh|MAH)$',
    r'^\d+\s*(w|W|watts?|Watts?)$',
    r'\d+\s*mah\s+battery',
    r'^\d+(\.\d+)?\s*(inch|"|″)$',
    r'^\d+(k|K)\s*(display|screen)$',
    r'^\d+\s*(gb|GB|tb|TB|mb|MB)$',
    r'^\d+\s*(ram|RAM|rom|ROM)$',
    r'^\d+\s*(hz|Hz|ghz|GHz|mhz|MHz)$',
    r'^\d+\s*fps$',
    r'^\d+\s*(mp|MP)$',
    r'^\d+\s*(nits|Nits)$',
    r'^\d+\s*%$',
    r'^\d+\s*(hrs?|Hrs?)$',
    r'^wi[-\s]*fi\s*\d+$',
    r'^bluetooth\s*\d+$',
]

# Valid organization names (including short ones like UK, US)
VALID_ORGANIZATIONS = {
    "IPL", "BCCI", "ICC", "FIFA", "UEFA", "NBA", "NFL", "MLB", "NHL",
    "NASA", "ISRO", "WHO", "UN", "NATO", "EU", "CWG", "AIIMS", "IIT",
    "IIM", "US", "UK", "UAE", "AI", "SAI", "HT", "BBC", "CNN", "NYT"
}

# Valid place names (including short ones)
VALID_PLACES = {
    "Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Hyderabad",
    "Pune", "Ahmedabad", "Jaipur", "Lucknow", "Kanpur", "Nagpur",
    "Indore", "Thane", "Bhopal", "Visakhapatnam", "Patna", "Vadodara",
    "Ludhiana", "Agra", "Nashik", "Ranchi", "Gurgaon", "Noida",
    "Afghanistan", "Australia", "England", "Brazil", "US", "UK", "India"
}


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class Category:
    name: str
    score: float = 0.0
    source: str = "spacy_ner"
    entity_type: str = ""
    is_clean: bool = True
    filter_reason: str = ""


# ── Filtering functions ───────────────────────────────────────────────────────

def is_product_spec(text: str) -> bool:
    """Check if text looks like a product specification"""
    text_stripped = text.strip()
    text_lower = text_stripped.lower()
    
    if text_stripped[0].isupper() and len(text_stripped) > 2:
        if len(text_stripped.split()) >= 2:
            return False
    
    for pattern in SPEC_PATTERNS:
        if re.match(pattern, text_lower, re.IGNORECASE):
            return True
    
    if re.match(r'^[\d\.\s]+(mah|gb|mb|hz|fps|mp|nits|w|watts?)$', text_lower):
        return True
    
    if re.match(r'^[\d\s\+\-\(\)\|]+$', text_stripped):
        return True
    
    return False


def is_too_generic(text: str, entity_type: str) -> bool:
    """
    Check if entity name is too generic.
    But respect entity type - valid places/organizations should NOT be filtered
    """
    # NEVER filter based on length if it's a recognized entity type
    if entity_type in ["place", "organization", "person"]:
        return False
    
    text_lower = text.lower()
    
    if text_lower in JUNK_WORDS:
        return True
    
    if len(text) <= 2:
        return True
    
    return False


def is_valid_person_name(name: str) -> bool:
    """Validate person name"""
    if is_product_spec(name):
        return False
    
    # Common short names that are valid
    if name in ["Advani", "Pankaj", "Kothari", "Khalida Popal", "Popal", "Stiell"]:
        return True
    
    if re.search(r'\d', name):
        return False
    
    words = name.split()
    if len(words) == 1:
        if len(name) > 3 and name[0].isupper():
            return True
        return False
    
    if not any(w[0].isupper() for w in words):
        return False
    
    return True


def is_valid_organization(name: str) -> bool:
    """Validate organization name"""
    name_upper = name.upper()
    if name_upper in VALID_ORGANIZATIONS:
        return True
    
    if is_product_spec(name):
        return False
    
    words = name.split()
    
    # Acronyms (all caps, 2-6 letters) are valid organizations
    if name.isupper() and 2 <= len(name) <= 6:
        return True
    
    if len(words) == 1:
        if len(name) >= 3 and name[0].isupper():
            return True
        return False
    
    return True


def is_valid_place(name: str) -> bool:
    """Validate place/location name"""
    if name in VALID_PLACES:
        return True
    
    if is_product_spec(name):
        return False
    
    if not name[0].isupper():
        return False
    
    # Allow 2-letter country codes (US, UK, etc.)
    if len(name) == 2 and name.isupper():
        return True
    
    if len(name) < 3:
        return False
    
    return True


# ── Topic Detection Function ─────────────────────────────────────────────────

def detect_topics(text: str) -> list[Category]:
    """
    Detect topic categories by matching keywords in the text.
    Returns Category objects for each matching topic.
    """
    text_lower = text.lower()
    matched_topics = []
    
    for topic_name, topic_info in TOPIC_CATEGORIES.items():
        keywords_matched = 0
        
        for keyword in topic_info["keywords"]:
            if keyword in text_lower:
                keywords_matched += 1
        
        if keywords_matched > 0:
            # Calculate confidence score based on keyword matches
            confidence = min(keywords_matched / 10, 1.0)
            
            matched_topics.append(Category(
                name=topic_name,
                score=round(confidence, 3),
                source="topic_detection",
                entity_type="",
                is_clean=True
            ))
    
    # Sort by score descending
    matched_topics.sort(key=lambda x: -x.score)
    
    return matched_topics


# ── Model loader ──────────────────────────────────────────────────────────────

@st.cache_resource
def _load_spacy():
    """Load spaCy large model with caching"""
    try:
        nlp = spacy.load("en_core_web_lg")
        return nlp
    except OSError:
        st.error("en_core_web_lg model not found. Please install it with: python -m spacy download en_core_web_lg")
        return None


# ── NER Extraction with strict filtering ─────────────────────────────────────

def extract_entities_with_spacy(cleaned_text: str, nlp) -> tuple[list[Category], list[Category], list[Category]]:
    """
    Extract named entities using spaCy NER with precise filtering.
    Returns: (raw_categories, clean_categories, discarded_categories)
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
    raw_categories: list[Category] = []
    
    for ent in doc.ents:
        if ent.label_ not in label_map:
            continue
        
        text = ent.text.strip()
        
        if len(text) < 2:
            continue
        
        entity_type = label_map[ent.label_]
        key = text.lower()
        
        if key not in seen:
            seen[key] = {"name": text, "count": 1, "type": entity_type}
        else:
            seen[key]["count"] += 1
    
    max_count = max([v["count"] for v in seen.values()]) if seen else 1
    
    for data in seen.values():
        raw_categories.append(Category(
            name=data["name"],
            score=round(data["count"] / max_count, 3),
            source="spacy_ner",
            entity_type=data["type"],
            is_clean=True,
        ))
    
    # Filter for quality
    clean_categories: list[Category] = []
    discarded_categories: list[Category] = []
    
    for cat in raw_categories:
        filter_reason = None
        
        # Check product specs first
        if is_product_spec(cat.name):
            filter_reason = "product_specification"
        else:
            # Validate based on entity type (with proper generic check that respects entity type)
            if cat.entity_type == "person":
                if not is_valid_person_name(cat.name):
                    filter_reason = "invalid_person_name"
            elif cat.entity_type == "organization":
                if not is_valid_organization(cat.name):
                    filter_reason = "invalid_organization"
            elif cat.entity_type == "place":
                if not is_valid_place(cat.name):
                    filter_reason = "invalid_place"
            elif cat.entity_type == "product":
                # Products are optional, keep them
                pass
            elif cat.entity_type == "event":
                # Events are valid categories
                pass
            else:
                # For other types, keep them with empty entity_type
                cat.entity_type = ""
            
            # Check generic only if not already filtered and not a valid entity type
            if not filter_reason and cat.entity_type not in ["person", "organization", "place"]:
                if is_too_generic(cat.name, cat.entity_type):
                    filter_reason = "too_generic"
        
        if filter_reason:
            cat.is_clean = False
            cat.filter_reason = filter_reason
            discarded_categories.append(cat)
        else:
            clean_categories.append(cat)
    
    return raw_categories, clean_categories, discarded_categories


# ── Main extraction function ─────────────────────────────────────────────────

def run_extraction(cleaned_text: str, raw_html: Optional[str] = None) -> dict:
    """
    Main extraction function - extracts named entities AND topic categories
    """
    proc = psutil.Process(os.getpid())
    ram0 = proc.memory_info().rss / 1024 / 1024
    t0 = time.monotonic()
    
    # Detect topic categories from the article text
    topic_categories = detect_topics(cleaned_text)
    
    # Load spaCy for NER
    nlp = _load_spacy()
    if nlp is None:
        st.error("spaCy model could not be loaded.")
        return {
            "raw_categories": [], 
            "clean_categories": topic_categories,
            "discarded_categories": [],
            "elapsed_s": 0, 
            "ram_mb": 0,
            "n_person": 0,
            "n_org": 0, 
            "n_place": 0,
            "n_topic": len(topic_categories)
        }
    
    # Extract entities using spaCy NER
    raw_cats, clean_cats, discarded_cats = extract_entities_with_spacy(cleaned_text, nlp)
    
    # Merge topic categories with cleaned entities
    all_clean_categories = clean_cats + topic_categories
    
    # Sort by score descending
    all_clean_categories.sort(key=lambda x: -x.score)
    
    elapsed = time.monotonic() - t0
    ram_used = proc.memory_info().rss / 1024 / 1024 - ram0
    
    return {
        "raw_categories": raw_cats,
        "clean_categories": all_clean_categories,
        "discarded_categories": discarded_cats,
        "elapsed_s": round(elapsed, 3),
        "ram_mb": round(ram_used, 1),
        "n_person": sum(1 for c in clean_cats if c.entity_type == "person"),
        "n_org": sum(1 for c in clean_cats if c.entity_type == "organization"),
        "n_place": sum(1 for c in clean_cats if c.entity_type == "place"),
        "n_topic": len(topic_categories)
    }


# ── Rendering functions ───────────────────────────────────────────────────────

def _bar(score: float) -> str:
    """Create a simple progress bar for scores"""
    pct = int(score * 100)
    return (
        f'<div style="display:flex;align-items:center;gap:8px">'
        f'<div style="flex:1;background:#e0e0e0;border-radius:3px;height:5px">'
        f'<div style="width:{pct}%;background:#1f77b4;height:5px;border-radius:3px"></div>'
        f'</div>'
        f'<span style="font-size:13px">{score:.2f}</span>'
        f'</div>'
    )


def render_table(categories: list[Category], title: str, show_filter_reason: bool = False):
    """Render a table of categories"""
    if not categories:
        st.info(f"No {title.lower()} to display.")
        return
    
    st.markdown(f"### {title}")
    
    rows = ""
    for cat in sorted(categories, key=lambda c: -c.score):
        if show_filter_reason and cat.filter_reason:
            rows += (
                f'<tr>'
                f'<td style="padding:7px 12px;font-size:13px">{cat.name}</td>'
                f'<td style="padding:7px 12px;font-size:13px">{cat.entity_type if cat.entity_type else "-"}</td>'
                f'<td style="padding:7px 12px;font-size:13px">{cat.source}</td>'
                f'<td style="padding:7px 12px">{_bar(cat.score)}</td>'
                f'<td style="padding:7px 12px;font-size:12px;color:#999">{cat.filter_reason}</td>'
                f'</tr>'
            )
        else:
            rows += (
                f'<tr>'
                f'<td style="padding:7px 12px;font-size:13px">{cat.name}</td>'
                f'<td style="padding:7px 12px;font-size:13px">{cat.entity_type if cat.entity_type else "-"}</td>'
                f'<td style="padding:7px 12px;font-size:13px">{cat.source}</td>'
                f'<td style="padding:7px 12px">{_bar(cat.score)}</td>'
                f'</tr>'
            )
    
    if show_filter_reason:
        headers = ["Entity", "Type", "Source", "Score", "Filter Reason"]
    else:
        headers = ["Entity", "Type", "Source", "Score"]
    
    header_html = "".join([
        f'<th style="padding:8px 12px;text-align:left;font-size:11px;'
        f'color:var(--streamlit-text-color);font-family:monospace;text-transform:uppercase">{h}</th>'
        for h in headers
    ])
    
    st.markdown(
        f'<table style="width:100%;border-collapse:collapse;border-radius:8px;overflow:hidden;border:1px solid var(--streamlit-border-color)">'
        f'<thead><tr style="background:var(--streamlit-secondary-bg-color)">{header_html} <tr></thead>'
        f'<tbody>{rows}</tbody></table>',
        unsafe_allow_html=True,
    )


def render_cat_results(result: dict):
    """Render category results in Streamlit UI"""
    if not result or (not result.get("raw_categories") and not result.get("clean_categories")):
        st.info("No categories or topics were found in the article.")
        return
    
    # Display metrics
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("Time", f"{result['elapsed_s']}s")
    with col2:
        st.metric("RAM", f"{result['ram_mb']:.0f} MB")
    with col3:
        st.metric("Persons", result["n_person"])
    with col4:
        st.metric("Organizations", result["n_org"])
    with col5:
        st.metric("Places", result["n_place"])
    with col6:
        st.metric("Topics", result.get("n_topic", 0))
    
    st.markdown("---")
    
    # Show raw extracted categories
    if result.get("raw_categories"):
        with st.expander("Raw Extracted Entities (All Entities)", expanded=False):
            render_table(result["raw_categories"], "Raw Entities from spaCy NER")
    
    # Show cleaned categories (entities + topics)
    st.markdown("### Cleaned & Validated Categories")
    st.caption("Named entities + Detected Topics")
    render_table(result["clean_categories"], "")
    
    # Show discarded categories
    if result.get("discarded_categories"):
        with st.expander(f"Discarded Entities ({len(result['discarded_categories'])} filtered out)", expanded=False):
            render_table(result["discarded_categories"], "Filtered Out Entities", show_filter_reason=True)
    
    # Download buttons
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if result["clean_categories"]:
            import pandas as pd
            csv_clean = pd.DataFrame([{
                "name": c.name,
                "entity_type": c.entity_type if c.entity_type else "",
                "source": c.source,
                "score": c.score,
            } for c in result["clean_categories"]]).to_csv(index=False).encode()
            st.download_button(
                "Download All Categories (CSV)",
                csv_clean,
                file_name="all_categories.csv",
                mime="text/csv",
                use_container_width=True,
            )
    
    with col2:
        if result.get("raw_categories"):
            import pandas as pd
            csv_raw = pd.DataFrame([{
                "name": c.name,
                "entity_type": c.entity_type if c.entity_type else "",
                "source": c.source,
                "score": c.score,
            } for c in result["raw_categories"]]).to_csv(index=False).encode()
            st.download_button(
                "Download Raw Entities (CSV)",
                csv_raw,
                file_name="raw_entities.csv",
                mime="text/csv",
                use_container_width=True,
            )
    
    with col3:
        if result.get("discarded_categories"):
            import pandas as pd
            csv_discarded = pd.DataFrame([{
                "name": c.name,
                "entity_type": c.entity_type if c.entity_type else "",
                "source": c.source,
                "score": c.score,
                "filter_reason": c.filter_reason,
            } for c in result["discarded_categories"]]).to_csv(index=False).encode()
            st.download_button(
                "Download Discarded Entities (CSV)",
                csv_discarded,
                file_name="discarded_entities.csv",
                mime="text/csv",
                use_container_width=True,
            )