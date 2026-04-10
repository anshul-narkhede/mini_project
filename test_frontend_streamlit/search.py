"""
Semantic search with smart auto-detection AND strict sidebar filters.
"""

import re
import numpy as np
import pandas as pd
import streamlit as st

# Rating text → numeric mapping (dataset stores "FourStar", "FiveStar", etc.)
RATING_MAP = {
    "onestar": 1.0,
    "twostar": 2.0,
    "threestar": 3.0,
    "fourstar": 4.0,
    "fivestar": 5.0,
    "all": 0.0,  # "All" means unrated
}

def _parse_rating(value) -> float:
    """Convert a rating value (text or numeric) to a float."""
    s = str(value).strip().lower()
    if s in RATING_MAP:
        return RATING_MAP[s]
    try:
        return float(s)
    except (ValueError, TypeError):
        return 0.0

def semantic_search(
    query: str,
    model,
    index,
    df: pd.DataFrame,
    top_k: int = 3,
    strict_filters: dict | None = None,
):
    """
    Perform semantic search with two layers of location filtering.
    """
    # --- Encode query --------------------------------------------------------
    query_vector = model.encode([query])
    query_vector = np.array(query_vector).astype("float32")

    # --- FAISS: rank ALL hotels by similarity --------------------------------
    distances, indices = index.search(query_vector, k=len(df))
    results_df = df.iloc[indices[0]].copy()

    # === LAYER 1: Strict sidebar filters (exact AND match) ===================
    sidebar_country = None
    sidebar_city = None

    if strict_filters:
        sidebar_country = strict_filters.get("country")
        sidebar_city = strict_filters.get("city")
        min_rating = strict_filters.get("min_rating", 0)

        if sidebar_country:
            results_df = results_df[
                results_df["countyName"].astype(str).str.strip() == sidebar_country
            ]
            st.info(f"🌍 Sidebar Country filter: **{sidebar_country}**")

        if sidebar_city:
            results_df = results_df[
                results_df["cityName"].astype(str).str.strip() == sidebar_city
            ]
            st.info(f"🏙️ Sidebar City filter: **{sidebar_city}**")

        if min_rating > 0:
            results_df = results_df[
                results_df["HotelRating"].apply(_parse_rating) >= min_rating
            ]

    # === LAYER 2: Smart auto-detection from query text =======================
    query_lower = query.lower()

    # Only auto-detect country if sidebar didn't already set one
    if not sidebar_country:
        all_countries = [str(c).strip() for c in df["countyName"].unique() if str(c).strip()]
        detected_countries = [
            c for c in all_countries
            if re.search(rf"\b{re.escape(c.lower())}\b", query_lower)
        ]
        if detected_countries:
            results_df = results_df[
                results_df["countyName"]
                .astype(str)
                .str.contains("|".join(detected_countries), case=False, na=False)
            ]
            st.info(f"🌍 Auto-detected Country: **{', '.join(detected_countries)}**")

    # Only auto-detect city if sidebar didn't already set one
    if not sidebar_city:
        all_cities = [str(c).strip() for c in df["cityName"].unique() if str(c).strip()]
        detected_cities = [
            c for c in all_cities
            if len(c) > 2 and re.search(rf"\b{re.escape(c.lower())}\b", query_lower)
        ]
        if detected_cities:
            results_df = results_df[
                results_df["cityName"]
                .astype(str)
                .str.contains("|".join(detected_cities), case=False, na=False)
            ]
            st.info(f"🏙️ Auto-detected City: **{', '.join(detected_cities)}**")

    # top_k=0 means "show all matching results"
    # To prevent browser freezing with Streamlit markdown, cap at 100 max.
    if top_k > 0:
        return results_df.head(top_k)
    return results_df.head(100)

