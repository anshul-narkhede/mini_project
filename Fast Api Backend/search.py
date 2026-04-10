"""
Semantic search logic with two-layer filtering:
  Layer 1: Strict user-provided filters (exact AND match)
  Layer 2: Smart auto-detection from query text (fallback)
"""

import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

from config import DESCRIPTION_TRUNCATE_LENGTH

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


def _detect_locations(query_lower: str, df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Detect known country and city names present in the user query
    using word-boundary regex matching.
    """
    all_countries = [str(c).strip() for c in df["countyName"].unique() if str(c).strip()]
    detected_countries = [
        c for c in all_countries
        if re.search(rf"\b{re.escape(c.lower())}\b", query_lower)
    ]

    all_cities = [str(c).strip() for c in df["cityName"].unique() if str(c).strip()]
    detected_cities = [
        c for c in all_cities
        if len(c) > 2 and re.search(rf"\b{re.escape(c.lower())}\b", query_lower)
    ]

    return detected_countries, detected_cities


def semantic_search(
    query: str,
    top_k: int,
    model: SentenceTransformer,
    index: faiss.IndexFlatL2,
    df: pd.DataFrame,
    strict_country: str | None = None,
    strict_city: str | None = None,
    min_rating: float = 0.0,
) -> dict:
    """
    Perform semantic search with two layers of location filtering:

    1. **Strict filters** (exact match, AND conditions) — applied first.
       These come from explicit API parameters (country, city, min_rating).
    2. **Smart auto-detection** (regex from query text) — applied only if
       the strict filter didn't already cover that dimension.

    Returns a dict with:
        results, strict_country, strict_city, min_rating,
        auto_detected_countries, auto_detected_cities
    """
    # --- Encode query --------------------------------------------------------
    query_vector = model.encode([query])
    query_vector = np.array(query_vector).astype("float32")

    # --- FAISS: rank ALL hotels by similarity --------------------------------
    distances, indices = index.search(query_vector, len(df))
    results_df = df.iloc[indices[0]].copy()

    # === LAYER 1: Strict user-provided filters (exact AND match) =============
    if strict_country:
        results_df = results_df[
            results_df["countyName"].astype(str).str.strip() == strict_country
        ]

    if strict_city:
        results_df = results_df[
            results_df["cityName"].astype(str).str.strip() == strict_city
        ]

    if min_rating > 0:
        results_df = results_df[
            results_df["HotelRating"].apply(_parse_rating) >= min_rating
        ]

    # === LAYER 2: Smart auto-detection from query text =======================
    auto_detected_countries = []
    auto_detected_cities = []

    query_lower = query.lower()

    # Only auto-detect country if strict filter didn't already set one
    if not strict_country:
        auto_detected_countries, _ = _detect_locations(query_lower, df)
        if auto_detected_countries:
            results_df = results_df[
                results_df["countyName"]
                .astype(str)
                .str.contains("|".join(auto_detected_countries), case=False, na=False)
            ]

    # Only auto-detect city if strict filter didn't already set one
    if not strict_city:
        _, auto_detected_cities = _detect_locations(query_lower, df)
        if auto_detected_cities:
            results_df = results_df[
                results_df["cityName"]
                .astype(str)
                .str.contains("|".join(auto_detected_cities), case=False, na=False)
            ]

    # === Format results ======================================================
    # top_k=0 means "return all matching results"
    if top_k > 0:
        top_matches = results_df.head(top_k).copy()
    else:
        top_matches = results_df.copy()

    top_matches["Description"] = top_matches["Description"].apply(
        lambda x: str(x).replace("\n", " ").replace("\r", " ").strip()[:DESCRIPTION_TRUNCATE_LENGTH] + "..."
    )

    result_columns = [
        "HotelName", "cityName", "countyName",
        "HotelRating", "HotelFacilities", "Description",
    ]
    results_list = top_matches[result_columns].to_dict(orient="records")

    return {
        "results": results_list,
        "strict_country": strict_country,
        "strict_city": strict_city,
        "min_rating": min_rating,
        "auto_detected_countries": auto_detected_countries,
        "auto_detected_cities": auto_detected_cities,
    }
