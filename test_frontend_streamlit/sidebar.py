"""
Sidebar component with cascading location dropdowns and strict filtering.
"""

import streamlit as st
import pandas as pd


# ---------------------------------------------------------------------------
# Sidebar CSS — injected once
# ---------------------------------------------------------------------------
SIDEBAR_CSS = """
<style>
    /* --- Sidebar Dark Theme --- */
    section[data-testid="stSidebar"] {
        background-color: #0A0E14;
        border-right: 1px solid #1A1F2B;
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label {
        color: #A0AEC0 !important;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    section[data-testid="stSidebar"] h2 {
        font-size: 18px !important;
        color: #00C853 !important;
        border-bottom: 1px solid #1A1F2B;
        padding-bottom: 8px;
    }
    /* Expander header */
    section[data-testid="stSidebar"] .streamlit-expanderHeader {
        background-color: #111820 !important;
        border-radius: 8px;
        color: #00C853 !important;
        font-weight: 600;
    }
    /* Filter badge */
    .filter-badge {
        display: inline-block;
        background: rgba(0, 200, 83, 0.15);
        color: #00C853;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 12px;
        margin: 2px 4px 2px 0;
        border: 1px solid rgba(0, 200, 83, 0.3);
    }
    .filter-summary {
        background-color: #111820;
        padding: 12px 15px;
        border-radius: 8px;
        margin-top: 10px;
        border: 1px solid #1A1F2B;
    }
</style>
"""


def _get_sorted_unique(series: pd.Series) -> list[str]:
    """Return sorted unique non-empty string values from a Series."""
    values = [str(v).strip() for v in series.unique() if str(v).strip()]
    return sorted(values)


def render_sidebar(df: pd.DataFrame) -> dict:
    """
    Render the sidebar with cascading location filters and return
    a dict of the user's selected filters.

    Returns:
        {
            "country": str | None,
            "city":    str | None,
            "min_rating": float,
            "top_k": int,
        }
    """
    st.sidebar.markdown(SIDEBAR_CSS, unsafe_allow_html=True)

    # ── Header ──────────────────────────────────────────────────
    st.sidebar.markdown("## 🎛️ Search Filters")
    st.sidebar.caption("Narrow down results with strict location matching")

    # ── Location Filters (inside expander) ──────────────────────
    with st.sidebar.expander("📍 Location Filters", expanded=True):

        # --- Country ---
        all_countries = _get_sorted_unique(df["countyName"])
        country_options = ["All Countries"] + all_countries

        selected_country = st.selectbox(
            "Country",
            options=country_options,
            index=0,
            help="Select a country to filter results. Leave as 'All Countries' to skip.",
        )

        country_value = None if selected_country == "All Countries" else selected_country

        # --- City (cascading: depends on selected country) ---
        if country_value:
            country_mask = df["countyName"].astype(str).str.strip() == country_value
            available_cities = _get_sorted_unique(df.loc[country_mask, "cityName"])
        else:
            available_cities = _get_sorted_unique(df["cityName"])

        city_options = ["All Cities"] + available_cities

        selected_city = st.selectbox(
            "City",
            options=city_options,
            index=0,
            help="Cities are filtered based on the selected country.",
        )

        city_value = None if selected_city == "All Cities" else selected_city

    # ── Additional Filters ──────────────────────────────────────
    with st.sidebar.expander("⭐ More Options", expanded=False):

        col1, col2 = st.columns(2)

        with col1:
            min_rating = st.slider(
                "Min Rating",
                min_value=0.0,
                max_value=5.0,
                value=0.0,
                step=0.5,
                help="Only show hotels with at least this star rating.",
            )

        with col2:
            result_options = ["All", 5, 10, 25, 50]
            selected_results = st.selectbox(
                "Results",
                options=result_options,
                index=0,
                help="Number of results to show. 'All' shows every matching hotel.",
            )
            top_k = 0 if selected_results == "All" else int(selected_results)

    # ── Active Filters Summary ──────────────────────────────────
    active_filters = []
    if country_value:
        active_filters.append(f"🌍 {country_value}")
    if city_value:
        active_filters.append(f"🏙️ {city_value}")
    if min_rating > 0:
        active_filters.append(f"⭐ ≥{min_rating}")

    if active_filters:
        badges = " ".join(
            f'<span class="filter-badge">{f}</span>' for f in active_filters
        )
        st.sidebar.markdown(
            f'<div class="filter-summary"><strong style="color:#A0AEC0;'
            f'font-size:11px;text-transform:uppercase;">Active Filters</strong>'
            f'<br>{badges}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(
            '<div class="filter-summary" style="color:#607D8B;font-size:13px;">'
            "No filters active — showing global results</div>",
            unsafe_allow_html=True,
        )

    return {
        "country": country_value,
        "city": city_value,
        "min_rating": min_rating,
        "top_k": top_k,
    }


def apply_strict_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """
    Apply strict AND-condition filters to the DataFrame.
    Only filters that are not None / 0 are applied.
    """
    filtered = df.copy()

    # Strict country match
    if filters["country"]:
        filtered = filtered[
            filtered["countyName"].astype(str).str.strip() == filters["country"]
        ]

    # Strict city match
    if filters["city"]:
        filtered = filtered[
            filtered["cityName"].astype(str).str.strip() == filters["city"]
        ]

    # Minimum rating filter
    if filters["min_rating"] > 0:
        filtered = filtered[
            pd.to_numeric(filtered["HotelRating"], errors="coerce").fillna(0)
            >= filters["min_rating"]
        ]

    return filtered
