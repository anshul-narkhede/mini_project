import streamlit as st

from config import apply_page_config
from data_loader import load_data
from model import load_model, build_index
from search import semantic_search
from sidebar import render_sidebar

# --- Apply UI Theme ---
apply_page_config()

# --- Main App ---
st.title("🌍 AI Travel Agent: Hotel Finder")
st.markdown(
    "Describe your ideal stay, including the city or country, "
    "and our AI will find the perfect match."
)

# Load everything up
with st.spinner("Loading AI models and hotel data..."):
    df = load_data()
    model = load_model()

    if not df.empty:
        index = build_index(model, df["Search_Text"].tolist())

# --- Sidebar: Cascading Location Filters ---
filters = render_sidebar(df)

# --- Search Input ---
query = st.text_input(
    "What kind of stay are you looking for?",
    placeholder="e.g., A quiet boutique hotel with a pool in Goa India",
)

if query and not df.empty:
    top_matches = semantic_search(
        query=query,
        model=model,
        index=index,
        df=df,
        top_k=filters["top_k"],
        strict_filters=filters,
    )

    if top_matches.empty:
        st.warning(
            "No hotels found matching your filters and description "
            "in our 10,000 hotel sample. Try relaxing the sidebar filters "
            "or adjusting your search!"
        )
    else:
        st.markdown(f"### ✨ Top {len(top_matches)} Recommendations")

        for _, hotel in top_matches.iterrows():
            st.markdown(
                f"""
            <div class="hotel-card">
                <div class="hotel-name">{hotel['HotelName']} <span class="rating-badge">{hotel['HotelRating']}</span></div>
                <div class="hotel-location">📍 {hotel['Address']}, {hotel['cityName']}, {hotel['countyName']}</div>
                <div class="hotel-desc">{str(hotel['Description'])[:400]}...</div>
                <div class="hotel-facilities"><strong>Amenities:</strong> {hotel['HotelFacilities']}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )