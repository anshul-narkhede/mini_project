import streamlit as st

PAGE_TITLE = "AI Travel Agent"
PAGE_ICON = " "
LAYOUT = "wide"

CUSTOM_CSS = """
<style>
    /* Neon Green & Deep Black Theme */
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    h1, h2, h3 {
        color: #00C853 !important;
        text-shadow: 0 0 8px rgba(0, 200, 83, 0.4);
    }
    .hotel-card {
        background-color: #1A1C23;
        padding: 25px;
        border-radius: 12px;
        border-left: 5px solid #00C853;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .hotel-name {
        font-size: 26px;
        font-weight: bold;
        color: #00C853;
        margin-bottom: 5px;
    }
    .hotel-location {
        color: #A0AEC0;
        font-size: 14px;
        margin-bottom: 15px;
        font-style: italic;
    }
    .hotel-desc {
        font-size: 16px;
        line-height: 1.5;
        margin-bottom: 15px;
        color: #E2E8F0;
    }
    .hotel-facilities {
        font-size: 13px;
        color: #00C853;
        background-color: rgba(0, 200, 83, 0.1);
        padding: 10px;
        border-radius: 8px;
        line-height: 1.6;
    }
    .rating-badge {
        background-color: #FFD700;
        color: #000;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 14px;
        margin-left: 10px;
    }
</style>
"""


def apply_page_config():
    """Set Streamlit page config and inject custom CSS."""
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout=LAYOUT)
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
