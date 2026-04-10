import streamlit as st
import pandas as pd


@st.cache_data
def load_data():
    """Load and preprocess hotel data from CSV."""
    try:
        df = pd.read_csv('data/hotels.csv', encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv('data/hotels.csv', encoding='latin1')

    # Take a random sample of 10,000 hotels
    # random_state=42 ensures reproducibility so caching works
    df = df.sample(n=10000, random_state=42)

    # Clean up column names and fill blanks
    df.columns = df.columns.str.strip()
    df = df.fillna('')

    # Combine relevant columns for the AI context
    df['Search_Text'] = (
        df['HotelName'].astype(str) + " " +
        df['Description'].astype(str) + " " +
        df['HotelFacilities'].astype(str) + " " +
        df['cityName'].astype(str) + " " +
        df['countyName'].astype(str)
    )

    return df
