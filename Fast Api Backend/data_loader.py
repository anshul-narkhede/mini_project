"""
Hotel data loading and preprocessing.
"""

import pandas as pd
from config import DATA_PATH, SAMPLE_SIZE, RANDOM_SEED


def load_hotel_data() -> pd.DataFrame:
    """
    Load hotel CSV, sample it, clean columns, and build
    a combined Search_Text column for semantic encoding.
    """
    try:
        df = pd.read_csv(DATA_PATH, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(DATA_PATH, encoding="latin1")

    # Reproducible random sample for speed + geographic diversity
    df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED)

    # Clean up column names and fill blanks
    df.columns = df.columns.str.strip()
    df = df.fillna("")

    # Combine relevant columns into a single searchable string
    df["Search_Text"] = (
        df["HotelName"].astype(str) + " " +
        df["Description"].astype(str) + " " +
        df["HotelFacilities"].astype(str) + " " +
        df["cityName"].astype(str) + " " +
        df["countyName"].astype(str)
    )

    return df
