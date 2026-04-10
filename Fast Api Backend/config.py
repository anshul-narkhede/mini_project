"""
Application configuration and settings.
"""

# --- API Metadata ---
API_TITLE = "Semantic Hotel Search API"
API_DESCRIPTION = "An AI-powered B2B backend for hotel recommendations."
API_VERSION = "1.0.0"

# --- Data Settings ---
DATA_PATH = "data/hotels.csv"
SAMPLE_SIZE = 10_000
RANDOM_SEED = 42

# --- Model Settings ---
MODEL_NAME = "all-MiniLM-L6-v2"

# --- Search Defaults ---
DEFAULT_TOP_K = 12
DESCRIPTION_TRUNCATE_LENGTH = 300
