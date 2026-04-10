"""
FastAPI application entry point.

Run with:
    uvicorn main:app --reload
"""

from fastapi import FastAPI

from config import API_TITLE, API_DESCRIPTION, API_VERSION
from data_loader import load_hotel_data
from model import load_model, build_faiss_index
from routes import router

# --- Create FastAPI App ---
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
)

# --- Register Routes ---
app.include_router(router)

# --- Load Engine into Memory (runs once at server startup) ---
print("Loading Model and Data. Please wait...")

df = load_hotel_data()
model = load_model()
index = build_faiss_index(model, df["Search_Text"].tolist())

print("✅ Server Ready! Waiting for requests...")
