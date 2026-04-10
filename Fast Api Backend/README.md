# Semantic Hotel Search — FastAPI Backend

A production-ready **B2B REST API** that powers semantic hotel search using AI. Send a natural language query and receive ranked hotel recommendations based on meaning, not just keywords.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?logo=fastapi)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Features

- **Single Endpoint, Full Power** — One `POST /search` endpoint handles everything.
- **Semantic Understanding** — Understands that "cozy mountain retreat" matches "rustic alpine cabin with fireplace".
- **Smart Location Detection** — Automatically detects and filters by cities and countries mentioned in the query.
- **Structured JSON Responses** — Clean, typed responses with Pydantic validation.
- **Interactive API Docs** — Built-in Swagger UI at `/docs` and ReDoc at `/redoc`.

---

## Project Structure

```
Fast Api Backend/
├── main.py             # Entry point — creates app, loads model & data at startup
├── config.py           # Centralized constants (API metadata, paths, defaults)
├── data_loader.py      # CSV loading, sampling, preprocessing
├── model.py            # SentenceTransformer model + FAISS index builder
├── schemas.py          # Pydantic request/response models
├── search.py           # Core semantic search + location filtering logic
├── routes.py           # API endpoint definitions
├── requirements.txt    # Python dependencies
├── api.py              # (Legacy) Original monolithic backend
├── data/
│   └── hotels.csv      # Hotel dataset
└── docs/
    └── BACKEND_ARCHITECTURE.md   # Technical deep-dive
```

---

## Quick Start

### Prerequisites

- Python 3.9+
- pip

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare data

Ensure `data/hotels.csv` exists with these columns:

| Column | Description |
|--------|-------------|
| `HotelName` | Hotel name |
| `Description` | Text description |
| `HotelFacilities` | Comma-separated amenities |
| `cityName` | City name |
| `countyName` | Country name |
| `Address` | Street address |
| `HotelRating` | Star rating |

### 3. Start the server

```bash
uvicorn main:app --reload
```

The server starts at `http://127.0.0.1:8000`.

---

## API Reference

### `POST /search`

Search for hotels using natural language.

**Request Body:**

```json
{
  "query": "luxury beach resort with pool in Bali",
  "top_k": 5
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `query` | string | *(required)* | Natural language search query |
| `top_k` | integer | 12 | Number of results to return |

**Response:**

```json
{
  "status": "success",
  "search_parameters": {
    "original_query": "luxury beach resort with pool in Bali",
    "filters_applied": {
      "countries": ["Indonesia"],
      "cities": ["Bali"]
    }
  },
  "results": [
    {
      "HotelName": "The Mulia Resort",
      "cityName": "Bali",
      "countyName": "Indonesia",
      "HotelRating": "5",
      "HotelFacilities": "Pool, Spa, Beach Access, ...",
      "Description": "A luxurious beachfront resort..."
    }
  ]
}
```

**Error Response (500):**

```json
{
  "detail": "Error description"
}
```

---

## Testing with Postman

### Step 1: Start the server

```bash
uvicorn main:app --reload
```

### Step 2: Create a new request in Postman

| Setting | Value |
|---------|-------|
| **Method** | `POST` |
| **URL** | `http://127.0.0.1:8000/search` |
| **Headers** | `Content-Type: application/json` |
| **Body** | Raw → JSON |

### Step 3: Paste this JSON body

```json
{
  "query": "quiet boutique hotel with pool in Goa India",
  "top_k": 3
}
```

### Step 4: Click **Send**

You should receive a JSON response with the top 3 matching hotels in Goa, India.

### Example queries to try

| Query | Expected Behavior |
|-------|-------------------|
| `"luxury beach resort in Bali"` | Filters to Bali, returns high-end coastal hotels |
| `"cheap hostel in Paris"` | Filters to Paris, returns budget accommodation |
| `"family hotel with waterpark"` | No location filter, returns family resorts globally |
| `"romantic getaway in Italy"` | Filters to Italy, returns romantic/boutique hotels |

---

## Interactive Docs

FastAPI auto-generates interactive API documentation:

| URL | Format |
|-----|--------|
| `http://127.0.0.1:8000/docs` | Swagger UI — try the API directly from your browser |
| `http://127.0.0.1:8000/redoc` | ReDoc — clean read-only documentation |

---

## Configuration

All settings are centralized in `config.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `DATA_PATH` | `data/hotels.csv` | Path to the hotel dataset |
| `SAMPLE_SIZE` | `10,000` | Number of hotels to load |
| `RANDOM_SEED` | `42` | Reproducibility seed |
| `MODEL_NAME` | `all-MiniLM-L6-v2` | SentenceTransformer model |
| `DEFAULT_TOP_K` | `12` | Default results per query |
| `DESCRIPTION_TRUNCATE_LENGTH` | `300` | Max description chars in response |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Framework** | FastAPI |
| **Server** | Uvicorn (ASGI) |
| **Validation** | Pydantic v2 |
| **NLP Model** | sentence-transformers (`all-MiniLM-L6-v2`) |
| **Vector Search** | FAISS (`IndexFlatL2`) |
| **Data Processing** | Pandas, NumPy |

---
