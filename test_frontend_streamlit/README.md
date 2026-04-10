# AI Travel Agent: Semantic Hotel Search

An AI‑powered hotel recommendation engine that understands **natural language** queries. Describe your ideal stay in plain English, and the system finds the best matches from 10,000+ hotels using **semantic vector search**.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit)

---

##  Features

- **Semantic Understanding** — Goes beyond keywords. Understands that "cozy mountain retreat" matches "rustic alpine cabin with fireplace".
- **Smart Location Detection** — Automatically detects cities and countries from your query and filters results.
- **Fast Vector Search** — FAISS‑powered nearest‑neighbor search over 10,000 hotel embeddings.
- **Premium Dark UI** — Neon green on deep black theme with styled hotel cards.



##  Quick Start

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

## Usage

Type a natural language query into the search box:

| Example Query | What It Finds |
|---------------|--------------|
| *"luxury beach resort with infinity pool in Bali"* | 5‑star coastal hotels in Bali with pool facilities |
| *"budget‑friendly hostel in Paris near the metro"* | Affordable accommodation in Paris with transit access |
| *"quiet mountain cabin with fireplace in Switzerland"* | Secluded alpine retreats in Swiss locations |
| *"family hotel with waterpark and kids club"* | Family‑oriented resorts with child amenities |

---

## How It Works

This project uses **semantic search** powered by the `all-MiniLM-L6-v2` sentence transformer and **FAISS** vector indexing.

For a complete technical deep‑dive — including model architecture diagrams, vector embedding explanations, and end‑to‑end flowcharts — see **[HOW_IT_WORKS.md](HOW_IT_WORKS.md)**.

**TL;DR:**

1. Hotel descriptions are converted into 384‑dimensional vectors using a transformer model.
2. Your search query is converted into a vector using the **same** model.
3. FAISS finds the hotels whose vectors are closest to your query vector.
4. Smart regex filters narrow results by detected city/country names.

---

## Configuration

| Setting | Location | Default |
|---------|----------|---------|
| Sample size | `data_loader.py` | 10,000 hotels |
| Random seed | `data_loader.py` | 42 |
| Model name | `model.py` | `all-MiniLM-L6-v2` |
| Results count | `search.py` | Top 3 |
| UI theme | `config.py` | Dark (Neon Green) |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Frontend** | Streamlit |
| **NLP Model** | sentence‑transformers (`all-MiniLM-L6-v2`) |
| **Vector Search** | FAISS (Facebook AI Similarity Search) |
| **Data Processing** | Pandas, NumPy |
| **Language** | Python 3.9+ |

---