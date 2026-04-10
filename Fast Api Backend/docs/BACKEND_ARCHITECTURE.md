# FastAPI Backend — Architecture Deep Dive

This document explains the complete internal working of the **Semantic Hotel Search API** — from server startup to JSON response.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Startup Sequence](#2-startup-sequence)
3. [Request Lifecycle](#3-request-lifecycle)
4. [Module Breakdown](#4-module-breakdown)
5. [Data Flow Diagram](#5-data-flow-diagram)
6. [Semantic Search Pipeline](#6-semantic-search-pipeline)
7. [Location Detection Engine](#7-location-detection-engine)
8. [API Contract (Schemas)](#8-api-contract-schemas)
9. [Error Handling](#9-error-handling)
10. [Performance Characteristics](#10-performance-characteristics)
11. [Comparison: Monolithic vs Modular](#11-comparison-monolithic-vs-modular)

---

## 1. System Overview

```mermaid
flowchart LR
    subgraph CLIENT ["Client (Postman / Frontend / cURL)"]
        A["POST /search\n{\n  query: '...'\n  top_k: 5\n}"]
    end

    subgraph SERVER ["FastAPI Backend"]
        B["routes.py\nValidate request"]
        C["search.py\nSemantic search"]
        D["model.py\nFAISS index"]
        E["data_loader.py\nHotel DataFrame"]
    end

    subgraph RESPONSE ["JSON Response"]
        F["{\n  status: 'success'\n  results: [...]\n}"]
    end

    A -->|"HTTP POST"| B
    B --> C
    C --> D
    C --> E
    C -->|"ranked results"| B
    B -->|"JSON"| F
```

The backend is a **stateless REST API** that holds the ML model and FAISS index **in memory**. Each request encodes the query, searches the index, applies location filters, and returns results — all in under 1 second.

---

## 2. Startup Sequence

When you run `uvicorn main:app`, the following happens **once** before any request is served:

```mermaid
sequenceDiagram
    participant U as Uvicorn
    participant M as main.py
    participant DL as data_loader.py
    participant MD as model.py

    U->>M: Import module
    M->>M: Create FastAPI app
    M->>M: Register routes (router)

    Note over M: "Loading Model and Data..."

    M->>DL: load_hotel_data()
    DL->>DL: Read CSV (UTF-8 / Latin-1)
    DL->>DL: Sample 10,000 rows
    DL->>DL: Clean columns, fill NaN
    DL->>DL: Build Search_Text column
    DL-->>M: Return DataFrame (df)

    M->>MD: load_model()
    MD->>MD: Download/load all-MiniLM-L6-v2
    MD-->>M: Return SentenceTransformer (model)

    M->>MD: build_faiss_index(model, texts)
    MD->>MD: Encode 10,000 texts → vectors
    MD->>MD: Create IndexFlatL2
    MD->>MD: Add vectors to index
    MD-->>M: Return FAISS index

    Note over M: "✅ Server Ready!"
    M-->>U: App ready to serve
```

### Startup timeline

| Step | What Happens | Approx. Time |
|------|-------------|--------------|
| CSV load & clean | Read 2.4 GB CSV, sample 10k, build Search_Text | ~5s |
| Model load | Download/cache `all-MiniLM-L6-v2` (22.7M params) | ~3s first time |
| Index build | Encode 10,000 texts → 384-dim vectors, build FAISS index | ~30-60s |
| **Total** | | **~40-70s first start** |

After the first run, the model is cached locally by `sentence-transformers`, reducing subsequent starts.

---

## 3. Request Lifecycle

```mermaid
flowchart TD
    A["Client sends\nPOST /search"] --> B["FastAPI receives\nHTTP request"]
    B --> C["Pydantic validates\nSearchQuery schema"]
    C -->|"Invalid"| D["422 Unprocessable\nEntity response"]
    C -->|"Valid"| E["routes.py\nsearch_hotels()"]
    E --> F["search.py\nsemantic_search()"]
    F --> G["Encode query\n→ 384-dim vector"]
    G --> H["FAISS index.search()\nRank all 10k hotels"]
    H --> I["Detect locations\nin query string"]
    I --> J["Apply country\n& city filters"]
    J --> K["Take top_k results"]
    K --> L["Truncate descriptions\nto 300 chars"]
    L --> M["Return results list\n+ detected filters"]
    M --> N["routes.py builds\nSearchResponse"]
    N --> O["200 OK\nJSON response"]

    E -->|"Exception"| P["500 Internal\nServer Error"]

    style D fill:#e74c3c,color:#fff
    style P fill:#e74c3c,color:#fff
    style O fill:#00C853,color:#000
```

### Step-by-step

1. **Request arrives** — FastAPI's ASGI server (Uvicorn) receives the HTTP POST.
2. **Schema validation** — Pydantic automatically validates the JSON body against `SearchQuery`. Missing `query` field → instant 422 error.
3. **Route handler** — `routes.py` calls `semantic_search()` with the validated data + shared resources (model, index, df).
4. **Query encoding** — The same `all-MiniLM-L6-v2` model encodes the query into a 384-dim vector.
5. **FAISS search** — Computes L2 distance to all 10,000 hotel vectors. Returns all hotels ranked by similarity.
6. **Location filtering** — Regex detects city/country names in the query string and filters the DataFrame.
7. **Top-K selection** — Takes the first `top_k` results after filtering.
8. **Response formatting** — Descriptions are truncated, and results are packaged into the `SearchResponse` schema.
9. **JSON sent** — FastAPI serializes the Pydantic model to JSON and sends the HTTP response.

---

## 4. Module Breakdown

```mermaid
graph TD
    subgraph ENTRY ["Entry Point"]
        MAIN["main.py\n• Creates FastAPI app\n• Registers routes\n• Loads model + data\n• Holds shared state"]
    end

    subgraph CORE ["Core Logic"]
        SEARCH["search.py\n• semantic_search()\n• _detect_locations()\n• _apply_location_filters()"]
        MODEL["model.py\n• load_model()\n• build_faiss_index()"]
        LOADER["data_loader.py\n• load_hotel_data()"]
    end

    subgraph API ["API Layer"]
        ROUTES["routes.py\n• POST /search\n• Error handling"]
        SCHEMAS["schemas.py\n• SearchQuery\n• SearchResponse\n• SearchParameters\n• SearchFilters"]
    end

    subgraph CONFIG_MOD ["Configuration"]
        CONFIG["config.py\n• API_TITLE, API_VERSION\n• DATA_PATH, SAMPLE_SIZE\n• MODEL_NAME, DEFAULT_TOP_K"]
    end

    MAIN --> ROUTES
    MAIN --> LOADER
    MAIN --> MODEL
    ROUTES --> SEARCH
    ROUTES --> SCHEMAS
    SEARCH --> MODEL
    SEARCH --> LOADER
    LOADER --> CONFIG_MOD
    MODEL --> CONFIG_MOD
    SCHEMAS --> CONFIG_MOD

    style MAIN fill:#00C853,color:#000,stroke:#00C853
    style ROUTES fill:#2196F3,color:#fff,stroke:#2196F3
    style SCHEMAS fill:#2196F3,color:#fff,stroke:#2196F3
    style SEARCH fill:#FF9800,color:#000,stroke:#FF9800
    style MODEL fill:#FF9800,color:#000,stroke:#FF9800
    style LOADER fill:#FF9800,color:#000,stroke:#FF9800
    style CONFIG_MOD fill:#607D8B,color:#fff,stroke:#607D8B
```

### Why this structure?

| Principle | How It's Applied |
|-----------|-----------------|
| **Separation of concerns** | Routes don't know about FAISS. Search doesn't know about HTTP. |
| **Single responsibility** | Each file does exactly one thing. |
| **Dependency inversion** | `config.py` holds all constants; modules import from it, not from each other. |
| **Testability** | `semantic_search()` is a pure function — pass in model/index/df, get results back. No server needed for unit tests. |

---

## 5. Data Flow Diagram

```mermaid
flowchart TD
    subgraph STARTUP ["One-time Startup"]
        CSV["data/hotels.csv\n(2.4 GB, 500k+ rows)"]
        CSV --> SAMPLE["Random sample\n10,000 rows"]
        SAMPLE --> CLEAN["Clean & build\nSearch_Text"]
        CLEAN --> DF["In-memory\nDataFrame"]
        CLEAN --> ENCODE["Encode 10k texts\nall-MiniLM-L6-v2"]
        ENCODE --> VECTORS["10,000 × 384\nfloat32 matrix"]
        VECTORS --> INDEX["FAISS\nIndexFlatL2"]
    end

    subgraph PER_REQUEST ["Per Request"]
        QUERY["User query string"] --> QENCODE["Encode query\nsame model"]
        QENCODE --> QVEC["1 × 384\nquery vector"]
        QVEC --> FSEARCH["FAISS search\nL2 distance"]
        INDEX -.->|"compare"| FSEARCH
        FSEARCH --> RANKED["All 10k hotels\nranked by distance"]
        RANKED --> LOCFILTER["Location filter\n(regex detection)"]
        DF -.->|"filter"| LOCFILTER
        LOCFILTER --> TOPK["Top K results"]
        TOPK --> JSON["JSON response"]
    end

    style STARTUP fill:#1A1C23,color:#fff
    style PER_REQUEST fill:#0E1117,color:#fff
```

---

## 6. Semantic Search Pipeline

### How `semantic_search()` works internally

```python
# File: search.py — simplified walkthrough

def semantic_search(query, top_k, model, index, df):

    # STEP 1: Encode the query into the same vector space as hotels
    query_vector = model.encode([query])        # Shape: (1, 384)

    # STEP 2: Find nearest neighbors in FAISS
    distances, indices = index.search(query_vector, len(df))
    # distances = [[0.23, 0.45, 0.67, ...]]   (L2 distances, ascending)
    # indices   = [[4521, 892, 3301, ...]]     (hotel row indices)

    # STEP 3: Reorder DataFrame by similarity
    results_df = df.iloc[indices[0]].copy()
    # Now row 0 = most similar hotel, row 9999 = least similar

    # STEP 4: Detect & apply location filters
    countries, cities = _detect_locations(query, df)
    results_df = _apply_location_filters(results_df, countries, cities)
    # Keeps only hotels in detected location (if any)

    # STEP 5: Return top K
    return results_df.head(top_k)
```

### Why search ALL hotels then filter?

```mermaid
flowchart LR
    subgraph APPROACH_1 ["Approach 1: Filter then Search"]
        A1["Filter to 'India' hotels\n(maybe 200 hotels)"] --> B1["Search only 200\nvectors"]
        B1 --> C1["Limited results\nMay miss good matches"]
    end

    subgraph APPROACH_2 ["Approach 2: Search then Filter ✅"]
        A2["Search ALL 10,000\nvectors"] --> B2["Get global ranking\nby similarity"]
        B2 --> C2["THEN filter to 'India'\nTop matches guaranteed"]
    end
```

We use **Approach 2** because:
- FAISS searches 10k vectors in < 5ms anyway (negligible cost)
- Ensures we get the **globally best** matches within the filtered location
- If the location filter returns nothing, we still have the global ranking to fall back on

---

## 7. Location Detection Engine

```mermaid
flowchart TD
    A["Query: 'luxury resort\nwith pool in Goa India'"] --> B["Lowercase:\n'luxury resort\nwith pool in goa india'"]

    B --> C["Load unique countries\nfrom df.countyName"]
    B --> D["Load unique cities\nfrom df.cityName"]

    C --> E["For each country:\nregex \\bCOUNTRY\\b\nin query"]
    D --> F["For each city:\nskip if len ≤ 2\nregex \\bCITY\\b\nin query"]

    E --> G{"Match\nfound?"}
    F --> H{"Match\nfound?"}

    G -->|"'India' ✓"| I["detected_countries\n= ['India']"]
    H -->|"'Goa' ✓"| J["detected_cities\n= ['Goa']"]

    I --> K["Filter: countyName\ncontains 'India'"]
    J --> L["Filter: cityName\ncontains 'Goa'"]

    K --> M["Filtered DataFrame"]
    L --> M
```

### Edge cases handled

| Scenario | How It's Handled |
|----------|-----------------|
| City name inside another word (e.g., "Paris" in "comparison") | `\b` word boundaries prevent partial matches |
| Special characters in names (e.g., "St. John's") | `re.escape()` safely escapes regex metacharacters |
| Very short city names (e.g., "Go", "Os") | Cities ≤ 2 chars are skipped entirely |
| No location mentioned | No filter applied; pure semantic ranking used |
| Multiple locations (e.g., "hotels in Paris or London") | Both cities detected; results include either |

---

## 8. API Contract (Schemas)

```mermaid
classDiagram
    class SearchQuery {
        +str query
        +int top_k = 12
    }

    class SearchResponse {
        +str status = "success"
        +SearchParameters search_parameters
        +list~dict~ results
    }

    class SearchParameters {
        +str original_query
        +SearchFilters filters_applied
    }

    class SearchFilters {
        +list~str~ countries
        +list~str~ cities
    }

    SearchResponse --> SearchParameters
    SearchParameters --> SearchFilters
```

### Pydantic validation flow

```mermaid
flowchart LR
    A["Raw JSON body"] --> B["Pydantic parses\nSearchQuery"]
    B -->|"Valid"| C["query = str\ntop_k = int"]
    B -->|"Missing 'query'"| D["422 Error:\nfield required"]
    B -->|"top_k = 'abc'"| E["422 Error:\nint expected"]
    B -->|"No top_k sent"| F["Default: top_k = 12"]
```

---

## 9. Error Handling

| Layer | Error Type | HTTP Code | Handled By |
|-------|-----------|-----------|------------|
| **Pydantic** | Invalid/missing fields | 422 | FastAPI automatic |
| **Route** | Any unhandled exception | 500 | `try/except` in `routes.py` |
| **FAISS** | Index corruption | 500 | Caught by route handler |
| **Data** | Missing CSV | Crash at startup | Fails fast with traceback |

```python
# routes.py — error boundary
try:
    results, countries, cities = semantic_search(...)
    return SearchResponse(...)
except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))
```

---

## 10. Performance Characteristics

### Startup cost (one-time)

| Operation | Time | Memory |
|-----------|------|--------|
| CSV load + sample | ~5s | ~200 MB |
| Model load | ~3s | ~80 MB |
| Encode 10k texts | ~30-60s | ~15 MB (vectors) |
| FAISS index build | < 1s | ~15 MB |
| **Total** | **~40-70s** | **~310 MB** |

### Per-request cost

| Operation | Time |
|-----------|------|
| Query encoding | ~5ms |
| FAISS search (10k vectors) | ~2ms |
| Location detection (regex) | ~10ms |
| DataFrame filtering | ~1ms |
| JSON serialization | ~1ms |
| **Total per request** | **~20ms** |

---

## 11. Comparison: Monolithic vs Modular

### Before (single `api.py` — 84 lines)

```
api.py
├── FastAPI app creation          (lines 9-14)
├── CSV loading & preprocessing   (lines 17-30)
├── Model loading & indexing      (lines 32-36)
├── Pydantic schema               (lines 40-42)
└── Search endpoint               (lines 45-84)
    ├── Query encoding
    ├── FAISS search
    ├── Location detection
    ├── Location filtering
    ├── Description truncation
    └── Response formatting
```

### After (7 files, clear boundaries)

```
main.py          → App creation + startup orchestration
config.py        → All constants in one place
data_loader.py   → Data ingestion (testable independently)
model.py         → ML model + index (swappable)
schemas.py       → API contract (self-documenting)
search.py        → Core logic (unit-testable, no HTTP dependency)
routes.py        → Thin controller (just wiring)
```

### Benefits

| Aspect | Monolithic | Modular |
|--------|-----------|---------|
| **Readability** | Scroll through 84 lines | Open the file you need |
| **Testability** | Must spin up full server | Test `semantic_search()` directly |
| **Team work** | Merge conflicts on one file | Each person owns a module |
| **Swapping models** | Edit deep in the file | Change `MODEL_NAME` in config |
| **Adding endpoints** | Grows the single file | Add to `routes.py`, logic in new module |

---
