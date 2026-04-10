from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re

# 1. Initialize the API Server
app = FastAPI(
    title="Semantic Hotel Search API",
    description="An AI-powered B2B backend for hotel recommendations.",
    version="1.0.0"
)

# 2. Load the Engine into Server Memory (Happens once when server starts)
print("Loading Model and Data. Please wait...")
try:
    df = pd.read_csv('hotels.csv', encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv('hotels.csv', encoding='latin1')

df = df.sample(n=10000, random_state=42)
df.columns = df.columns.str.strip()
df = df.fillna('')
df['Search_Text'] = df['HotelName'].astype(str) + " " + \
                    df['Description'].astype(str) + " " + \
                    df['HotelFacilities'].astype(str) + " " + \
                    df['cityName'].astype(str) + " " + \
                    df['countyName'].astype(str)

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df['Search_Text'].tolist(), convert_to_tensor=False)
embeddings = np.array(embeddings).astype('float32')
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
print("✅ Server Ready! Waiting for requests...")

# 3. Define what the incoming JSON request should look like
class SearchQuery(BaseModel):
    query: str
    top_k: int = 12

# 4. The API Endpoint
@app.post("/search")
async def search_hotels(request: SearchQuery):
    try:
        # Convert incoming query to vector
        query_vector = model.encode([request.query])
        query_vector = np.array(query_vector).astype('float32')
        
        # Search Vector Database
        distances, indices = index.search(query_vector, len(df))
        results_df = df.iloc[indices[0]].copy()
        
        # Smart Location Filters
        query_lower = request.query.lower()
        all_countries = [str(c).strip() for c in df['countyName'].unique() if str(c).strip()]
        detected_countries = [c for c in all_countries if re.search(rf"\b{re.escape(c.lower())}\b", query_lower)]
        all_cities = [str(c).strip() for c in df['cityName'].unique() if str(c).strip()]
        detected_cities = [c for c in all_cities if len(c) > 2 and re.search(rf"\b{re.escape(c.lower())}\b", query_lower)]

        if detected_countries:
            results_df = results_df[results_df['countyName'].astype(str).str.contains('|'.join(detected_countries), case=False, na=False)]
        if detected_cities:
            results_df = results_df[results_df['cityName'].astype(str).str.contains('|'.join(detected_cities), case=False, na=False)]

        # Get top matches and format for JSON response
        top_matches = results_df.head(request.top_k)
        
        # Clean the descriptions just like we did in the UI
        top_matches['Description'] = top_matches['Description'].apply(lambda x: str(x).replace('\n', ' ').replace('\r', ' ').strip()[:300] + "...")
        
        return {
            "status": "success",
            "search_parameters": {
                "original_query": request.query,
                "filters_applied": {"countries": detected_countries, "cities": detected_cities}
            },
            "results": top_matches[['HotelName', 'cityName', 'countyName', 'HotelRating', 'HotelFacilities', 'Description']].to_dict(orient="records")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))