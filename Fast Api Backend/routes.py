"""
API route definitions.
"""

from fastapi import APIRouter, HTTPException

from schemas import SearchQuery, SearchResponse, SearchParameters, SearchFilters
from search import semantic_search

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search_hotels(request: SearchQuery):
    """
    Search for hotels using natural language.

    **Required:**
    - `query` (str): Natural language search query

    **Optional strict filters (AND conditions):**
    - `country` (str | null): Exact country name — only hotels in this country are returned
    - `city` (str | null): Exact city name — only hotels in this city are returned
    - `min_rating` (float, 0-5): Minimum star rating filter
    - `top_k` (int, default 12): Number of results to return

    If `country` or `city` is null/omitted, the system falls back to
    auto-detecting location names from the query text.
    """
    from main import model, index, df

    try:
        search_result = semantic_search(
            query=request.query,
            top_k=request.top_k,
            model=model,
            index=index,
            df=df,
            strict_country=request.country,
            strict_city=request.city,
            min_rating=request.min_rating,
        )

        return SearchResponse(
            status="success",
            search_parameters=SearchParameters(
                original_query=request.query,
                filters_applied=SearchFilters(
                    strict_country=search_result["strict_country"],
                    strict_city=search_result["strict_city"],
                    min_rating=search_result["min_rating"],
                    auto_detected_countries=search_result["auto_detected_countries"],
                    auto_detected_cities=search_result["auto_detected_cities"],
                ),
            ),
            results=search_result["results"],
            result_count=len(search_result["results"]),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
