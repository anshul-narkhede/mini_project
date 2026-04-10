"""
Pydantic models for API request and response validation.
"""

from pydantic import BaseModel, Field
from config import DEFAULT_TOP_K


class SearchQuery(BaseModel):
    """Incoming search request body."""
    query: str = Field(..., description="Natural language search query")
    top_k: int = Field(0, ge=0, description="Number of results to return. 0 means return ALL matching results.")

    # --- Optional strict location filters (AND conditions) ---
    country: str | None = Field(None, description="Exact country name to filter by (e.g., 'India'). Leave null to skip.")
    city: str | None = Field(None, description="Exact city name to filter by (e.g., 'Goa'). Leave null to skip.")
    min_rating: float = Field(0.0, ge=0.0, le=5.0, description="Minimum star rating (0-5). Default 0 means no filter.")


class SearchFilters(BaseModel):
    """Location filters applied — both user-provided and auto-detected."""
    strict_country: str | None = Field(None, description="Country set explicitly by the user")
    strict_city: str | None = Field(None, description="City set explicitly by the user")
    min_rating: float = Field(0.0, description="Minimum rating filter applied")
    auto_detected_countries: list[str] = Field(default_factory=list, description="Countries auto-detected from query text")
    auto_detected_cities: list[str] = Field(default_factory=list, description="Cities auto-detected from query text")


class SearchParameters(BaseModel):
    """Echo of the parsed search parameters."""
    original_query: str
    filters_applied: SearchFilters


class SearchResponse(BaseModel):
    """Top-level response envelope."""
    status: str = "success"
    search_parameters: SearchParameters
    results: list[dict]
    result_count: int = Field(0, description="Number of results returned")
