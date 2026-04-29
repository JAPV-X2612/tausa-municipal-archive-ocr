"""
Pydantic request and response schemas for the Tausa Archive API.
"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Incoming chat message from the frontend."""

    message: str = Field(..., min_length=1, max_length=2000)
    n_sources: int = Field(5, ge=1, le=20, description="Number of archive chunks to retrieve.")


class SourceCitation(BaseModel):
    """Metadata for a single retrieved archive chunk shown in the citation panel."""

    document_title: str
    page_number: int
    excerpt: str = Field(description="First 250 characters of the retrieved chunk.")
    distance: float = Field(description="Cosine distance from the query (lower = more relevant).")


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str
    collection_count: int = Field(description="Total number of chunks in the vector store.")
