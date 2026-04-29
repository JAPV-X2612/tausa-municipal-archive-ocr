"""
Pydantic request and response schemas for the Tausa Archive API.
"""

from pydantic import BaseModel, Field

from src.config import settings


class ChatRequest(BaseModel):
    """Incoming chat message from the frontend."""

    message: str = Field(..., min_length=1, max_length=2000)
    # n_sources is optional: if omitted by the client the server default
    # (RAG_N_RESULTS env var) is used, keeping RAG tuning server-side.
    n_sources: int = Field(
        default=settings.RAG_N_RESULTS,
        ge=1,
        le=50,
        description="Number of archive chunks to retrieve. Defaults to RAG_N_RESULTS.",
    )


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
