"""
FastAPI application for the Tausa Municipal Archive RAG backend.

Exposes two endpoints:
  GET  /health  — liveness check with vector store chunk count.
  POST /chat    — RAG-augmented chat with streaming SSE response.

The ArchiveRetriever and AsyncAnthropic client are initialised once at
startup via the lifespan context manager and shared across all requests
through app.state, avoiding per-request model-loading overhead.
"""

from contextlib import asynccontextmanager

import anthropic
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from src.api.chat import stream_chat_response
from src.api.schemas import ChatRequest, HealthResponse
from src.config import settings
from src.rag.retriever import ArchiveRetriever


# -------------------------------------------------------------------------
# Lifespan: initialise shared resources once at startup
# -------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the retriever and Anthropic client once at server startup."""
    app.state.retriever = ArchiveRetriever()
    app.state.claude = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
    yield
    # No explicit teardown required for these resources.


# -------------------------------------------------------------------------
# Application
# -------------------------------------------------------------------------

app = FastAPI(
    title="Tausa Municipal Archive API",
    description="RAG-powered chat interface for historical archive documents (1925–1954).",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# -------------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health(request: Request) -> HealthResponse:
    """Return server status and the total number of indexed archive chunks."""
    count: int = request.app.state.retriever._collection.count()
    return HealthResponse(status="ok", collection_count=count)


@app.post("/chat", tags=["chat"])
async def chat(request: Request, body: ChatRequest) -> StreamingResponse:
    """Stream a RAG-augmented answer for a natural-language archive query.

    The response is a Server-Sent Events stream. The first event is always
    ``{"type": "citations", "sources": [...]}`` followed by one or more
    ``{"type": "text", "delta": "..."}`` events, and a final
    ``{"type": "done"}`` event.
    """
    return StreamingResponse(
        stream_chat_response(
            query=body.message,
            retriever=request.app.state.retriever,
            client=request.app.state.claude,
            n_sources=body.n_sources,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disables Nginx response buffering.
        },
    )
