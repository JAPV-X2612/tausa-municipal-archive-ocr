"""
RAG + Claude chat logic for the Tausa Archive API.

Retrieves relevant archive chunks, builds a grounded prompt, and streams
the Claude response as Server-Sent Events (SSE). Each SSE stream starts
with a citations event so the frontend can render the source panel before
the answer text arrives.

SSE event types emitted:
  {"type": "citations", "sources": [...]}   — sent once, before any text
  {"type": "text",      "delta":   "..."}   — one per streamed text chunk
  {"type": "done"}                          — signals end of stream
  {"type": "error",     "message": "..."}   — sent on unexpected failure
"""

import json
from typing import AsyncIterator

import anthropic
from fastapi.concurrency import run_in_threadpool

from src.config import settings
from src.rag.retriever import ArchiveRetriever, RetrievalResult

# -------------------------------------------------------------------------
# System prompt
# -------------------------------------------------------------------------

_SYSTEM_PROMPT: str = """Eres un asistente archivista experto en el archivo histórico municipal \
de Tausa, Cundinamarca, Colombia. Ayudas a funcionarios de la alcaldía a consultar documentos \
históricos que abarcan el período 1925–1954.

Reglas estrictas:
1. Responde ÚNICAMENTE con base en el contexto del archivo proporcionado a continuación.
2. Si la respuesta no se encuentra en el contexto, indícalo claramente: \
"No encontré información sobre eso en los documentos disponibles."
3. Cita las fuentes usando la notación [FUENTE N] en tu respuesta.
4. Responde siempre en español, con un tono profesional y claro.
5. No inventes información ni extrapoles más allá de lo que dicen los documentos."""

_CONTEXT_ITEM_TEMPLATE: str = (
    "[FUENTE {n}] {title} — Página {page}\n"
    "{text}\n"
)


# -------------------------------------------------------------------------
# Public interface
# -------------------------------------------------------------------------


async def stream_chat_response(
    query: str,
    retriever: ArchiveRetriever,
    client: anthropic.AsyncAnthropic,
    n_sources: int = 5,
) -> AsyncIterator[str]:
    """Yield SSE-formatted strings for a full RAG + Claude chat turn.

    Runs the synchronous retrieval step in a thread pool to avoid blocking
    the async event loop, then streams the Claude response.

    Args:
        query:     User's natural-language question.
        retriever: Initialised ArchiveRetriever (shared across requests).
        client:    Async Anthropic client (shared across requests).
        n_sources: Number of archive chunks to retrieve.

    Yields:
        SSE-formatted strings (``"data: {...}\\n\\n"``).
    """
    try:
        # Retrieval is sync (ChromaDB + sentence-transformers); run off the event loop.
        results: list[RetrievalResult] = await run_in_threadpool(
            retriever.retrieve, query, n_sources
        )

        # Emit citations before any text so the frontend can render the panel immediately.
        citations = [
            {
                "document_title": r.document_title,
                "page_number": r.page_number,
                "excerpt": r.text[:250],
                "distance": round(r.distance, 4),
            }
            for r in results
        ]
        yield _sse({"type": "citations", "sources": citations})

        # Build the grounded context block.
        context_block = "\n\n".join(
            _CONTEXT_ITEM_TEMPLATE.format(
                n=i + 1,
                title=r.document_title,
                page=r.page_number,
                text=r.text,
            )
            for i, r in enumerate(results)
        )
        user_message = f"Pregunta: {query}\n\nCONTEXTO DEL ARCHIVO:\n{context_block}"

        # Stream the Claude response.
        async with client.messages.stream(
            model=settings.CLAUDE_MODEL,
            max_tokens=settings.MAX_OUTPUT_TOKENS,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        ) as stream:
            async for text_chunk in stream.text_stream:
                yield _sse({"type": "text", "delta": text_chunk})

        yield _sse({"type": "done"})

    except Exception as exc:  # noqa: BLE001
        yield _sse({"type": "error", "message": str(exc)})


# -------------------------------------------------------------------------
# Private helpers
# -------------------------------------------------------------------------


def _sse(payload: dict) -> str:
    """Format a dict as a Server-Sent Event data line."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
