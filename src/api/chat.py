"""
RAG + Claude chat logic for the Tausa Archive API.

Pipeline for each request:
  1. Fetch RAG_FETCH_CANDIDATES chunks from ChromaDB.
  2. Filter by RAG_MAX_DISTANCE to discard unrelated chunks.
  3. Emit the relevant chunks as citations (SSE) before any text.
  4. Build a grounded context block and call Claude with web search enabled,
     so the model can complement archive findings with public historical context.
  5. Stream Claude's response as SSE text deltas.

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

# Used as the function-level default so direct callers (tests, scripts) also
# pick up the configured value rather than a stale hardcoded number.
_DEFAULT_N_SOURCES: int = settings.RAG_N_RESULTS

# -------------------------------------------------------------------------
# System prompt
# -------------------------------------------------------------------------

_SYSTEM_PROMPT: str = """\
Eres un asistente archivista experto en el archivo histórico municipal de Tausa, \
Cundinamarca, Colombia. Ayudas a funcionarios de la alcaldía a consultar documentos \
históricos digitalizados mediante OCR.

INVENTARIO DEL ARCHIVO DISPONIBLE:
- Administración General de Salinas (1931–1942): Registro tabular de distribución de sal \
(decalitros, valores en pesos, números de informe) por distribuidor. Incluye nombres como \
Pascual Rodríguez, Gilbert González y otros. Organizado por mes y fecha.
- Administración General de Salinas (1942–1948): Continuación del registro anterior. \
Distribuidores como Gargan y Gangaly, Canto y Puars. Datos de volúmenes y valores de sal.
- Libro del Despacho del Alcalde (1925–1928): Libro de posesiones de empleados municipales \
desde febrero de 1925. Actas de posesión de funcionarios.
- Libro del Despacho del Alcalde (1928–1932): Actas de posesión de empleados y funcionarios \
del municipio de Tausa.
- Libro del Despacho del Alcalde (1933): Libro de posesión de empleados, año 1933.
- Libro del Despacho del Alcalde (1953–1954): Contratos municipales de obras y servicios, \
correspondencia oficial. Incluye contratos con particulares firmados por el Personero Municipal.
- Diario Oficial — Estados Unidos de Colombia, Partes 1 y 2: Publicaciones oficiales del \
gobierno nacional colombiano (circa 1865), con decretos del Poder Ejecutivo, comunicaciones \
diplomáticas y balances del Tesoro Nacional. Documento de contexto histórico nacional, \
anterior al período municipal principal.

Tienes acceso a dos fuentes de información:
- CONTEXTO DEL ARCHIVO: Fragmentos de los documentos anteriores (fuente primaria).
- Búsqueda web: Para complementar con contexto histórico general cuando el archivo no \
contiene la información solicitada.

Reglas:
1. El CONTEXTO DEL ARCHIVO es tu fuente primaria. Cita siempre con [FUENTE N].
2. Usa TODOS los fragmentos del archivo disponibles antes de recurrir a la web. \
Si varios fragmentos contienen información parcial, intégralos en una respuesta coherente.
3. Usa la búsqueda web SOLO para:
   - Contexto histórico general sobre Colombia, Cundinamarca o Tausa.
   - Verificar o completar datos del archivo con información de dominio público.
   - Responder cuando el archivo definitivamente no contiene información relevante.
4. Distingue siempre el origen: "Según el archivo..." vs "Según fuentes externas...".
5. Si el archivo tiene texto ilegible o incompleto, indícalo con ⚠️ y recomienda \
la consulta directa del documento original en el archivo físico municipal.
6. Nunca inventes datos del archivo. Si algo no está en los documentos, dilo con claridad.
7. Responde siempre en español con tono profesional y claro.\
"""

# Wrap the system prompt in a cached content block so the Anthropic API
# reuses the prompt tokens across requests (90 % discount on input cost).
# The cache is ephemeral (5-minute TTL) and is refreshed automatically on
# each request that arrives after the cache expires.
_SYSTEM_PROMPT_BLOCKS: list[dict] = [
    {
        "type": "text",
        "text": _SYSTEM_PROMPT,
        "cache_control": {"type": "ephemeral"},
    }
]

_CONTEXT_ITEM_TEMPLATE: str = "[FUENTE {n}] {title} — Página {page}\n{text}\n"


# -------------------------------------------------------------------------
# Public interface
# -------------------------------------------------------------------------


async def stream_chat_response(
    query: str,
    retriever: ArchiveRetriever,
    client: anthropic.AsyncAnthropic,
    n_sources: int = _DEFAULT_N_SOURCES,
) -> AsyncIterator[str]:
    """Yield SSE-formatted strings for a full RAG + Claude chat turn.

    Args:
        query:     User's natural-language question.
        retriever: Initialised ArchiveRetriever (shared across requests).
        client:    Async Anthropic client (shared across requests).
        n_sources: Maximum number of archive chunks shown as citations.

    Yields:
        SSE-formatted strings (``"data: {...}\\n\\n"``).
    """
    try:
        # --- 1. Retrieve and filter ----------------------------------------
        # If the query explicitly names a document, fetch all its chunks in
        # page order (document-aware mode). Otherwise, run the standard
        # semantic similarity search with distance filtering.
        doc_source = retriever.detect_document_reference(query)

        if doc_source:
            context_results: list[RetrievalResult] = await run_in_threadpool(
                retriever.retrieve_by_document, doc_source
            )
            archive_match = bool(context_results)
        else:
            candidates: list[RetrievalResult] = await run_in_threadpool(
                retriever.retrieve, query, settings.RAG_FETCH_CANDIDATES
            )
            archive_match = any(
                r.distance <= settings.RAG_MAX_DISTANCE for r in candidates
            )
            context_results = candidates[:n_sources]

        # --- 2. Emit citations ---------------------------------------------
        yield _sse(
            {
                "type": "citations",
                "sources": [
                    {
                        "document_title": r.document_title,
                        "page_number": r.page_number,
                        "excerpt": r.text[:250],
                        "distance": round(r.distance, 4),
                    }
                    for r in context_results
                ],
            }
        )

        # --- 3. Build context block ----------------------------------------
        context_block = _build_context(context_results, archive_match)
        user_message = f"Pregunta: {query}\n\nCONTEXTO DEL ARCHIVO:\n{context_block}"

        # --- 4. Stream Claude response (with optional web search) ----------
        # Web search requires the beta client namespace; the standard
        # messages.stream() does not accept a `betas` keyword argument.
        common_kwargs: dict = dict(
            model=settings.CLAUDE_MODEL,
            max_tokens=settings.MAX_OUTPUT_TOKENS,
            system=_SYSTEM_PROMPT_BLOCKS,
            messages=[{"role": "user", "content": user_message}],
        )

        if settings.WEB_SEARCH_ENABLED:
            stream_ctx = client.beta.messages.stream(
                **common_kwargs,
                tools=[{"type": "web_search_20250305", "name": "web_search"}],
                betas=["web-search-2025-03-05"],
            )
        else:
            stream_ctx = client.messages.stream(**common_kwargs)

        async with stream_ctx as stream:
            async for text_chunk in stream.text_stream:
                yield _sse({"type": "text", "delta": text_chunk})

        yield _sse({"type": "done"})

    except Exception as exc:  # noqa: BLE001
        yield _sse({"type": "error", "message": str(exc)})


# -------------------------------------------------------------------------
# Private helpers
# -------------------------------------------------------------------------


def _build_context(results: list[RetrievalResult], archive_match: bool) -> str:
    """Format retrieved chunks into the context block injected into the prompt.

    Args:
        results:       Filtered list of RetrievalResult objects.
        archive_match: True when at least one chunk cleared the distance threshold.

    Returns:
        Formatted multi-line string ready to embed in the user message.
    """
    items = "\n\n".join(
        _CONTEXT_ITEM_TEMPLATE.format(
            n=i + 1,
            title=r.document_title,
            page=r.page_number,
            text=r.text,
        )
        for i, r in enumerate(results)
    )

    if not archive_match:
        items = (
            "[NOTA INTERNA: Los siguientes fragmentos son los más cercanos encontrados "
            "en el archivo, pero su similitud con la pregunta es baja. Úsalos solo como "
            "referencia de contexto y complementa con búsqueda web si es necesario.]\n\n"
            + items
        )

    return items


def _sse(payload: dict) -> str:
    """Format a dict as a Server-Sent Event data line."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
