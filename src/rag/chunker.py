"""
Text chunking for the RAG ingestion pipeline.

Splits a single page's transcription into paragraph-level chunks suitable
for embedding and semantic retrieval.
"""

_MIN_CHUNK_CHARS: int = 30


def chunk_page(transcription: str) -> list[str]:
    """Split a page transcription into paragraph-level chunks.

    Splits on double newlines (paragraph breaks produced by the OCR
    normaliser) and discards chunks that are too short to carry meaning.
    Keeping whole paragraphs together — rather than fixed-size windows —
    preserves the coherence of table rows and narrative passages, which
    is important for accurate retrieval on historical ledger documents.

    Args:
        transcription: Full transcription text of a single page.

    Returns:
        List of non-empty text chunks suitable for embedding.
    """
    chunks = [chunk.strip() for chunk in transcription.split("\n\n")]
    return [chunk for chunk in chunks if len(chunk) >= _MIN_CHUNK_CHARS]
