"""
Text chunking for the RAG ingestion pipeline.

Splits a single page's transcription into paragraph-level chunks suitable
for embedding and semantic retrieval. Chunks that are too short or consist
almost entirely of uppercase text (institutional headers, stamps) are
discarded because they add noise to the vector index without carrying
queryable content.
"""

# Minimum character length for a chunk to be indexed.
# Chunks shorter than this are typically date stamps, reference codes,
# or partial lines with no standalone meaning.
_MIN_CHUNK_CHARS: int = 80

# If more than this fraction of alphabetic characters are uppercase, the
# chunk is treated as a header or stamp and excluded.
# 0.85 is conservative enough to pass normal sentences that start with
# proper nouns while still catching all-caps institutional headings.
_MAX_UPPERCASE_RATIO: float = 0.85


def _is_header_chunk(text: str) -> bool:
    """Return True if the chunk looks like a header or institutional stamp.

    Checks the ratio of uppercase to total alphabetic characters. All-caps
    strings (e.g. "ADMINISTRACIÓN GENERAL DE SALINAS") are noise for
    semantic search because they appear on almost every page.
    """
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return True
    return sum(1 for c in letters if c.isupper()) / len(letters) > _MAX_UPPERCASE_RATIO


def chunk_page(transcription: str) -> list[str]:
    """Split a page transcription into paragraph-level chunks.

    Splits on double newlines (paragraph breaks produced by the OCR
    normaliser) and discards chunks that are too short or are pure headers.
    Keeping whole paragraphs together — rather than fixed-size windows —
    preserves the coherence of table rows and narrative passages, which
    is important for accurate retrieval on historical ledger documents.

    Args:
        transcription: Full transcription text of a single page.

    Returns:
        List of non-empty, content-bearing text chunks suitable for embedding.
    """
    chunks = [chunk.strip() for chunk in transcription.split("\n\n")]
    return [
        chunk
        for chunk in chunks
        if len(chunk) >= _MIN_CHUNK_CHARS and not _is_header_chunk(chunk)
    ]
