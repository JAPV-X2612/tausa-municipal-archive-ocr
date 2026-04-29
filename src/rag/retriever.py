"""
RAG retrieval module for the Tausa Municipal Archive.

Provides two retrieval strategies:

1. Semantic search (``retrieve``): embeds the query and finds the
   most similar chunks across the entire collection. Used for specific,
   factual queries (e.g. "¿Quién era el alcalde en 1935?").

2. Document-aware retrieval (``retrieve_by_document``): fetches ALL
   chunks from a specific document, ordered by page. Used when the
   query explicitly names a document (e.g. "Dame un resumen del
   documento Administración General de Salinas de 1931 a 1942").

The active strategy is selected in ``chat.py`` by calling
``detect_document_reference`` before choosing the retrieval path.
"""

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import chromadb

from src.config import settings
from src.rag._constants import COLLECTION_NAME
from src.rag.embedder import ArchiveEmbedder

# Maximum number of chunks fetched from a single document in document-
# aware mode. 100 covers even the largest documents in the archive while
# keeping prompt size within reason (~30 000 characters of context).
_MAX_DOCUMENT_CHUNKS: int = 100


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------


def _normalise(text: str) -> str:
    """Lowercase, strip diacritics and punctuation for fuzzy title matching."""
    text = text.lower()
    # Decompose accented characters (á → a + combining accent)
    text = unicodedata.normalize("NFD", text)
    # Drop all combining (accent) characters
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    # Replace non-alphanumeric characters with spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# -------------------------------------------------------------------------
# Public types
# -------------------------------------------------------------------------


@dataclass
class RetrievalResult:
    """A single retrieved chunk with its provenance metadata.

    Fields:
        text:           The chunk text returned from the vector store.
        document_title: Human-readable title of the source document.
        source_file:    Relative path to the original PDF.
        page_number:    Page within the source document (1-based).
        chunk_index:    Position of this chunk within its page (0-based).
        distance:       Cosine distance from the query (lower = more similar).
                        Set to 0.0 for document-aware results (no query vector).
    """

    text: str
    document_title: str
    source_file: str
    page_number: int
    chunk_index: int
    distance: float


# -------------------------------------------------------------------------
# Retriever
# -------------------------------------------------------------------------


class ArchiveRetriever:
    """Retrieves relevant archive passages using semantic or document-aware search.

    Instantiate once per process (e.g., at API startup) to avoid repeated
    model-loading overhead. The document index is built at construction time
    from the ChromaDB metadata so that ``detect_document_reference`` is a
    pure in-memory operation during request handling.
    """

    def __init__(self) -> None:
        """Load the ChromaDB collection, embedding model, and document index."""
        client = chromadb.PersistentClient(path=str(settings.CHROMA_DB_DIR))
        self._collection = client.get_collection(name=COLLECTION_NAME)
        self._embedder = ArchiveEmbedder()
        self._doc_index: list[dict] = self._build_doc_index()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def detect_document_reference(self, query: str) -> str | None:
        """Return the source_file key if the query names a known document.

        Normalises both the query and each document's title / file stem and
        checks for substring containment. This is intentionally simple: it
        avoids an extra API call while handling the most common patterns like
        quoted titles and file-stem references.

        Args:
            query: Natural-language question from the user.

        Returns:
            The ``source_file`` metadata value for the matched document, or
            ``None`` if no document is explicitly named.
        """
        norm_q = _normalise(query)
        for doc in self._doc_index:
            # Match against the human-readable title
            if doc["norm_title"] and doc["norm_title"] in norm_q:
                return doc["source_file"]
            # Match against the filename stem (e.g. "admin_salinas_1931_1942")
            stem = doc["norm_stem"]
            if stem and len(stem) > 10 and stem in norm_q:
                return doc["source_file"]
        return None

    def retrieve(self, query: str, n_results: int = 5) -> list[RetrievalResult]:
        """Find the most semantically relevant archive chunks for a query.

        Args:
            query:     Natural-language question or search phrase.
            n_results: Number of top results to return.

        Returns:
            List of RetrievalResult instances ordered by relevance (closest first).
        """
        query_embedding = self._embedder.embed([query])[0]
        raw = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        results: list[RetrievalResult] = []
        for doc, meta, dist in zip(
            raw["documents"][0],
            raw["metadatas"][0],
            raw["distances"][0],
        ):
            results.append(
                RetrievalResult(
                    text=doc,
                    document_title=meta["document_title"],
                    source_file=meta["source_file"],
                    page_number=meta["page_number"],
                    chunk_index=meta["chunk_index"],
                    distance=dist,
                )
            )
        return results

    def retrieve_by_document(
        self, source_file: str, max_chunks: int = _MAX_DOCUMENT_CHUNKS
    ) -> list[RetrievalResult]:
        """Fetch all chunks from a specific document, ordered by page position.

        Bypasses the vector similarity search and returns every chunk stored
        for the given document. Suitable for full-document summaries and
        comprehensive queries about a named source.

        Args:
            source_file: The ``source_file`` metadata value that identifies
                         the document (returned by ``detect_document_reference``).
            max_chunks:  Upper limit on the number of chunks returned.

        Returns:
            List of RetrievalResult instances ordered by page_number then
            chunk_index, with distance=0.0 (no query vector used).
        """
        raw = self._collection.get(
            where={"source_file": {"$eq": source_file}},
            include=["documents", "metadatas"],
            limit=max_chunks,
        )
        results = [
            RetrievalResult(
                text=doc,
                document_title=meta["document_title"],
                source_file=meta["source_file"],
                page_number=meta["page_number"],
                chunk_index=meta["chunk_index"],
                distance=0.0,
            )
            for doc, meta in zip(raw["documents"], raw["metadatas"])
        ]
        results.sort(key=lambda r: (r.page_number, r.chunk_index))
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_doc_index(self) -> list[dict]:
        """Build an in-memory index of distinct documents for title matching.

        Fetches only the metadata (no embeddings or document text) from the
        collection once at startup. Each entry stores the original title and
        file path alongside their normalised forms for fast substring matching.
        """
        raw = self._collection.get(include=["metadatas"])
        seen: dict[str, dict] = {}
        for meta in raw["metadatas"]:
            key = meta["source_file"]
            if key not in seen:
                stem = Path(key).stem.replace("_transcription", "")
                seen[key] = {
                    "document_title": meta["document_title"],
                    "source_file": key,
                    "norm_title": _normalise(meta["document_title"]),
                    "norm_stem": _normalise(stem),
                }
        return list(seen.values())
