"""
RAG retrieval module for the Tausa Municipal Archive.

Provides semantic search over the ChromaDB vector store, returning
relevant archive passages with full source metadata for citation display.
"""

from dataclasses import dataclass

import chromadb

from src.config import settings
from src.rag._constants import COLLECTION_NAME
from src.rag.embedder import ArchiveEmbedder


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
    """

    text: str
    document_title: str
    source_file: str
    page_number: int
    chunk_index: int
    distance: float


class ArchiveRetriever:
    """Retrieves relevant archive passages for a natural-language query.

    Embeds the query with the same model used during ingestion and
    performs cosine similarity search against the persistent ChromaDB
    collection. Instantiate once per process (e.g., at API startup) to
    avoid repeated model-loading overhead.
    """

    def __init__(self) -> None:
        """Load the ChromaDB collection and initialise the embedding model."""
        client = chromadb.PersistentClient(path=str(settings.CHROMA_DB_DIR))
        self._collection = client.get_collection(name=COLLECTION_NAME)
        self._embedder = ArchiveEmbedder()

    def retrieve(self, query: str, n_results: int = 5) -> list[RetrievalResult]:
        """Find the most semantically relevant archive chunks for a query.

        Args:
            query:     Natural-language question or search phrase in Spanish or English.
            n_results: Number of top results to return (default: 5).

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
