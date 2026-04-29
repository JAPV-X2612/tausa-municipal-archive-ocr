"""Shared constants for the RAG ingestion and retrieval modules."""

# ChromaDB collection that stores all archive chunks.
# Must be identical between ingestion and retrieval.
COLLECTION_NAME: str = "tausa_archive"
