"""
ChromaDB ingestion pipeline for the Tausa Municipal Archive.

Reads all JSON transcription files produced by the OCR pipeline, splits
each page into paragraph-level chunks, embeds them with a multilingual
sentence-transformers model, and upserts into a persistent ChromaDB
collection. The operation is idempotent: re-running it updates existing
chunks without creating duplicates.
"""

import json
from pathlib import Path

import chromadb
from chromadb import Collection

from src.config import settings
from src.rag._constants import COLLECTION_NAME
from src.rag.chunker import chunk_page
from src.rag.embedder import ArchiveEmbedder

# Maximum number of vectors sent in a single ChromaDB upsert call.
# Keeps memory usage bounded when processing large documents.
_UPSERT_BATCH_SIZE: int = 500


class ArchiveIngester:
    """Ingests OCR transcription JSONs into the ChromaDB vector store.

    Each JSON page is split into paragraph chunks, embedded, and stored
    with metadata (document title, source file, page number) that the
    retriever uses to generate source citations.
    """

    def __init__(self) -> None:
        """Initialise the ChromaDB client, collection, and embedding model."""
        self._client = chromadb.PersistentClient(path=str(settings.CHROMA_DB_DIR))
        self._collection: Collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._embedder = ArchiveEmbedder()

    def ingest_all(self, outputs_dir: Path) -> None:
        """Ingest all transcription JSON files found in the given directory.

        Args:
            outputs_dir: Directory containing ``*_transcription.json`` files.
        """
        json_files = sorted(outputs_dir.glob("*_transcription.json"))
        if not json_files:
            print(f"No transcription JSON files found in {outputs_dir}")
            return

        for json_path in json_files:
            self._ingest_file(json_path)

        total = self._collection.count()
        print(f"\n✅ Ingestion complete. Collection '{COLLECTION_NAME}' now has {total:,} chunks.")

    def reset(self) -> None:
        """Delete all vectors from the collection and start fresh.

        Safe to call before ``ingest_all`` when a full re-index is needed
        (e.g., after re-transcribing documents with a different model).
        """
        self._client.delete_collection(COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"Collection '{COLLECTION_NAME}' has been reset.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ingest_file(self, json_path: Path) -> None:
        """Chunk, embed, and upsert a single transcription JSON file.

        Args:
            json_path: Path to a ``*_transcription.json`` file.
        """
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        doc_title: str = payload["document_title"]
        source_file: str = payload["source_file"]
        total_pages: int = payload["total_pages"]
        pages: list[dict] = payload["processed_pages"]

        print(f"\n📂 {json_path.name}")
        print(f"   {doc_title} — {len(pages)} page(s)")

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        for page in pages:
            page_number: int = page["page_number"]
            chunks = chunk_page(page["transcription"])

            for chunk_idx, chunk_text in enumerate(chunks):
                chunk_id = f"{json_path.stem}_p{page_number:04d}_c{chunk_idx:03d}"
                ids.append(chunk_id)
                documents.append(chunk_text)
                metadatas.append(
                    {
                        "document_title": doc_title,
                        "source_file": source_file,
                        "page_number": page_number,
                        "total_pages": total_pages,
                        "chunk_index": chunk_idx,
                    }
                )

        if not ids:
            print("   ⚠️  No chunks produced — skipping.")
            return

        print(f"   Embedding {len(ids):,} chunks...", end=" ", flush=True)
        embeddings = self._embedder.embed(documents)
        print("done.")

        for i in range(0, len(ids), _UPSERT_BATCH_SIZE):
            self._collection.upsert(
                ids=ids[i : i + _UPSERT_BATCH_SIZE],
                documents=documents[i : i + _UPSERT_BATCH_SIZE],
                embeddings=embeddings[i : i + _UPSERT_BATCH_SIZE],
                metadatas=metadatas[i : i + _UPSERT_BATCH_SIZE],
            )

        print(f"   ✅ {len(ids):,} chunks upserted.")
