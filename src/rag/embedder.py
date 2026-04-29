"""
Sentence-Transformers embedding wrapper for the Tausa archive RAG pipeline.

Loads the multilingual model once and exposes a simple encode interface
used by both the ingester (batch) and the retriever (single query).
"""

from sentence_transformers import SentenceTransformer

# Free multilingual model with strong Spanish performance.
# Max sequence length: 128 tokens (~500–600 characters).
# Dimension: 768. Model size on disk: ~420 MB.
_MODEL_NAME: str = "paraphrase-multilingual-mpnet-base-v2"


class ArchiveEmbedder:
    """Wraps a SentenceTransformer model for text embedding.

    Loads the model on construction and reuses it across all encode calls.
    Instantiate once per process to avoid repeated model loading overhead.
    """

    def __init__(self) -> None:
        """Load the sentence-transformers model from local cache or HuggingFace Hub."""
        self._model = SentenceTransformer(_MODEL_NAME)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Encode a list of strings into dense embedding vectors.

        Args:
            texts: One or more strings to encode.

        Returns:
            List of float vectors, one per input string. Each vector has
            768 dimensions matching the model's output size.
        """
        return self._model.encode(texts, show_progress_bar=False).tolist()
