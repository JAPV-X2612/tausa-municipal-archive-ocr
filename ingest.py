"""
CLI entry point for ingesting OCR transcriptions into ChromaDB.

Reads all *_transcription.json files from the outputs directory, splits
each page into paragraph-level chunks, embeds them with a multilingual
sentence-transformers model, and upserts into a persistent ChromaDB
collection.

Usage:
    # Ingest all transcription JSONs (safe to re-run; uses upsert):
    python ingest.py

    # Full re-index: clear the collection first, then ingest:
    python ingest.py --reset

    # Ingest from a custom outputs directory:
    python ingest.py --outputs-dir path/to/outputs
"""

import argparse
from pathlib import Path

from src.config import settings
from src.rag.ingester import ArchiveIngester


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest OCR transcription JSONs into ChromaDB for RAG retrieval.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=settings.OUTPUTS_DIR,
        metavar="DIR",
        help="Directory containing *_transcription.json files (default: outputs/).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete all existing vectors before ingesting. Use for a full re-index.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    ingester = ArchiveIngester()

    if args.reset:
        ingester.reset()

    ingester.ingest_all(args.outputs_dir)


if __name__ == "__main__":
    main()
