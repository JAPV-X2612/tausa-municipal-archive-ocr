"""
CLI entry point for the Tausa Municipal Archive OCR transcription pipeline.

Parses command-line arguments and delegates execution to the TranscriptionPipeline
facade. Responsible only for argument parsing and validation — no business logic.

Usage:
    python transcribe.py --pdf assets/docs/tausa_alcaldia_despacho_alcalde_1953_1954.pdf
    python transcribe.py --pdf assets/docs/tausa_alcaldia_despacho_alcalde_1953_1954.pdf --pages 1-3
    python transcribe.py --pdf assets/docs/tausa_alcaldia_despacho_alcalde_1953_1954.pdf --json
"""

import argparse
import sys
from pathlib import Path

from src.config import settings
from src.ocr.pipeline import TranscriptionPipeline


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description="OCR transcription pipeline for historical handwritten municipal records."
    )
    parser.add_argument(
        "--pdf",
        required=True,
        type=Path,
        help="Path to the source PDF file (e.g. assets/docs/tausa_alcaldia_despacho_alcalde_1953_1954.pdf).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Human-readable document title. Defaults to the PDF filename stem.",
    )
    parser.add_argument(
        "--pages",
        type=str,
        default=None,
        help="Page range to process, e.g. '1-5' or '1,3,7'. Defaults to all pages.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Also save results as a structured JSON file with page-level metadata.",
    )
    return parser


def _validate_environment() -> None:
    """Validate required environment variables before starting the pipeline.

    Raises:
        SystemExit: If ANTHROPIC_API_KEY is not configured.
    """
    if not settings.ANTHROPIC_API_KEY:
        print("❌ ANTHROPIC_API_KEY is not set.")
        print("   Export it before running: set ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)


def main() -> None:
    """Parse arguments, validate environment, and execute the transcription pipeline."""
    parser = _build_argument_parser()
    args = parser.parse_args()

    _validate_environment()

    pdf_path: Path = args.pdf
    if not pdf_path.exists():
        print(f"❌ PDF file not found: {pdf_path}")
        sys.exit(1)

    document_title: str = args.title or pdf_path.stem.replace("_", " ").title()

    pipeline = TranscriptionPipeline()
    pipeline.run(
        pdf_path=pdf_path,
        document_title=document_title,
        page_range=args.pages,
        save_json=args.json,
    )


if __name__ == "__main__":
    main()