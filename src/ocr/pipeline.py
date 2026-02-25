"""
Pipeline orchestrator for the Tausa Municipal Archive OCR pipeline.

Coordinates PDF loading, page transcription, and result persistence,
following the facade pattern to provide a single entry point for the CLI.
"""

import time
from pathlib import Path

import anthropic

from src.config import settings
from src.models.models import PageResult, PipelineResult
from src.ocr.pdf_processor import load_pdf_pages, parse_page_range
from src.ocr.transcriber import PageTranscriber
from src.storage.repository import TranscriptionRepository


class TranscriptionPipeline:
    """Orchestrates the end-to-end OCR transcription of a PDF document.

    Follows the facade pattern: callers interact only with this class,
    without needing knowledge of the underlying PDF, image, or API modules.
    """

    def __init__(self) -> None:
        """Initialize the pipeline with an authenticated Anthropic client."""
        client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        self._transcriber = PageTranscriber(client)
        self._repository = TranscriptionRepository(settings.OUTPUTS_DIR)

    def run(
        self,
        pdf_path: Path,
        document_title: str,
        page_range: str | None = None,
        save_json: bool = False,
    ) -> PipelineResult:
        """Execute the full transcription pipeline for a PDF document.

        Args:
            pdf_path: Path to the source PDF file.
            document_title: Human-readable title used in prompts and output metadata.
            page_range: Optional page range string (e.g. '1-5', '1,3,7').
                        Defaults to all pages when None.
            save_json: If True, also persist results as a structured JSON file.

        Returns:
            PipelineResult containing all page transcriptions and metadata.
        """
        images = load_pdf_pages(pdf_path)
        total_pages = len(images)
        indices = (
            parse_page_range(page_range, total_pages)
            if page_range
            else list(range(total_pages))
        )

        print(f"📄 {pdf_path.name} — {total_pages} pages detected")
        print(f"   Processing pages: {[i + 1 for i in indices]}\n")

        result = PipelineResult(
            source_file=str(pdf_path),
            document_title=document_title,
            model=settings.CLAUDE_MODEL,
            total_pages=total_pages,
        )

        for idx in indices:
            page_number = idx + 1
            print(f"🔍 Page {page_number}/{total_pages}...", end=" ", flush=True)

            start = time.monotonic()
            transcription = self._transcriber.transcribe(
                image=images[idx],
                page_number=page_number,
                total_pages=total_pages,
                document_title=document_title,
            )
            elapsed = time.monotonic() - start

            print(f"✅ ({elapsed:.1f}s)")

            result.processed_pages.append(
                PageResult(
                    page_number=page_number,
                    transcription=transcription,
                    processing_time_seconds=round(elapsed, 2),
                )
            )

            self._repository.save_progress(result, pdf_path)

            if idx != indices[-1]:
                time.sleep(settings.INTER_PAGE_DELAY_SECONDS)

        if save_json:
            self._repository.save_json(result, pdf_path)

        print(f"\n✅ Done — {len(result.processed_pages)} page(s) transcribed.")
        return result