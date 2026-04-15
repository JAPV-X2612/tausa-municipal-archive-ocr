"""
Pipeline orchestrator for the Tausa Municipal Archive OCR pipeline.

Coordinates PDF loading, page transcription, normalisation, and result
persistence, following the facade pattern to provide a single entry point
for the CLI.
"""

import time
from pathlib import Path

import anthropic

from src.config import settings
from src.models.models import PageResult, PipelineResult
from src.ocr.pdf_processor import load_pdf_pages, parse_page_range
from src.ocr.text_normalizer import normalize_transcription
from src.ocr.transcriber import PageTranscriber
from src.storage.repository import TranscriptionRepository

# Approximate cost per token for Claude Sonnet 4.6 (USD).
# Used only for the informational cost estimate printed at the end of each run.
_INPUT_COST_PER_TOKEN: float = 3.00 / 1_000_000   # $3.00 per 1M input tokens
_OUTPUT_COST_PER_TOKEN: float = 15.00 / 1_000_000  # $15.00 per 1M output tokens


class TranscriptionPipeline:
    """Orchestrates the end-to-end OCR transcription of a PDF document.

    Follows the facade pattern: callers interact only with this class,
    without needing knowledge of the underlying PDF, image, normalisation,
    or API modules.
    """

    def __init__(self) -> None:
        """Initialise the pipeline with an authenticated Anthropic client."""
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

        For each page the pipeline:
        1. Converts the PDF page to an enhanced PIL image.
        2. Sends it to the Claude Vision API.
        3. Normalises the raw transcription (removes hyphenated line-breaks,
           collapses excess blank lines).
        4. Persists progress incrementally to a plain-text file after every page
           so that no work is lost if the pipeline is interrupted.
        5. When ``save_json`` is True, also updates the structured JSON file
           after every page for the same crash-safety guarantee.

        Args:
            pdf_path: Path to the source PDF file.
            document_title: Human-readable title used in prompts and output metadata.
            page_range: Optional page range string (e.g. ``'1-5'``, ``'1,3,7'``).
                        Defaults to all pages when ``None``.
            save_json: If ``True``, persist results as a structured JSON file
                       after every page (incremental) in addition to the TXT file.

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
            transcription_result = self._transcriber.transcribe(
                image=images[idx],
                page_number=page_number,
                total_pages=total_pages,
                document_title=document_title,
            )
            elapsed = time.monotonic() - start

            normalised_text = normalize_transcription(transcription_result.text)

            print(
                f"✅ ({elapsed:.1f}s | "
                f"in: {transcription_result.input_tokens:,} / "
                f"out: {transcription_result.output_tokens:,} tokens)"
            )

            result.processed_pages.append(
                PageResult(
                    page_number=page_number,
                    transcription=normalised_text,
                    processing_time_seconds=round(elapsed, 2),
                    input_tokens=transcription_result.input_tokens,
                    output_tokens=transcription_result.output_tokens,
                )
            )

            self._repository.save_progress(result, pdf_path)
            if save_json:
                self._repository.save_json(result, pdf_path)

            if idx != indices[-1]:
                time.sleep(settings.INTER_PAGE_DELAY_SECONDS)

        self._print_summary(result, pdf_path, save_json)
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _print_summary(
        self, result: PipelineResult, pdf_path: Path, save_json: bool
    ) -> None:
        """Print a cost and token summary for the completed pipeline run.

        Args:
            result: Completed pipeline result containing per-page token counts.
            pdf_path: Source PDF path, used to derive the JSON output path.
            save_json: Whether the JSON output file was written.
        """
        pages = len(result.processed_pages)
        total_in = result.total_input_tokens
        total_out = result.total_output_tokens
        estimated_cost = (
            total_in * _INPUT_COST_PER_TOKEN
            + total_out * _OUTPUT_COST_PER_TOKEN
        )

        print(f"\n✅ Done — {pages} page(s) transcribed.")
        print(f"   Tokens    → input: {total_in:,}  |  output: {total_out:,}")
        print(f"   Est. cost → ${estimated_cost:.4f} USD")
        if save_json:
            json_path = self._repository.resolve_json_path(pdf_path)
            print(f"   JSON      → {json_path}")
