"""
Storage repository for the Tausa Municipal Archive OCR pipeline.

Implements the repository pattern to abstract persistence concerns from
the pipeline orchestration layer. Supports incremental TXT progress saves
and structured JSON output with full page-level metadata.
"""

import json
from dataclasses import asdict
from pathlib import Path

from src.config import settings
from src.ocr.pipeline import PipelineResult

class TranscriptionRepository:
    """Persists transcription results to the local filesystem.

    Provides two storage strategies:
    - Incremental TXT: written after every page to prevent data loss on failure.
    - Structured JSON: written on demand with full metadata for downstream RAG ingestion.
    """

    _TXT_SEPARATOR: str = "─" * 70
    _HEADER_TEMPLATE: str = (
        "TRANSCRIPCIÓN OCR — {document_title}\n"
        "Municipio de Tausa, Cundinamarca, Colombia\n"
        "Modelo: {model} | DPI: {dpi}\n"
        "{separator}\n"
    )

    def __init__(self, output_dir: Path) -> None:
        """Initialize the repository and ensure the output directory exists.

        Args:
            output_dir: Base directory where all output files will be written.
        """
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def save_progress(self, result: "PipelineResult", pdf_path: Path) -> None:
        """Write all transcribed pages so far to a plain-text file.

        Overwrites the file on each call so it always reflects current progress.
        Called after every page to ensure no work is lost on interruption.

        Args:
            result: Current pipeline result containing transcribed pages.
            pdf_path: Source PDF path used to derive the output filename.
        """
        output_path = self._resolve_txt_path(pdf_path)
        lines: list[str] = [
            self._HEADER_TEMPLATE.format(
                document_title=result.document_title,
                model=result.model,
                dpi=settings.PDF_DPI,
                separator=self._TXT_SEPARATOR,
            )
        ]

        for page in result.processed_pages:
            lines.append(f"\n{self._TXT_SEPARATOR}")
            lines.append(f"PÁGINA {page.page_number}")
            lines.append(f"{self._TXT_SEPARATOR}\n")
            lines.append(page.transcription)
            lines.append("")

        output_path.write_text("\n".join(lines), encoding="utf-8")

    def save_json(self, result: "PipelineResult", pdf_path: Path) -> None:
        """Persist the full pipeline result as a structured JSON file.

        The JSON output preserves page-level metadata required for RAG ingestion,
        including source file, document title, page numbers, and processing times.

        Args:
            result: Completed pipeline result to serialize.
            pdf_path: Source PDF path used to derive the output filename.
        """
        json_path = self._resolve_json_path(pdf_path)
        payload = asdict(result)
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"✅ JSON saved: {json_path}")

    def _resolve_txt_path(self, pdf_path: Path) -> Path:
        """Derive the .txt output path from the source PDF filename.

        Args:
            pdf_path: Source PDF path.

        Returns:
            Resolved Path for the plain-text transcription file.
        """
        return self._output_dir / f"{pdf_path.stem}_transcription.txt"

    def _resolve_json_path(self, pdf_path: Path) -> Path:
        """Derive the .json output path from the source PDF filename.

        Args:
            pdf_path: Source PDF path.

        Returns:
            Resolved Path for the structured JSON output file.
        """
        return self._output_dir / f"{pdf_path.stem}_transcription.json"