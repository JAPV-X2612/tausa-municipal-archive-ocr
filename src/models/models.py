"""
Shared data models for the Tausa Municipal Archive OCR pipeline.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class TranscriptionResult:
    """Raw output from a single Claude Vision API call.

    Separates the API response payload from the normalised transcription
    stored in PageResult, keeping the transcriber free of post-processing
    concerns.
    """

    text: str
    input_tokens: int
    output_tokens: int


@dataclass
class PageResult:
    """Holds the normalised transcription and metadata for a single document page."""

    page_number: int
    transcription: str
    processing_time_seconds: float
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class PipelineResult:
    """Aggregated result of a full document transcription run."""

    source_file: str
    document_title: str
    model: str
    total_pages: int
    processed_pages: list[PageResult] = field(default_factory=list)
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def total_input_tokens(self) -> int:
        """Sum of input tokens consumed across all processed pages."""
        return sum(p.input_tokens for p in self.processed_pages)

    @property
    def total_output_tokens(self) -> int:
        """Sum of output tokens generated across all processed pages."""
        return sum(p.output_tokens for p in self.processed_pages)
