"""
Shared data models for the Tausa Municipal Archive OCR pipeline.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class PageResult:
    """Holds the transcription result and metadata for a single document page."""

    page_number: int
    transcription: str
    processing_time_seconds: float


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