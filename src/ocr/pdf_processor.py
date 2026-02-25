"""
PDF processor module for the Tausa Municipal Archive OCR pipeline.

Responsible for converting PDF documents into sequences of PIL images,
applying the single responsibility principle for PDF-to-image conversion.
"""

from pathlib import Path

from pdf2image import convert_from_path
from PIL import Image

from src.config import settings


def load_pdf_pages(pdf_path: Path) -> list[Image.Image]:
    """Convert all pages of a PDF file into a list of PIL Image objects.

    Args:
        pdf_path: Absolute or relative path to the source PDF file.

    Returns:
        Ordered list of PIL Image objects, one per PDF page.

    Raises:
        FileNotFoundError: If the PDF file does not exist at the given path.
        ValueError: If the PDF contains no pages.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    images = convert_from_path(str(pdf_path), dpi=settings.PDF_DPI)

    if not images:
        raise ValueError(f"No pages found in PDF: {pdf_path}")

    return images


def parse_page_range(page_range: str, total_pages: int) -> list[int]:
    """Parse a human-readable page range string into a list of zero-based page indices.

    Supports comma-separated values and hyphen-separated ranges.
    Examples: '1-5', '1,3,7', '2-4,8'.

    Args:
        page_range: String specifying pages using 1-based numbering.
        total_pages: Total number of pages in the document, used to clamp the upper bound.

    Returns:
        Sorted list of unique zero-based page indices.

    Raises:
        ValueError: If a page number token cannot be parsed as an integer.
    """
    indices: set[int] = set()

    for token in page_range.split(","):
        token = token.strip()
        if "-" in token:
            start, end = token.split("-", maxsplit=1)
            indices.update(range(int(start) - 1, min(int(end), total_pages)))
        else:
            indices.add(int(token) - 1)

    return sorted(indices)