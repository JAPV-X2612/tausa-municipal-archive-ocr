"""
Rebuild a structured JSON transcription file from an existing TXT transcription.

Use this when a JSON file has been accidentally deleted or corrupted but the
corresponding TXT file (written incrementally by the pipeline) is intact.

Token counts and processing times cannot be recovered from the TXT file and
will be set to 0 in the rebuilt JSON.

Usage:
    python scripts/rebuild_json_from_txt.py <txt_path>

Example:
    python scripts/rebuild_json_from_txt.py \
        outputs/administracion_general_salinas_1931_1942_transcription.txt
"""

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

_SEPARATOR = "─" * 70
_PAGE_PATTERN = re.compile(r"^PAGE (\d+)$")


def parse_txt(txt_path: Path) -> dict:
    """Parse a pipeline TXT file and return a dict matching PipelineResult's schema.

    Args:
        txt_path: Path to the plain-text transcription file.

    Returns:
        Dictionary matching the JSON schema produced by TranscriptionRepository.save_json().
    """
    lines = txt_path.read_text(encoding="utf-8").splitlines()

    # Extract header metadata from the first four lines:
    #   Line 0: "OCR TRANSCRIPTION — {document_title}"
    #   Line 1: "Municipality of Tausa, Cundinamarca, Colombia"
    #   Line 2: "Model: {model} | DPI: {dpi}"
    #   Line 3: separator
    document_title = lines[0].removeprefix("OCR TRANSCRIPTION — ").strip()
    model_line = lines[2]  # "Model: claude-sonnet-4-6 | DPI: 200"
    model = model_line.split("|")[0].removeprefix("Model:").strip()

    pages: list[dict] = []
    i = 0
    while i < len(lines):
        if lines[i] == _SEPARATOR and i + 1 < len(lines):
            match = _PAGE_PATTERN.match(lines[i + 1])
            if match:
                page_number = int(match.group(1))
                # Skip: separator, "PAGE N", separator, blank line
                content_start = i + 4
                # Collect lines until the next separator block or end of file
                content_lines: list[str] = []
                j = content_start
                while j < len(lines):
                    if lines[j] == _SEPARATOR:
                        break
                    content_lines.append(lines[j])
                    j += 1
                # Strip leading/trailing blank lines from the page block
                transcription = "\n".join(content_lines).strip()
                pages.append(
                    {
                        "page_number": page_number,
                        "transcription": transcription,
                        "processing_time_seconds": 0.0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                    }
                )
                i = j
                continue
        i += 1

    # Derive source_file and total_pages from what we know
    pdf_stem = txt_path.stem.removesuffix("_transcription")
    source_file = str(Path("assets") / "docs" / f"{pdf_stem}.pdf")
    total_pages = max((p["page_number"] for p in pages), default=0)

    return {
        "source_file": source_file,
        "document_title": document_title,
        "model": model,
        "total_pages": total_pages,
        "processed_pages": pages,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python scripts/rebuild_json_from_txt.py <txt_path>")
        sys.exit(1)

    txt_path = Path(sys.argv[1])
    if not txt_path.exists():
        print(f"Error: file not found — {txt_path}")
        sys.exit(1)

    print(f"Parsing {txt_path.name}...")
    payload = parse_txt(txt_path)
    page_count = len(payload["processed_pages"])
    print(f"  Found {page_count} page(s).")

    json_path = txt_path.with_name(txt_path.stem + ".json")
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  Written → {json_path}")
    print("Done. Note: token counts and processing times are set to 0 (not recoverable from TXT).")


if __name__ == "__main__":
    main()
