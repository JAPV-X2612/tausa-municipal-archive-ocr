"""
Text normalization utilities for OCR transcription post-processing.

Applies lightweight cleaning to raw Claude transcriptions before storage,
removing visual artefacts introduced by handwritten line-wrapping while
preserving the semantic paragraph structure required for accurate chunking.
"""

import re


def normalize_transcription(text: str) -> str:
    """Remove handwriting artefacts from a raw OCR transcription.

    Applies two transformations in sequence:

    1. Joins hyphenated line-breaks that split a single word across lines
       (e.g. ``"Munici-\\ncipal"`` → ``"Municipal"``).  These are visual
       layout artefacts of the original manuscript, not intentional word
       boundaries.

    2. Collapses three or more consecutive newlines to a single blank line,
       standardising paragraph separators without losing document structure.

    The function intentionally preserves:
    - Double newlines (``\\n\\n``) — paragraph and clause boundaries.
    - Single newlines (``\\n``) — line breaks within a section.

    Args:
        text: Raw transcription string returned by the Claude Vision API.

    Returns:
        Cleaned transcription with word-split artefacts removed and
        paragraph whitespace normalised.
    """
    # Join words split across lines with a trailing hyphen (layout artefact).
    # Pattern: hyphen immediately followed by newline — no spaces in between.
    text = re.sub(r"-\n", "", text)

    # Cap runs of three or more newlines to a single paragraph break.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()
