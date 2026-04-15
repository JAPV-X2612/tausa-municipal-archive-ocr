"""
Transcriber module for the Tausa Municipal Archive OCR pipeline.

Encapsulates all communication with the Anthropic Messages API, including
retry logic for rate limiting and transient API errors.
"""

import time

import anthropic
from PIL import Image

from src.config import settings
from src.models.models import TranscriptionResult
from src.ocr.image_processor import prepare_image_for_api
from src.prompts.templates import (
    TRANSCRIPTION_PAGE_PROMPT_TEMPLATE,
    TRANSCRIPTION_SYSTEM_PROMPT,
)


class PageTranscriber:
    """Sends individual document pages to the Anthropic API and returns transcriptions.

    Implements exponential-style retry logic for rate limit and transient API
    errors, isolating all API concerns from the orchestration layer.
    """

    def __init__(self, client: anthropic.Anthropic) -> None:
        """Initialize the transcriber with an authenticated Anthropic client.

        Args:
            client: Authenticated Anthropic API client instance.
        """
        self._client = client

    def transcribe(
        self,
        image: Image.Image,
        page_number: int,
        total_pages: int,
        document_title: str,
    ) -> TranscriptionResult:
        """Transcribe a single document page image using the Claude Vision API.

        Args:
            image: PIL Image of the document page to transcribe.
            page_number: 1-based page number within the document.
            total_pages: Total number of pages in the document.
            document_title: Human-readable document title for prompt context.

        Returns:
            TranscriptionResult containing the transcription text and the
            number of input/output tokens consumed by the API call.
            On failure, text contains an error marker and token counts are 0.
        """
        base64_data, media_type = prepare_image_for_api(image)
        page_prompt = TRANSCRIPTION_PAGE_PROMPT_TEMPLATE.format(
            page_number=page_number,
            total_pages=total_pages,
            document_title=document_title,
        )

        for attempt in range(1, settings.RETRY_ATTEMPTS + 1):
            try:
                response = self._client.messages.create(
                    model=settings.CLAUDE_MODEL,
                    max_tokens=settings.MAX_OUTPUT_TOKENS,
                    system=TRANSCRIPTION_SYSTEM_PROMPT,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": base64_data,
                                    },
                                },
                                {"type": "text", "text": page_prompt},
                            ],
                        }
                    ],
                )
                return TranscriptionResult(
                    text=response.content[0].text,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                )

            except anthropic.RateLimitError:
                wait_seconds = settings.RETRY_DELAY_SECONDS * attempt
                print(f"  ⚠  Rate limit hit. Retrying in {wait_seconds}s (attempt {attempt})...")
                time.sleep(wait_seconds)

            except anthropic.APIError as error:
                if attempt == settings.RETRY_ATTEMPTS:
                    return TranscriptionResult(
                        text=f"[TRANSCRIPTION ERROR — page {page_number}: {error}]",
                        input_tokens=0,
                        output_tokens=0,
                    )
                time.sleep(settings.RETRY_DELAY_SECONDS)

        return TranscriptionResult(
            text=f"[TRANSCRIPTION FAILED — page {page_number}: all {settings.RETRY_ATTEMPTS} attempts exhausted]",
            input_tokens=0,
            output_tokens=0,
        )
