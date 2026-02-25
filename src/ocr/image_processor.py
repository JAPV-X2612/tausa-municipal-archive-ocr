"""
Image processor module for the Tausa Municipal Archive OCR pipeline.

Handles image enhancement for handwritten document legibility and
conversion to base64-encoded PNG for the Anthropic Vision API.
"""

import io
import base64

from PIL import Image, ImageEnhance

from src.config import settings


def prepare_image_for_api(image: Image.Image) -> tuple[str, str]:
    """Enhance a PIL image and encode it as base64 PNG for the Anthropic API.

    Applies contrast and sharpness enhancement optimized for faded historical
    handwritten documents, then encodes the result as a base64 PNG string.
    Color information is preserved to retain ink color and stamp distinctions.

    Args:
        image: Source PIL Image object from PDF conversion.

    Returns:
        Tuple of (base64_encoded_string, media_type) where media_type is
        always 'image/png'.
    """
    resized = _resize_if_needed(image)
    enhanced = _enhance_for_ocr(resized)
    encoded = _encode_to_base64_png(enhanced)
    return encoded, "image/png"


def _resize_if_needed(image: Image.Image) -> Image.Image:
    """Downscale the image if its width exceeds the configured maximum.

    Args:
        image: Source PIL Image.

    Returns:
        Resized PIL Image if width exceeded the limit, otherwise the original.
    """
    if image.width <= settings.IMAGE_MAX_WIDTH:
        return image

    scale = settings.IMAGE_MAX_WIDTH / image.width
    new_height = int(image.height * scale)
    return image.resize((settings.IMAGE_MAX_WIDTH, new_height), Image.Resampling.LANCZOS)


def _enhance_for_ocr(image: Image.Image) -> Image.Image:
    """Apply contrast and sharpness enhancement to improve OCR accuracy.

    Enhancement factors are configurable via environment variables to allow
    tuning without code changes across different document batches.

    Args:
        image: Source PIL Image.

    Returns:
        Enhanced PIL Image with improved contrast and sharpness.
    """
    image = image.convert("RGB")
    image = ImageEnhance.Contrast(image).enhance(settings.CONTRAST_FACTOR)
    image = ImageEnhance.Sharpness(image).enhance(settings.SHARPNESS_FACTOR)
    return image


def _encode_to_base64_png(image: Image.Image) -> str:
    """Serialize a PIL Image to a base64-encoded PNG string.

    PNG is used instead of JPEG to avoid lossy compression artifacts on
    fine ink strokes in historical handwritten documents.

    Args:
        image: Enhanced PIL Image to encode.

    Returns:
        Base64-encoded string representation of the PNG image.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return base64.standard_b64encode(buffer.getvalue()).decode("utf-8")