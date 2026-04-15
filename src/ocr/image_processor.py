"""
Image processor module for the Tausa Municipal Archive OCR pipeline.

Handles image enhancement for handwritten document legibility and
conversion to base64-encoded JPEG for the Anthropic Vision API.
"""

import io
import base64

from PIL import Image, ImageEnhance

from src.config import settings


def prepare_image_for_api(image: Image.Image) -> tuple[str, str]:
    """Enhance a PIL image and encode it as base64 JPEG for the Anthropic API.

    Applies contrast and sharpness enhancement optimised for faded historical
    handwritten documents, then encodes the result as a base64 JPEG string.
    Color information is preserved to retain ink color and stamp distinctions.

    JPEG is used instead of PNG because the Anthropic Vision API enforces a
    5 MB per-image limit. High-resolution scans of dense manuscript pages
    routinely exceed that limit as lossless PNG but stay well within it as
    JPEG at quality 88. At that quality level the compression artefacts are
    imperceptible to a vision model performing OCR on already-aged documents.

    Args:
        image: Source PIL Image object from PDF conversion.

    Returns:
        Tuple of (base64_encoded_string, media_type) where media_type is
        always ``'image/jpeg'``.
    """
    resized = _resize_if_needed(image)
    enhanced = _enhance_for_ocr(resized)
    encoded = _encode_to_base64_jpeg(enhanced)
    return encoded, "image/jpeg"


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


def _encode_to_base64_jpeg(image: Image.Image) -> str:
    """Serialise a PIL Image to a base64-encoded JPEG string.

    Args:
        image: Enhanced PIL Image to encode.

    Returns:
        Base64-encoded string representation of the JPEG image.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=settings.JPEG_QUALITY, optimize=True)
    return base64.standard_b64encode(buffer.getvalue()).decode("utf-8")
