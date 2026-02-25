"""
Centralized configuration and settings for the Tausa Municipal Archive OCR pipeline.

Loads environment variables and defines application-wide constants following
the twelve-factor app methodology for configuration management.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# BASE PATHS
# =============================================================================

ROOT_DIR: Path = Path(__file__).resolve().parents[4]
ASSETS_DIR: Path = ROOT_DIR / "assets"
DOCS_DIR: Path = ASSETS_DIR / "docs"
OUTPUTS_DIR: Path = ROOT_DIR / "outputs"

# =============================================================================
# ANTHROPIC API
# =============================================================================

ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5")
MAX_OUTPUT_TOKENS: int = int(os.getenv("MAX_OUTPUT_TOKENS", "4096"))

# =============================================================================
# IMAGE PROCESSING
# =============================================================================

PDF_DPI: int = int(os.getenv("PDF_DPI", "200"))
IMAGE_FORMAT: str = "PNG"
IMAGE_MAX_WIDTH: int = int(os.getenv("IMAGE_MAX_WIDTH", "1600"))
CONTRAST_FACTOR: float = float(os.getenv("CONTRAST_FACTOR", "1.3"))
SHARPNESS_FACTOR: float = float(os.getenv("SHARPNESS_FACTOR", "1.2"))

# =============================================================================
# PIPELINE BEHAVIOR
# =============================================================================

RETRY_ATTEMPTS: int = int(os.getenv("RETRY_ATTEMPTS", "3"))
RETRY_DELAY_SECONDS: int = int(os.getenv("RETRY_DELAY_SECONDS", "5"))
INTER_PAGE_DELAY_SECONDS: float = float(os.getenv("INTER_PAGE_DELAY_SECONDS", "1.0"))

# =============================================================================
# STORAGE
# =============================================================================

CHROMA_DB_DIR: Path = ROOT_DIR / "chroma_db"
SQLITE_DB_PATH: Path = OUTPUTS_DIR / "archive_index.db"