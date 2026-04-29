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

ROOT_DIR: Path = Path(__file__).resolve().parents[2]
ASSETS_DIR: Path = ROOT_DIR / "assets"
DOCS_DIR: Path = ASSETS_DIR / "docs"
OUTPUTS_DIR: Path = ROOT_DIR / "outputs"

# =============================================================================
# ANTHROPIC API
# =============================================================================

ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
MAX_OUTPUT_TOKENS: int = int(os.getenv("MAX_OUTPUT_TOKENS", "4096"))

# =============================================================================
# IMAGE PROCESSING
# =============================================================================

PDF_DPI: int = int(os.getenv("PDF_DPI", "200"))
IMAGE_FORMAT: str = "PNG"
IMAGE_MAX_WIDTH: int = int(os.getenv("IMAGE_MAX_WIDTH", "1600"))
CONTRAST_FACTOR: float = float(os.getenv("CONTRAST_FACTOR", "1.3"))
SHARPNESS_FACTOR: float = float(os.getenv("SHARPNESS_FACTOR", "1.2"))
JPEG_QUALITY: int = int(os.getenv("JPEG_QUALITY", "88"))

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

# =============================================================================
# API SERVER
# =============================================================================

# Comma-separated list of allowed CORS origins.
# In production, set this to the exact Vercel frontend URL.
ALLOWED_ORIGINS: list[str] = os.getenv(
    "ALLOWED_ORIGINS", "http://localhost:3000"
).split(",")

# Default number of archive chunks shown to the user as citations.
RAG_N_RESULTS: int = int(os.getenv("RAG_N_RESULTS", "10"))

# Number of candidates fetched from ChromaDB before relevance filtering.
# Must be >= RAG_N_RESULTS. Extra headroom ensures enough results survive
# the distance filter.
RAG_FETCH_CANDIDATES: int = int(os.getenv("RAG_FETCH_CANDIDATES", "20"))

# Maximum cosine distance for a chunk to be considered relevant.
# ChromaDB cosine distance: 0.0 = identical, 1.0 = unrelated.
# Chunks above this threshold are excluded from the context sent to Claude.
# Tune down (e.g. 0.45) for stricter relevance, up (e.g. 0.70) if too few
# results are returned for broad queries.
RAG_MAX_DISTANCE: float = float(os.getenv("RAG_MAX_DISTANCE", "0.55"))

# Enable Anthropic's built-in web search tool so Claude can complement
# archive answers with public historical context when needed.
# Set to "false" to disable (e.g. to reduce costs or latency).
WEB_SEARCH_ENABLED: bool = os.getenv("WEB_SEARCH_ENABLED", "true").lower() == "true"