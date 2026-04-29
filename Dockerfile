# =============================================================================
# Tausa Municipal Archive — API Server
# =============================================================================
# Multi-concern notes:
#   - CPU-only PyTorch is installed explicitly to keep the image ~2 GB smaller
#     than the default CUDA build. sentence-transformers reuses the existing
#     torch installation and does not pull CUDA packages.
#   - The sentence-transformers embedding model is downloaded at build time
#     (baked into the image) to eliminate cold-start latency on first request.
#   - Railway does not support the VOLUME instruction. Persistent storage for
#     /app/chroma_db must be configured as a Railway Volume in the dashboard
#     (Service → Volumes → Mount path: /app/chroma_db).
# =============================================================================

FROM python:3.12-slim

# ── System dependencies ───────────────────────────────────────────────────────
# poppler-utils is required by pdf2image for PDF-to-image conversion.
RUN apt-get update \
    && apt-get install -y --no-install-recommends poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Install CPU-only PyTorch before the rest of requirements.txt.
# pip will see torch as already satisfied when sentence-transformers is
# installed, preventing it from pulling the 2.5 GB CUDA wheel.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Pre-download embedding model ──────────────────────────────────────────────
# Baking the model into the image (~420 MB) avoids a download on first
# startup. HF_HOME is set here and re-exported at runtime so the container
# finds the cached files in the same location.
ENV HF_HOME=/app/.cache/huggingface
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')"

# ── Application code ──────────────────────────────────────────────────────────
COPY src/ ./src/
COPY outputs/ ./outputs/
COPY serve.py ingest.py transcribe.py .env.example entrypoint.sh ./

# ── Security: non-root user ───────────────────────────────────────────────────
RUN useradd -m -u 1000 appuser \
    && chown -R appuser:appuser /app \
    && chmod +x /app/entrypoint.sh
USER appuser

# ── Runtime ───────────────────────────────────────────────────────────────────
# NOTE: Railway does not support the VOLUME instruction. Configure persistent
# storage for /app/chroma_db via the Railway dashboard:
# Service → Volumes → Add Volume → Mount path: /app/chroma_db

EXPOSE 8000

# Allow 120 s for startup: model load + initial ingest on first deploy.
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

ENTRYPOINT ["/app/entrypoint.sh"]
