#!/bin/sh
# =============================================================================
# Container entrypoint — auto-ingest then start the API server.
#
# On the very first deployment the ChromaDB volume is empty. This script
# detects that state and runs ingest.py before starting the server, so the
# app is immediately queryable after startup. On all subsequent restarts the
# collection already exists and ingest is skipped.
# =============================================================================
set -e

echo "==> Checking ChromaDB collection..."

if python - <<'PYEOF'
import sys
import chromadb
from src.config import settings

try:
    client = chromadb.PersistentClient(path=str(settings.CHROMA_DB_DIR))
    col = client.get_collection("tausa_archive")
    count = col.count()
    print(f"    Collection found: {count:,} chunks — skipping ingest.")
    sys.exit(0 if count > 0 else 1)
except Exception:
    sys.exit(1)
PYEOF
then
    echo "==> ChromaDB already populated."
else
    echo "==> Collection empty or missing — running initial ingest..."
    python ingest.py
    echo "==> Ingest complete."
fi

echo "==> Starting API server..."
exec python serve.py
