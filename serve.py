"""
CLI entry point for running the Tausa Archive FastAPI server.

Usage:
    python serve.py              # production-like (no reload)
    python serve.py --reload     # development mode with auto-reload

The server listens on 0.0.0.0:8000 by default. Override with:
    PORT=9000 python serve.py
"""

import argparse
import os

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Tausa Archive API server.")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on code changes (development only).",
    )
    args = parser.parse_args()

    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
