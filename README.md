# Tausa Municipal Archive — Historical OCR Pipeline

An AI-powered OCR pipeline that transcribes handwritten manuscripts from the municipal archive
of Tausa, Cundinamarca, Colombia (1925–1954) using the Claude Vision API. Transcribed documents
are stored in a vector database to enable semantic Q&A over the historical corpus.

---

## Prerequisites

- Python 3.9 or higher
- Poppler (converts PDF pages to images)
- Anthropic API key

---

## Step 1 — Install Poppler

**Windows:**
```
1. Download: https://github.com/oschwartz10612/poppler-windows/releases
2. Extract to C:\poppler
3. Add C:\poppler\Library\bin to your system PATH
4. Verify: pdftoppm --version
```

**macOS:**
```bash
brew install poppler
```

**Ubuntu / Debian:**
```bash
sudo apt-get install poppler-utils
```

---

## Step 2 — Create virtual environment and install dependencies

```bash
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (macOS / Linux)
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Step 3 — Configure environment variables

```bash
cp .env.example .env
# Edit .env and set your ANTHROPIC_API_KEY
```

Get your key at: https://console.anthropic.com/keys

---

## Step 4 — Run the transcription pipeline

### Transcribe a full PDF:
```bash
python transcribe.py --pdf assets/docs/tausa_alcaldia_despacho_alcalde_1953_1954.pdf \
                     --title "Tausa - Despacho del Alcalde 1953-1954" \
                     --json
```

### Transcribe specific pages (useful for initial validation):
```bash
# First 3 pages only
python transcribe.py --pdf assets/docs/tausa_alcaldia_despacho_alcalde_1953_1954.pdf --pages 1-3

# Non-consecutive pages
python transcribe.py --pdf assets/docs/tausa_alcaldia_despacho_alcalde_1953_1954.pdf --pages 1,5,10
```

### CLI reference:

| Flag | Required | Description |
|------|----------|-------------|
| `--pdf` | Yes | Path to the source PDF file |
| `--title` | No | Human-readable document title (defaults to filename) |
| `--pages` | No | Page range, e.g. `1-5` or `1,3,7` (defaults to all pages) |
| `--json` | No | Also save results as structured JSON with page-level metadata |

---

## Step 5 — Review output

Each pipeline run produces two files in `outputs/`:

| File | Description |
|------|-------------|
| `<pdf-name>_transcription.txt` | Plain-text transcription, saved incrementally after each page |
| `<pdf-name>_transcription.json` | Structured JSON with full page metadata (only with `--json`) |

The pipeline saves progress after every page — if interrupted, already-processed pages are preserved.

---

## Estimated API cost

Each page consumes approximately:
- ~800–1 200 input tokens (image)
- ~500–800 output tokens (transcription)

| Document | Pages | Approx. cost |
|----------|-------|--------------|
| Despacho del Alcalde 1953–1954 | 15 | ~$0.15–0.30 USD |
| Full corpus (~250 pages) | 250 | ~$2.50–5.00 USD |

---

## Troubleshooting

**`Unable to get page count. Is poppler installed and in PATH?`**
Install Poppler (see Step 1) and ensure it is on your system PATH.

**`ANTHROPIC_API_KEY is not set`**
Verify the variable is exported in your active terminal session and that `.env` exists.

**Many `[illegible]` markers in output**
Increase `PDF_DPI` in `.env` from `200` to `300`. Note: higher DPI means larger images,
more tokens, and higher cost per page.

**Rate limit errors**
The pipeline handles retries automatically. If errors persist, increase
`RETRY_DELAY_SECONDS` in `.env`.

---

## Project structure

```
tausa-municipal-archive-ocr/
├── transcribe.py              # CLI entry point for the OCR pipeline
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variable reference
├── assets/
│   └── docs/                  # Source PDF documents
├── outputs/                   # Transcription results (TXT + JSON)
└── src/
    ├── config/settings.py     # Centralised configuration
    ├── models/models.py       # Shared data models
    ├── ocr/
    │   ├── pipeline.py        # Orchestrator (facade pattern)
    │   ├── pdf_processor.py   # PDF → PIL image conversion
    │   ├── image_processor.py # Image enhancement + base64 encoding
    │   └── transcriber.py     # Claude Vision API client
    ├── prompts/templates.py   # System and page-level prompt templates
    ├── rag/retriever.py       # Semantic retrieval over ChromaDB (Step 2)
    └── storage/repository.py  # TXT and JSON persistence (repository pattern)
```
