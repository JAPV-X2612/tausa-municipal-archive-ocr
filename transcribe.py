"""
transcribe.py
─────────────────────────────────────────────────────────────
Pipeline OCR para documentos manuscritos históricos del
municipio de Tausa, Cundinamarca (1925–1954).

Uso:
    python transcribe.py --pdf archivo.pdf
    python transcribe.py --pdf archivo.pdf --pages 1-5
    python transcribe.py --pdf archivo.pdf --output resultado.txt
"""

import io
import os
import json
import sys
import time
import base64
import argparse
import anthropic

from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image

# ─────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────

CLAUDE_MODEL   = "claude-opus-4-6"    # Mejor capacidad de visión
DPI            = 200                  # Resolución de conversión PDF → imagen
MAX_IMG_WIDTH  = 1600                 # Máx ancho en px (evita tokens excesivos)
RETRY_ATTEMPTS = 3                    # Reintentos ante error de API
RETRY_DELAY    = 5                    # Segundos entre reintentos

SYSTEM_PROMPT = """Eres un experto paleógrafo y archivista especializado en documentos 
históricos colombianos del siglo XX. Tu tarea es transcribir con máxima fidelidad 
documentos manuscritos en español de la alcaldía del municipio de Tausa, Cundinamarca, 
escritos entre 1925 y 1954.

Reglas estrictas:
1. Transcribe TODO el texto visible, respetando la ortografía original (incluso errores)
2. Mantén la estructura del documento: títulos, numerales, párrafos, firmas
3. Si una palabra es ilegible, escribe [ilegible] en su lugar
4. Si una sección es parcialmente legible, transcribe lo que puedas y marca el resto [ilegible]
5. NO corrijas ni modernices el texto
6. NO agregues interpretaciones ni comentarios propios dentro de la transcripción
7. Al final de cada página, agrega una línea separadora: ─────────────────
8. Responde SOLO con la transcripción, sin preámbulos"""

PAGE_PROMPT = """Transcribe con máxima fidelidad el texto manuscrito de esta imagen.
Es una página de un libro de contratos verbales del Despacho del Alcalde del municipio 
de Tausa, Cundinamarca, Colombia, circa 1953–1954.

Incluye: número de contrato, partes contratantes, cédulas, cláusulas, valor, 
plazo, fecha y firmas/testigos."""


# ─────────────────────────────────────────────────────────────
# FUNCIONES AUXILIARES
# ─────────────────────────────────────────────────────────────

def image_to_base64(img: Image.Image) -> tuple[str, str]:
    """Convierte imagen PIL a base64 JPEG."""
    # Redimensionar si es muy ancha
    if img.width > MAX_IMG_WIDTH:
        ratio  = MAX_IMG_WIDTH / img.width
        height = int(img.height * ratio)
        img    = img.resize((MAX_IMG_WIDTH, height), Image.Resampling.LANCZOS)

    # Mejorar contraste para manuscritos
    img = enhance_for_ocr(img)

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=92)
    b64 = base64.standard_b64encode(buffer.getvalue()).decode("utf-8")
    return b64, "image/jpeg"


def enhance_for_ocr(img: Image.Image) -> Image.Image:
    """Mejora leve de contraste y nitidez para caligrafía antigua."""
    from PIL import ImageEnhance
    img = img.convert("RGB")
    img = ImageEnhance.Contrast(img).enhance(1.3)
    img = ImageEnhance.Sharpness(img).enhance(1.2)
    return img


def parse_page_range(page_range: str, total_pages: int) -> list[int]:
    """Parsea '1-5' o '1,3,5' o '3' a lista de índices 0-based."""
    pages = []
    for part in page_range.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-")
            pages.extend(range(int(start) - 1, min(int(end), total_pages)))
        else:
            pages.append(int(part) - 1)
    return sorted(set(pages))


def transcribe_page(client: anthropic.Anthropic, img: Image.Image, page_num: int) -> str:
    """Envía una página a Claude y retorna la transcripción."""
    b64, media_type = image_to_base64(img)

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            response = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": b64,
                                },
                            },
                            {
                                "type": "text",
                                "text": f"[Página {page_num}]\n\n{PAGE_PROMPT}",
                            },
                        ],
                    }
                ],
            )
            return response.content[0].text

        except anthropic.RateLimitError:
            wait = RETRY_DELAY * attempt
            print(f"  ⚠️  Rate limit. Esperando {wait}s...", flush=True)
            time.sleep(wait)

        except anthropic.APIError as e:
            if attempt == RETRY_ATTEMPTS:
                return f"[ERROR en página {page_num}: {e}]"
            time.sleep(RETRY_DELAY)

    return f"[FALLO después de {RETRY_ATTEMPTS} intentos en página {page_num}]"


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OCR manuscrito histórico con Claude")
    parser.add_argument("--pdf",    required=True, help="Ruta al archivo PDF")
    parser.add_argument("--pages",  default=None,  help="Páginas a procesar, ej: '1-5' o '1,3,7'")
    parser.add_argument("--output", default=None,  help="Archivo de salida (.txt). Por defecto: <pdf>_transcription.txt")
    parser.add_argument("--json",   action="store_true", help="También guardar salida en JSON con metadata")
    args = parser.parse_args()

    # ── Validaciones ──────────────────────────────────────────
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"❌ Archivo no encontrado: {pdf_path}")
        sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ Variable ANTHROPIC_API_KEY no configurada.")
        print("   Exporta tu clave: export ANTHROPIC_API_KEY='sk-ant-...'")
        sys.exit(1)

    output_path = Path(args.output) if args.output else pdf_path.with_suffix("_transcription.txt")

    # ── Convertir PDF a imágenes ───────────────────────────────
    print(f"📄 Cargando PDF: {pdf_path.name}")
    images = convert_from_path(str(pdf_path), dpi=DPI)
    total  = len(images)
    print(f"   → {total} páginas detectadas")

    page_indices = parse_page_range(args.pages, total) if args.pages else list(range(total))
    print(f"   → Procesando páginas: {[i+1 for i in page_indices]}\n")

    # ── Cliente Anthropic ──────────────────────────────────────
    client = anthropic.Anthropic(api_key=api_key)

    # ── Transcripción página a página ─────────────────────────
    results   = []
    full_text = []

    header = (
        f"TRANSCRIPCIÓN OCR - {pdf_path.name}\n"
        f"Municipio de Tausa, Cundinamarca | Libro de Contratos Verbales 1953-1954\n"
        f"Modelo: {CLAUDE_MODEL} | DPI: {DPI}\n"
        f"{'═' * 70}\n\n"
    )
    full_text.append(header)

    for idx in page_indices:
        page_num = idx + 1
        print(f"🔍 Procesando página {page_num}/{total}...", end=" ", flush=True)

        start    = time.time()
        text     = transcribe_page(client, images[idx], page_num)
        elapsed  = time.time() - start

        print(f"✅ ({elapsed:.1f}s)")

        page_header = f"\n{'─' * 70}\nPÁGINA {page_num}\n{'─' * 70}\n\n"
        full_text.append(page_header + text + "\n")

        results.append({
            "page":           page_num,
            "transcription":  text,
            "processing_time": round(elapsed, 2),
        })

        # Guardar progreso incremental (por si falla a mitad)
        output_path.write_text("\n".join(full_text), encoding="utf-8")

        # Pausa breve para no saturar la API
        if idx != page_indices[-1]:
            time.sleep(1)

    # ── Guardar resultados ─────────────────────────────────────
    output_path.write_text("\n".join(full_text), encoding="utf-8")
    print(f"\n✅ Transcripción guardada en: {output_path}")

    if args.json:
        json_path = output_path.with_suffix(".json")
        json_data = {
            "source":     str(pdf_path),
            "model":      CLAUDE_MODEL,
            "dpi":        DPI,
            "total_pages": total,
            "pages":      results,
        }
        json_path.write_text(json.dumps(json_data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"✅ JSON guardado en:          {json_path}")

    print(f"\n🎉 Listo. {len(page_indices)} página(s) procesadas.")


if __name__ == "__main__":
    main()
