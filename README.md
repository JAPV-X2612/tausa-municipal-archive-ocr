# OCR Documentos Históricos — Municipio de Tausa

Pipeline para transcribir caligrafía manuscrita de los libros del Despacho 
del Alcalde de Tausa, Cundinamarca (1925–1954) usando la API de Claude.

---

## Requisitos previos

- Python 3.9 o superior  
- Poppler instalado (convierte PDF a imágenes)  
- Clave API de Anthropic (plan Pro o API directa)

---

## PASO 1 — Instalar Poppler

**Windows:**
```
1. Descarga: https://github.com/oschwartz10612/poppler-windows/releases
2. Descomprime en C:\poppler
3. Agrega C:\poppler\Library\bin al PATH del sistema
4. Verifica: pdftoppm --version
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

## PASO 2 — Crear entorno virtual e instalar dependencias

```bash
# Crear entorno virtual
python -m venv venv

# Activar (Windows)
venv\Scripts\activate

# Activar (macOS/Linux)
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

---

## PASO 3 — Configurar tu API Key de Anthropic

Obtén tu clave en: https://console.anthropic.com/keys

**Windows (PowerShell):**
```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-api03-TU_CLAVE_AQUI"
```

**Windows (CMD):**
```cmd
set ANTHROPIC_API_KEY=sk-ant-api03-TU_CLAVE_AQUI
```

**macOS / Linux:**
```bash
export ANTHROPIC_API_KEY="sk-ant-api03-TU_CLAVE_AQUI"
```

> 💡 Para que sea permanente en Linux/macOS, agrega la línea `export ANTHROPIC_API_KEY=...`
> al final de tu `~/.bashrc` o `~/.zshrc`.

---

## PASO 4 — Ejecutar la transcripción

### Transcribir el PDF completo:
```bash
python transcribe.py --pdf Despacho_Del_alcalde_1953-1954.pdf
```

### Transcribir solo algunas páginas (útil para prueba inicial):
```bash
# Solo las primeras 3 páginas
python transcribe.py --pdf Despacho_Del_alcalde_1953-1954.pdf --pages 1-3

# Páginas específicas
python transcribe.py --pdf Despacho_Del_alcalde_1953-1954.pdf --pages 1,5,10
```

### Con nombre de salida personalizado:
```bash
python transcribe.py --pdf Despacho_Del_alcalde_1953-1954.pdf --output contratos_1953.txt
```

### Incluyendo salida JSON con metadata:
```bash
python transcribe.py --pdf Despacho_Del_alcalde_1953-1954.pdf --json
```

---

## PASO 5 — Revisar resultados

El script genera automáticamente:

| Archivo | Descripción |
|---------|-------------|
| `Despacho_Del_alcalde_1953-1954_transcripcion.txt` | Texto completo de todos los contratos |
| `Despacho_Del_alcalde_1953-1954_transcripcion.json` | Mismo contenido con metadata por página (solo con `--json`) |

**El script guarda progreso incrementalmente** — si se interrumpe, ya tendrás 
el texto de las páginas procesadas hasta ese punto.

---

## Costo estimado

Cada página de este tipo de documento usa aproximadamente:
- ~800–1200 tokens de entrada (imagen)  
- ~500–800 tokens de salida (transcripción)

| Archivo | Páginas | Costo aprox. |
|---------|---------|--------------|
| Despacho_1953-1954.pdf | 15 | ~$0.15–0.30 USD |
| Colección completa (est. 500 págs.) | 500 | ~$5–10 USD |

---

## Solución de problemas

**Error: `poppler not installed`**  
→ Instala Poppler (ver Paso 1) y asegúrate de que esté en el PATH.

**Error: `ANTHROPIC_API_KEY no configurada`**  
→ Verifica que exportaste la variable en la terminal activa (no en otra pestaña).

**Transcripción con muchos `[ilegible]`**  
→ Sube el DPI en `transcribe.py`: cambia `DPI = 200` a `DPI = 300`.  
   Nota: imágenes más grandes = más tokens = mayor costo.

**Rate limit (demasiadas peticiones)**  
→ El script ya gestiona reintentos automáticos. Si persiste, aumenta  
  `RETRY_DELAY = 5` a `RETRY_DELAY = 10` en `transcribe.py`.

---

## Estructura del proyecto

```
ocr_tausa/
├── transcribe.py        ← Script principal
├── requirements.txt     ← Dependencias Python
├── README.md            ← Esta guía
└── resultados/          ← Carpeta sugerida para tus outputs
```

---

## Próximo paso: hacer el corpus consultable

Una vez tengas los `.txt` de todos los archivos, puedes crear un chatbot 
que responda preguntas sobre los documentos usando RAG (retrieval-augmented generation).

Herramientas recomendadas:
- **LlamaIndex** — indexación y consulta de documentos locales
- **ChromaDB** — base vectorial gratuita y local  
- **Streamlit** — interfaz web simple para el chatbot

Pídele a Claude que te arme ese pipeline cuando tengas los textos listos.
