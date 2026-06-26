"""
Prueba 2 y 3: Análisis cualitativo con APIs de visión (Claude y GPT-4o).
Manda la photo + una pregunta de experto en aparejos y devuelve un dictamen.

Require variables de entorno:
    export ANTHROPIC_API_KEY=sk-ant-...
    export OPENAI_API_KEY=sk-...

Uso:
    python vision_api.py entradas/mi_foto.jpg claude
    python vision_api.py entradas/mi_foto.jpg gpt
    python vision_api.py entradas/mi_foto.jpg ambos
"""

import base64
import mimetypes
import os
import sys
from pathlib import Path

PROMPT = """Eres un inspector certificado de aparejos de izaje (rigging).
Analiza la eslinga, grillete u otro elemento de izaje en la imagen.

Responde SOLO en este formato:
- ELEMENTO: (eslinga textil / grillete / cadena / gancho / otro)
- ESTADO: (OK / DESGASTE LEVE / CONDENAR / NO SE PUEDE DETERMINAR)
- DEFECTOS VISIBLE: lista breve (cortes, deshilachado, deformación, corrosión, etc.)
- CONFIANZA: (alta / media / baja)
- RECOMENDACIÓN: una frase

Si la imagen no permite un juicio fiable, dilo claramente.
IMPORTANTE: este es un análisis preliminar, NO sustituye una inspección física certificada."""


def encode(img_path: Path):
    mime = mimetypes.guess_type(img_path)[0] or "image/jpeg"
    data = base64.standard_b64encode(img_path.read_bytes()).decode()
    return mime, data


def run_claude(img_path: Path):
    from anthropic import Anthropic

    mime, data = encode(img_path)
    client = Anthropic()
    msg = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=600,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": mime, "data": data}},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ],
    )
    return msg.content[0].text


def run_gpt(img_path: Path):
    from openai import OpenAI

    mime, data = encode(img_path)
    client = OpenAI()
    resp = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=600,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}},
                ],
            }
        ],
    )
    return resp.choices[0].message.content


def main(img_path: str, backend: str):
    img_path = Path(img_path)
    if not img_path.exists():
        sys.exit(f"No existe la imagen: {img_path}")

    backends = ["claude", "gpt"] if backend == "ambos" else [backend]
    for b in backends:
        print(f"\n=== {b.upper()} :: {img_path.name} ===")
        try:
            if b == "claude":
                if not os.getenv("ANTHROPIC_API_KEY"):
                    print("Falta ANTHROPIC_API_KEY")
                    continue
                print(run_claude(img_path))
            elif b == "gpt":
                if not os.getenv("OPENAI_API_KEY"):
                    print("Falta OPENAI_API_KEY")
                    continue
                print(run_gpt(img_path))
            else:
                print(f"Backend desconocido: {b}")
        except Exception as e:
            print(f"Error con {b}: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("Uso: python vision_api.py <ruta_imagen> <claude|gpt|ambos>")
    main(sys.argv[1], sys.argv[2])
