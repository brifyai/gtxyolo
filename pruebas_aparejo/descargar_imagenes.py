"""
Descarga las imágenes de los productos del informe, cada una etiquetada
con su producto, estado (dictamen real) y descripción de falla.

Salida:
  dataset/<estado>/<id>_<n>.jpg        imágenes organizadas por dictamen
  dataset/etiquetas.csv                tabla con metadatos de cada imagen
"""
import csv
import re
from pathlib import Path

import requests

from kull_api import KullClient

OUT = Path("dataset")


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (s or "sin_estado").lower()).strip("_")


def main(paginas=1, limit=10):
    c = KullClient()
    OUT.mkdir(exist_ok=True)
    filas = []

    for page in range(1, paginas + 1):
        items = c.productos(limit=limit, page=page)["data"]["items"]
        for p in items:
            estado = p.get("statusLabel") or "sin_estado"
            defecto = " ".join((p.get("comentario") or "").split())
            destino = OUT / slug(estado)
            destino.mkdir(parents=True, exist_ok=True)

            for n, img in enumerate(p.get("imagenes", []), 1):
                url = img["url"]
                nombre = f"{p['id']}_{n}.jpg"
                ruta = destino / nombre
                try:
                    r = requests.get(url, timeout=60)
                    r.raise_for_status()
                    ruta.write_bytes(r.content)
                    ok = True
                except Exception as e:
                    print(f"  ! error {url}: {e}")
                    ok = False
                if ok:
                    filas.append({
                        "archivo": str(ruta),
                        "producto_id": p["id"],
                        "nombre": p.get("nombre", ""),
                        "tipo": (p.get("productoCatalogo") or {}).get("descripcion", ""),
                        "estado": estado,
                        "defecto": defecto,
                        "faena": (p.get("informe") or {}).get("faenaNombre", ""),
                        "url": url,
                    })
                    print(f"  ✓ {ruta}  [{estado}] {defecto[:40]}")

    with open(OUT / "etiquetas.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)

    print(f"\nTotal imágenes descargadas: {len(filas)}")
    print(f"Etiquetas en: {OUT/'etiquetas.csv'}")


if __name__ == "__main__":
    import sys
    paginas = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    main(paginas=paginas)
