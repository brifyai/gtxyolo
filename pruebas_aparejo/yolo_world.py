"""
Prueba 1: YOLO-World (detección open-vocabulary, SIN entrenar).
Detecta objetos a partir de prompts de texto. Corre 100% local.

Uso:
    python yolo_world.py entradas/mi_foto.jpg
"""

import sys
from pathlib import Path

from ultralytics import YOLOWorld

# Clases que queremos detectar, descritas en texto libre (inglés funciona mejor).
CLASES = [
    "lifting sling",
    "frayed sling",
    "cut webbing strap",
    "shackle",
    "bent shackle",
    "corroded metal shackle",
    "hook",
    "chain",
]


def main(img_path: str):
    img_path = Path(img_path)
    if not img_path.exists():
        sys.exit(f"No existe la imagen: {img_path}")

    # yolov8x-worldv2 se descarga solo la primera vez (~140 MB).
    model = YOLOWorld("yolov8x-worldv2.pt")
    model.set_classes(CLASES)

    salida = Path("salidas")
    salida.mkdir(exist_ok=True)

    results = model.predict(str(img_path), conf=0.01, verbose=False)
    r = results[0]
    r.save(filename=str(salida / f"yoloworld_{img_path.stem}.jpg"))

    print(f"\n=== YOLO-World :: {img_path.name} ===")
    if len(r.boxes) == 0:
        print("No se detectó nada por encima del umbral.")
    for b in r.boxes:
        cls = CLASES[int(b.cls)]
        conf = float(b.conf)
        xyxy = [round(v, 1) for v in b.xyxy[0].tolist()]
        print(f"  {cls:28s} conf={conf:.2f}  bbox={xyxy}")
    print(f"\nImagen anotada guardada en: salidas/yoloworld_{img_path.stem}.jpg")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Uso: python yolo_world.py <ruta_imagen>")
    main(sys.argv[1])
