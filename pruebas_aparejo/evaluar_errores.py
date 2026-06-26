"""
Analiza los errores de predicción del modelo entrenado sobre el conjunto de test.
Genera un reporte detallado con las imágenes mal clasificadas, confianza y metadatos.
"""

import csv
from pathlib import Path

import torch

from ultralytics import YOLO

MODEL_PATH = Path("runs/classify/runs_aparejo/cls_v3_large_640/weights/best.pt")
ETIQUETAS_CSV = Path("dataset_cls/etiquetas.csv")


def main():
    model_path = MODEL_PATH
    if not model_path.exists():
        model_path = Path("pruebas_aparejo") / MODEL_PATH

    if not model_path.exists():
        print(f"No se encontró el modelo en {MODEL_PATH} ni en {Path('pruebas_aparejo') / MODEL_PATH}")
        return

    print(f"Cargando modelo desde {model_path}...")
    model = YOLO(str(model_path))
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    csv_path = ETIQUETAS_CSV
    if not csv_path.exists():
        csv_path = Path("pruebas_aparejo") / ETIQUETAS_CSV

    if not csv_path.exists():
        print(f"No se encontró el archivo de etiquetas en {ETIQUETAS_CSV}")
        return

    # Leer con csv.DictReader
    df_test = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] == "test":
                df_test.append(row)

    print(f"Evaluando {len(df_test)} imágenes de test...")

    errores = []
    aciertos = 0

    for row in df_test:
        img_path = Path(row["archivo"])
        if not img_path.exists():
            alt_img_path = Path("pruebas_aparejo") / img_path
            if alt_img_path.exists():
                img_path = alt_img_path
            else:
                continue

        # Inferencia
        results = model(str(img_path), verbose=False)
        result = results[0]

        # Obtener predicción
        probs = result.probs
        top1_idx = probs.top1
        pred_class = result.names[top1_idx]  # conforme o dado_de_baja
        conf = float(probs.top1conf)

        real_class = row["clase"]  # conforme o dado_de_baja

        if pred_class == real_class:
            aciertos += 1
        else:
            errores.append(
                {
                    "Archivo": img_path.name,
                    "Dictamen Real": real_class,
                    "Predicción": pred_class,
                    "Confianza": f"{conf:.2%}",
                    "Tipo": row["tipo"],
                    "Defecto": row["defecto"],
                    "URL": row["url"],
                }
            )

    total = len(df_test)
    accuracy = aciertos / total if total > 0 else 0
    print("\nResultados de Test:")
    print(f"Total: {total} | Aciertos: {aciertos} | Errores: {len(errores)}")
    print(f"Accuracy calculado: {accuracy:.2%}")

    # Generar reporte Markdown
    reporte_path = Path("reporte_errores.md")
    if csv_path.parent.name == "dataset_cls":
        reporte_path = csv_path.parent.parent / "reporte_errores.md"

    with open(reporte_path, "w", encoding="utf-8") as f:
        f.write("# Reporte de Errores de Clasificación\n\n")
        f.write(f"- **Total imágenes de test:** {total}\n")
        f.write(f"- **Aciertos:** {aciertos} ({accuracy:.2%})\n")
        f.write(f"- **Errores (Falsos Positivos / Falsos Negativos):** {len(errores)} ({1 - accuracy:.2%})\n\n")

        f.write("## Detalle de Errores\n\n")
        f.write("| Archivo | Dictamen Real | Predicción | Confianza | Tipo Producto | Defecto/Comentario | URL |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for err in errores:
            f.write(
                f"| {err['Archivo']} | **{err['Dictamen Real']}** | *{err['Predicción']}* | {err['Confianza']} | {err['Tipo']} | {err['Defecto']} | [Ver Imagen]({err['URL']}) |\n"
            )

    print(f"\nReporte de errores guardado en: {reporte_path.resolve()}")


if __name__ == "__main__":
    main()
