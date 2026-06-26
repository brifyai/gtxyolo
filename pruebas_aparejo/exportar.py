"""
Exporta el modelo entrenado a formatos optimizados para móviles (TFLite y CoreML)
para permitir el uso 100% offline en dispositivos Android e iOS.
"""

from pathlib import Path

from ultralytics import YOLO

MODEL_PATH = Path("runs/classify/runs_aparejo/cls_v2_11k/weights/best.pt")


def main():
    model_path = MODEL_PATH
    if not model_path.exists():
        model_path = Path("pruebas_aparejo") / MODEL_PATH

    if not model_path.exists():
        print(f"No se encontró el archivo de pesos del modelo en: {MODEL_PATH}")
        return

    print(f"Cargando modelo entrenado desde {model_path}...")
    model = YOLO(str(model_path))

    # Exportar a TFLite (Android / Multiplataforma)
    print("\n=== Exportando a formato TFLite (Android)... ===")
    try:
        tflite_path = model.export(format="tflite", imgsz=224)
        print(f"✅ Guardado modelo TFLite en: {tflite_path}")
    except Exception as e:
        print(f"❌ Error al exportar a TFLite: {e}")

    # Exportar a CoreML (iOS / macOS)
    print("\n=== Exportando a formato CoreML (iOS)... ===")
    try:
        coreml_path = model.export(format="coreml", imgsz=224)
        print(f"✅ Guardado modelo CoreML en: {coreml_path}")
    except Exception as e:
        print(f"❌ Error al exportar a CoreML: {e}")


if __name__ == "__main__":
    main()
