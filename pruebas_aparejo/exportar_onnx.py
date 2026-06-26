"""
Exporta el modelo entrenado a formato ONNX.
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

    print("\n=== Exportando a formato ONNX... ===")
    try:
        onnx_path = model.export(format="onnx", imgsz=224)
        print(f"✅ Guardado modelo ONNX en: {onnx_path}")
    except Exception as e:
        print(f"❌ Error al exportar a ONNX: {e}")

if __name__ == "__main__":
    main()
