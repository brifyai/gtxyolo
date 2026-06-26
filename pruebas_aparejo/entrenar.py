"""
Entrena un clasificador YOLO Large (conforme vs dado_de_baja) en el dataset balanceado v3.
Usa CUDA (GPU NVIDIA) si está disponible, o MPS (Apple Silicon) como fallback.
"""

from pathlib import Path

import torch

from ultralytics import YOLO

DATA = str(Path("dataset_cls").resolve())


def main():
    # Seleccionar dispositivo automáticamente (CUDA en tu nueva RTX 5060, MPS en Mac, o CPU)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Utilizando dispositivo de entrenamiento: {device}")

    model = YOLO("yolo11l-cls.pt")  # Modelo Large para máxima capacidad de visualización de fisuras
    model.train(
        data=DATA,
        epochs=40,
        imgsz=640,  # Alta resolución para no perder detalles de fisuras en metales
        batch=16,  # Batch size de 16 (ideal para GPUs de 8GB/12GB VRAM como la RTX 5060 a 640x640)
        device=device,
        project="runs_aparejo",
        name="cls_v3_large_640",
        patience=10,
    )
    # Evaluación final sobre el split de test
    metrics = model.val(split="test")
    print("\n=== Métricas en TEST ===")
    print("top1 accuracy:", metrics.top1)


if __name__ == "__main__":
    main()
