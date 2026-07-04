"""Entrena YOLOv8-seg para segmentacion de instancias de banano (produccion).

Camino de MAXIMA precision, cuando ya tienes tiles etiquetados. Requiere:
    pip install -r requirements-deep.txt

Uso:
    python deep/train_yolo.py --data deep/banana.yaml --epochs 100 --model yolov8s-seg.pt
"""
from __future__ import annotations

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="deep/banana.yaml")
    ap.add_argument("--model", default="yolov8s-seg.pt", help="Pesos base preentrenados")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--name", default="banano_seg")
    ap.add_argument("--workers", type=int, default=0,
                    help="Workers del DataLoader (0 evita fallos CUDA en Windows)")
    ap.add_argument("--seed", type=int, default=0, help="Semilla para reproducibilidad")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit(
            "Falta ultralytics. Instala: pip install -e .[deep]"
        ) from e

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        patience=25,
        workers=args.workers,
        seed=args.seed,
        deterministic=True,
        # aumentos utiles para dron: mosaico + rotaciones (banano es rotacion-invariante)
        degrees=180,
        fliplr=0.5,
        flipud=0.5,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
    )
    print(f"Entrenamiento terminado. Pesos en runs/segment/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
