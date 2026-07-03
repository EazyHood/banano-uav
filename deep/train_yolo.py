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
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit(
            "Falta ultralytics. Instala: pip install -r requirements-deep.txt"
        )

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        patience=25,
        # aumentos utiles para dron: mosaico + rotaciones (banano es rotacion-invariante)
        degrees=180,
        fliplr=0.5,
        flipud=0.5,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
    )
    print("Entrenamiento terminado. Pesos en runs/segment/%s/weights/best.pt" % args.name)


if __name__ == "__main__":
    main()
