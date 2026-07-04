"""Inferencia con un modelo YOLOv8-seg entrenado sobre tiles o una imagen.

Uso:
    python deep/infer_yolo.py --weights runs/segment/banano_seg/weights/best.pt \
        --image tile.png --out outputs
"""
from __future__ import annotations

import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit(
            "Falta ultralytics. Instala: pip install -e .[deep]"
        ) from e

    os.makedirs(args.out, exist_ok=True)
    model = YOLO(args.weights)
    results = model.predict(args.image, conf=args.conf, save=True, project=args.out, name="pred")

    counts = {}
    for r in results:
        if r.boxes is None:
            continue
        for c in r.boxes.cls.tolist():
            name = model.names[int(c)]
            counts[name] = counts.get(name, 0) + 1
    print(json.dumps({"conteo": counts}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
