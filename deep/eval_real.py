"""Evaluacion HONESTA del modelo real sobre el split de TEST real retenido.

Mide precision/recall/mAP del detector de banano sobre imagenes UAV REALES que el
modelo NUNCA vio (split test), y guarda ejemplos visuales de deteccion sobre tiles
reales. Estas son las cifras defendibles de rendimiento en campo.

    python deep/eval_real.py --weights models/banano_real_v1.pt \
        --data realdata/DS-v1/ds-v1/banano_real.yaml --out real_eval
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="real_eval")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--samples", type=int, default=12)
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit("Falta ultralytics. pip install -e .[deep]") from e

    os.makedirs(args.out, exist_ok=True)
    model = YOLO(args.weights)

    # 1. metricas sobre el split de TEST real (imagenes nunca vistas)
    metrics = model.val(data=args.data, split="test", imgsz=args.imgsz, verbose=False)
    box = metrics.box
    report = {
        "split": "test (real, retenido)",
        "mAP50": round(float(box.map50), 4),
        "mAP50_95": round(float(box.map), 4),
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
        "n_imagenes_test": int(getattr(metrics, "seen", 0)) or None,
    }
    with open(os.path.join(args.out, "real_test_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))

    # 2. ejemplos visuales de deteccion sobre tiles reales de test
    root = None
    with open(args.data, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("path:"):
                root = line.split(":", 1)[1].strip()
    test_dir = os.path.join(root, "images", "test") if root else None
    if test_dir and os.path.isdir(test_dir):
        imgs = glob.glob(os.path.join(test_dir, "*.jpg")) + glob.glob(os.path.join(test_dir, "*.png"))
        random.Random(0).shuffle(imgs)
        sample = imgs[: args.samples]
        if sample:
            model.predict(sample, conf=args.conf, imgsz=args.imgsz, save=True,
                          project=args.out, name="ejemplos_reales", exist_ok=True, verbose=False)
            print(f"\nEjemplos visuales sobre tiles REALES en: {args.out}/ejemplos_reales")

    print(f"\nMetricas: {args.out}/real_test_metrics.json")


if __name__ == "__main__":
    main()
