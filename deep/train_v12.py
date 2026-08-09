"""Entrenamiento v12: YOLO26m multi-finca con augmentation cenital.

Receta (2026-08-09), con las lecciones de v1-v11:
- YOLO26m desde COCO (STAL + ProgLoss: mejor en objetos pequeños; NMS-free).
- 768px batch 4: con batch 8, 26m usaba 7.8/8.1 GiB y CUDA paginaba a memoria
  compartida (5.6 s/it, 9x más lento). batch 4 deja margen para el escritorio.
- workers=0 SIEMPRE en Windows (v2/seg2 murieron con workers>0).
- Augmentation cenital: la vista nadir es invariante a rotación y volteo vertical,
  los defaults de YOLO (solo fliplr) no lo aprovechan. scale 0.6 simula altura.
- Datos v12: v10 + elliot/m2/lasuiza (dedup); armah reservada como finca nunca vista.

Uso: .venv\\Scripts\\python.exe deep\\train_v12.py
Si se corta: resume=True con model=runs12/banana_v12_26m/weights/last.pt
"""
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from ultralytics import YOLO  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

model = YOLO("yolo26m.pt")
model.train(
    data=str(ROOT / "realdata/v12.yaml"),
    epochs=80,
    patience=20,
    imgsz=768,
    batch=4,
    workers=0,
    device=0,
    project=str(ROOT / "runs12"),
    name="banana_v12_26m",
    exist_ok=True,
    # augmentation cenital
    degrees=180.0,
    flipud=0.5,
    scale=0.6,
)
