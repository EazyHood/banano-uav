r"""Smoke test: YOLO26 sobre la RTX 5060 (sm_120) — inferencia + mini-train.

Uso: .venv\Scripts\python.exe deep\smoke_yolo26.py
"""
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0))

model = YOLO("yolo26m.pt")  # descarga los pesos si faltan
imgs = sorted((ROOT / "realdata/count_banana_plants/valid/images").glob("*.jpg"))[:2]
res = model.predict([str(p) for p in imgs], imgsz=768, conf=0.25, device=0, verbose=False)
for p, r in zip(imgs, res):
    print(f"predict {p.name}: {len(r.boxes)} cajas (COCO, sin fine-tune: se espera ~0)")

# Mini-train: 1 época sobre una fracción pequeña para validar backward + AMP en sm_120.
model.train(
    data=str(ROOT / "realdata/t768.yaml"),
    epochs=1,
    fraction=0.1,
    imgsz=768,
    batch=8,
    workers=0,
    device=0,
    project=str(ROOT / "runs12"),
    name="smoke26",
    exist_ok=True,
    plots=False,
    val=False,
    verbose=False,
)
print("SMOKE OK: YOLO26m entrena en esta GPU")
