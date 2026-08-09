"""Evalúa un modelo YOLO sobre un data.yaml y REGISTRA las métricas en un JSON.

Nace de una lección de julio-2026: el 0.747 de v10 en el holdout cross-finca no
quedó escrito en ningún fichero del repo. Cada evaluación importante debe dejar
huella con fecha, pesos, datos y cifras.

Uso:
  .venv\\Scripts\\python.exe deep\\eval_record.py --weights runs10/.../best.pt \\
      --data realdata/holdout.yaml --imgsz 768 --name v10_holdout
Escribe real_eval/<name>.json (y lo imprime).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--imgsz", type=int, default=768)
    ap.add_argument("--name", required=True, help="nombre del JSON en real_eval/")
    ap.add_argument("--conf", type=float, default=0.001, help="conf para métricas mAP")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    weights = Path(args.weights)
    sha = hashlib.sha256(weights.read_bytes()).hexdigest()[:16]
    model = YOLO(str(weights))
    m = model.val(
        data=str(args.data), imgsz=args.imgsz, conf=args.conf, batch=args.batch,
        device=0, workers=0, plots=False, verbose=False,
    )
    record = {
        "fecha": dt.datetime.now().isoformat(timespec="seconds"),
        "weights": str(weights),
        "weights_sha256_16": sha,
        "data": str(args.data),
        "imgsz": args.imgsz,
        "metrics": {
            "mAP50": round(float(m.box.map50), 4),
            "mAP50_95": round(float(m.box.map), 4),
            "precision": round(float(m.box.mp), 4),
            "recall": round(float(m.box.mr), 4),
        },
    }
    out = ROOT / "real_eval" / f"{args.name}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(record, indent=2, ensure_ascii=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
