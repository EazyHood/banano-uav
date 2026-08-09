r"""Error de CONTEO de un modelo YOLO sobre un data.yaml, con barrido de confianza.

El mAP infravalora lo que le importa a una finca: ¿cuántas plantas hay? Esta
herramienta predice UNA vez a confianza mínima y luego barre umbrales, así que
calibrar el umbral de conteo es barato. Registra el resultado en real_eval/.

Uso:
  .venv\Scripts\python.exe deep\eval_count.py --weights w.pt \
      --data realdata\holdout_armah.yaml --imgsz 768 --name v12_armah_count
  (opcional: --conf 0.30 fijo en vez de barrido)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def _val_images(data_yaml: str) -> list[Path]:
    spec = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    base = Path(spec.get("path", "."))
    val = spec["val"]
    dirs = [val] if isinstance(val, str) else list(val)
    imgs: list[Path] = []
    for d in dirs:
        p = base / d
        imgs.extend(f for f in sorted(p.iterdir()) if f.suffix.lower() in IMG_EXT)
    return imgs


def _gt_count(img: Path) -> int:
    lbl = img.parent.parent / "labels" / (img.stem + ".txt")
    if not lbl.exists():
        return 0
    return sum(1 for line in lbl.read_text().splitlines() if line.strip())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--imgsz", type=int, default=768)
    ap.add_argument("--name", required=True)
    ap.add_argument("--conf", type=float, default=None, help="umbral fijo (sin barrido)")
    args = ap.parse_args()

    imgs = _val_images(args.data)
    gts = np.array([_gt_count(p) for p in imgs])
    model = YOLO(args.weights)

    # una sola pasada a conf minima; el conteo por umbral se deriva de las confianzas
    det_confs: list[np.ndarray] = []
    for r in model.predict([str(p) for p in imgs], conf=0.01, imgsz=args.imgsz,
                           device=0, stream=True, verbose=False):
        b = getattr(r, "boxes", None)
        det_confs.append(b.conf.cpu().numpy() if b is not None and b.conf is not None
                         else np.empty(0))

    def count_metrics(thr: float) -> dict:
        preds = np.array([(c >= thr).sum() for c in det_confs])
        total_gt, total_pred = int(gts.sum()), int(preds.sum())
        nz = gts > 0
        mape = float(np.mean(np.abs(preds[nz] - gts[nz]) / gts[nz])) if nz.any() else None
        return {
            "conf": round(thr, 3),
            "total_gt": total_gt,
            "total_pred": total_pred,
            "error_conteo_total": round(abs(total_pred - total_gt) / total_gt, 4)
            if total_gt else None,
            "MAPE_por_imagen": round(mape, 4) if mape is not None else None,
        }

    if args.conf is not None:
        best = count_metrics(args.conf)
        sweep = [best]
    else:
        sweep = [count_metrics(t) for t in np.arange(0.05, 0.91, 0.05)]
        best = min((s for s in sweep if s["error_conteo_total"] is not None),
                   key=lambda s: (s["error_conteo_total"], s["MAPE_por_imagen"] or 1))

    record = {
        "fecha": dt.datetime.now().isoformat(timespec="seconds"),
        "weights": str(args.weights),
        "data": str(args.data),
        "imgsz": args.imgsz,
        "n_imagenes": len(imgs),
        "mejor": best,
        "barrido": sweep,
    }
    out = ROOT / "real_eval" / f"{args.name}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: record[k] for k in ("weights", "data", "n_imagenes", "mejor")},
                     indent=2, ensure_ascii=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
