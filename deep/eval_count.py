r"""Error de CONTEO de un modelo YOLO sobre un data.yaml, con barrido de confianza.

El mAP infravalora lo que le importa a una finca: ¿cuántas plantas hay? Esta
herramienta predice UNA vez a confianza mínima y luego barre umbrales, así que
calibrar el umbral de conteo es barato. Registra el resultado en real_eval/.

DOS cifras, y la diferencia importa:

  - `mejor`   : el mejor umbral del barrido y su error. Es **in-sample**: el umbral
                se elige mirando el mismo conjunto sobre el que se reporta el error,
                así que es el techo optimista, no lo que verás en una finca nueva.
  - `honesto` : calibración cruzada en 2 pliegues. Parte cada carpeta de val por la
                mitad, elige el umbral en un pliegue y reporta el error en el OTRO
                (y al revés). Nadie eligió nada mirando lo que luego se mide.

`honesto` es la cifra publicable. Sale gratis: las predicciones ya están hechas.

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


def _val_images(data_yaml: str) -> tuple[list[Path], list[int]]:
    """Devuelve las imagenes de val y, por imagen, el indice de su carpeta de origen.

    La carpeta importa: cada una suele ser una finca distinta, y los pliegues de la
    calibracion cruzada se parten dentro de cada carpeta para que ningun pliegue se
    quede sin una finca entera.
    """
    spec = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    base = Path(spec.get("path", "."))
    val = spec["val"]
    dirs = [val] if isinstance(val, str) else list(val)
    imgs: list[Path] = []
    grupos: list[int] = []
    for gi, d in enumerate(dirs):
        p = base / d
        encontradas = [f for f in sorted(p.iterdir()) if f.suffix.lower() in IMG_EXT]
        imgs.extend(encontradas)
        grupos.extend([gi] * len(encontradas))
    return imgs, grupos


def _pliegues(grupos: list[int]) -> tuple[list[int], list[int]]:
    """Parte cada carpeta por la mitad: bloque inicial -> A, bloque final -> B.

    Por bloques y no alternando: los tiles vecinos de un mismo vuelo se parecen tanto
    que un pliegue alternado seria casi una copia del otro y la cifra "honesta"
    saldria tan optimista como la in-sample.
    """
    a: list[int] = []
    b: list[int] = []
    for gi in sorted(set(grupos)):
        idx = [i for i, g in enumerate(grupos) if g == gi]
        corte = len(idx) // 2
        a.extend(idx[:corte])
        b.extend(idx[corte:])
    return a, b


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

    imgs, grupos = _val_images(args.data)
    gts = np.array([_gt_count(p) for p in imgs])
    model = YOLO(args.weights)

    # una sola pasada a conf minima; el conteo por umbral se deriva de las confianzas
    det_confs: list[np.ndarray] = []
    for r in model.predict([str(p) for p in imgs], conf=0.01, imgsz=args.imgsz,
                           device=0, stream=True, verbose=False):
        b = getattr(r, "boxes", None)
        det_confs.append(b.conf.cpu().numpy() if b is not None and b.conf is not None
                         else np.empty(0))

    def _preds(thr: float) -> np.ndarray:
        return np.array([int((c >= thr).sum()) for c in det_confs])

    def _resumen(preds: np.ndarray, idx: np.ndarray | None = None) -> dict:
        g = gts if idx is None else gts[idx]
        p = preds if idx is None else preds[idx]
        total_gt, total_pred = int(g.sum()), int(p.sum())
        nz = g > 0
        mape = float(np.mean(np.abs(p[nz] - g[nz]) / g[nz])) if nz.any() else None
        return {
            "total_gt": total_gt,
            "total_pred": total_pred,
            "error_conteo_total": round(abs(total_pred - total_gt) / total_gt, 4)
            if total_gt else None,
            "MAPE_por_imagen": round(mape, 4) if mape is not None else None,
        }

    def count_metrics(thr: float, idx: np.ndarray | None = None) -> dict:
        return {"conf": round(float(thr), 3), **_resumen(_preds(thr), idx)}

    def _mejor_en(idx: np.ndarray | None, thrs: list[float]) -> dict | None:
        cands = [c for c in (count_metrics(t, idx) for t in thrs)
                 if c["error_conteo_total"] is not None]
        if not cands:
            return None
        return min(cands, key=lambda s: (s["error_conteo_total"], s["MAPE_por_imagen"] or 1))

    thrs = [round(float(t), 3) for t in np.arange(0.05, 0.91, 0.05)]

    honesto: dict | None = None
    best: dict | None
    if args.conf is not None:
        best = count_metrics(args.conf)
        sweep = [best]
        thrs = [round(float(args.conf), 3)]
    else:
        sweep = [count_metrics(t) for t in thrs]
        best = _mejor_en(None, thrs)

        # Calibracion cruzada en 2 pliegues: el umbral se elige en un pliegue y el
        # error se mide en el otro. Es la unica de las dos cifras que puede publicarse.
        ia, ib = _pliegues(grupos)
        if ia and ib:
            a, b = np.array(ia), np.array(ib)
            ma, mb = _mejor_en(a, thrs), _mejor_en(b, thrs)
            if ma and mb:
                mezcla = np.empty(len(gts), dtype=int)
                mezcla[b] = _preds(ma["conf"])[b]   # umbral de A -> se mide en B
                mezcla[a] = _preds(mb["conf"])[a]   # umbral de B -> se mide en A
                honesto = {
                    "conf_de_A_medido_en_B": ma["conf"],
                    "conf_de_B_medido_en_A": mb["conf"],
                    "n_A": len(ia),
                    "n_B": len(ib),
                    **_resumen(mezcla),
                }

    record = {
        "fecha": dt.datetime.now().isoformat(timespec="seconds"),
        "weights": str(args.weights),
        "data": str(args.data),
        "imgsz": args.imgsz,
        "n_imagenes": len(imgs),
        "aviso_mejor": "in-sample: el umbral se eligio mirando este mismo conjunto. "
                       "Para publicar, usa 'honesto' (calibracion cruzada en 2 pliegues).",
        "mejor": best,
        "honesto": honesto,
        "barrido": sweep,
        # Conteos por umbral y por imagen: permiten recalibrar cualquier cosa despues
        # sin volver a tocar la GPU (esta suite ya se perdio una vez por no guardarlos).
        "umbrales": thrs,
        "gt_por_imagen": [int(x) for x in gts],
        "grupo_por_imagen": grupos,
        "conteos_por_umbral": [[int(x) for x in _preds(t)] for t in thrs],
    }
    out = ROOT / "real_eval" / f"{args.name}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: record[k] for k in ("weights", "data", "n_imagenes", "mejor",
                                             "honesto")},
                     indent=2, ensure_ascii=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
