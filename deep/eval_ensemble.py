r"""Evalua un ENSAMBLE de modelos (o uno solo) con el mismo raton de medir.

Por que existe: ultralytics sabe evaluar UN modelo. Para saber si fusionar dos
modelos mejora algo hay que medir los tres con el MISMO codigo, o la comparacion
no vale. Este script hace las tres cosas con el mismo camino.

Control obligatorio (`--validar`): con un solo modelo, el mAP50 que calcula aqui
debe parecerse al que dejo ultralytics en real_eval/. Si no se parece, las cifras
del ensamble no valen nada y el script lo dice.

Dos formas de puntuar una deteccion fusionada, y la diferencia es el experimento:
  --score presentes : media ponderada entre los modelos QUE LA VIERON. Una planta
                      vista por uno solo conserva su confianza -> mas recall.
  --score todos     : media ponderada sobre TODOS los modelos. La vista por uno
                      solo se queda a la mitad -> mas precision, menos recall.

Uso:
  python deep/eval_ensemble.py --weights models/banana_multifarm_v10.pt \
      --data realdata/holdout_armah.yaml --name val_v10_armah --validar 0.172
  python deep/eval_ensemble.py --weights models/a.pt runs12/.../best.pt \
      --pesos 1 1 --score presentes --data realdata/holdout_armah.yaml \
      --name ens_armah
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_count import _gt_count, _pliegues, _val_images  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- cajas


def _iou_matriz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU de cada caja de `a` (N,4) contra cada una de `b` (M,4), en xyxy."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def fusionar(
    por_modelo: list[tuple[np.ndarray, np.ndarray]],
    pesos: list[float],
    iou_thr: float,
    modo: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Agrupa las cajas de varios modelos y devuelve (cajas, puntuaciones).

    Agrupamiento avaro por confianza: cada caja entra en el primer grupo cuyo
    representante solape >= iou_thr. La caja del grupo es la media ponderada por
    confianza*peso; la puntuacion depende de `modo` (ver el docstring del modulo).

    No se usa `ensemble_boxes.weighted_boxes_fusion` a proposito: reescala la
    puntuacion fusionada por (tamano del grupo / n de modelos), asi que una planta
    vista por un solo modelo se hunde bajo el umbral sin que nadie lo note. Aqui
    esa decision es explicita y se puede medir en las dos direcciones.
    """
    cajas_l, punt_l, modelo_l = [], [], []
    for i, (b, s) in enumerate(por_modelo):
        if len(b):
            cajas_l.append(b)
            punt_l.append(s)
            modelo_l.append(np.full(len(b), i))
    if not cajas_l:
        return np.empty((0, 4)), np.empty(0)
    cajas = np.concatenate(cajas_l)
    punt = np.concatenate(punt_l)
    modelo = np.concatenate(modelo_l)

    orden = np.argsort(-punt)
    cajas, punt, modelo = cajas[orden], punt[orden], modelo[orden]

    reps: list[np.ndarray] = []          # caja representante de cada grupo
    grupos: list[list[int]] = []
    for i in range(len(cajas)):
        if reps:
            ious = _iou_matriz(cajas[i : i + 1], np.array(reps))[0]
            j = int(np.argmax(ious))
            if ious[j] >= iou_thr:
                grupos[j].append(i)
                continue
        reps.append(cajas[i])
        grupos.append([i])

    peso_total = float(sum(pesos))
    out_cajas, out_punt = [], []
    for miembros in grupos:
        idx = np.array(miembros)
        w = punt[idx] * np.array([pesos[m] for m in modelo[idx]])
        out_cajas.append((cajas[idx] * w[:, None]).sum(0) / max(w.sum(), 1e-9))
        # Un modelo puede aportar varias cajas al grupo: se queda con su mejor.
        mejor_por_modelo: dict[int, float] = {}
        for k in idx:
            m = int(modelo[k])
            mejor_por_modelo[m] = max(mejor_por_modelo.get(m, 0.0), float(punt[k]))
        suma = sum(pesos[m] * s for m, s in mejor_por_modelo.items())
        divisor = (
            sum(pesos[m] for m in mejor_por_modelo) if modo == "presentes" else peso_total
        )
        out_punt.append(suma / max(divisor, 1e-9))
    return np.array(out_cajas), np.array(out_punt)


# ----------------------------------------------------------------------------- AP


def ap50(
    preds: list[tuple[np.ndarray, np.ndarray]], gts: list[np.ndarray]
) -> tuple[float, float, float]:
    """AP a IoU 0.5 (interpolacion de 101 puntos) + precision y recall al mejor F1."""
    n_gt = sum(len(g) for g in gts)
    if n_gt == 0:
        return 0.0, 0.0, 0.0
    todas_p, todos_tp = [], []
    for (pb, ps), gb in zip(preds, gts):
        if len(pb) == 0:
            continue
        orden = np.argsort(-ps)
        pb, ps = pb[orden], ps[orden]
        tp = np.zeros(len(ps), dtype=bool)
        if len(gb):
            ious = _iou_matriz(pb, gb)
            usada = np.zeros(len(gb), dtype=bool)
            for i in range(len(pb)):
                fila = np.where(usada, -1.0, ious[i])
                j = int(np.argmax(fila))
                if fila[j] >= 0.5:
                    usada[j] = True
                    tp[i] = True
        todas_p.append(ps)
        todos_tp.append(tp)
    if not todas_p:
        return 0.0, 0.0, 0.0

    punt = np.concatenate(todas_p)
    tp = np.concatenate(todos_tp)
    orden = np.argsort(-punt)
    tp = tp[orden]
    tpc = np.cumsum(tp)
    fpc = np.cumsum(~tp)
    recall = tpc / n_gt
    precision = tpc / np.maximum(tpc + fpc, 1e-9)

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    ap = float(np.trapezoid(np.interp(x, mrec, mpre), x))

    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    k = int(np.argmax(f1))
    return ap, float(precision[k]), float(recall[k])


# ------------------------------------------------------------------------- conteo


def metricas_conteo(
    punt_por_imagen: list[np.ndarray], gts: np.ndarray, grupos: list[int]
) -> tuple[dict, dict | None, list[dict]]:
    """Error de conteo: el in-sample y el honesto (calibracion cruzada en 2 pliegues)."""
    thrs = [round(float(t), 3) for t in np.arange(0.05, 0.91, 0.05)]

    def preds(t: float) -> np.ndarray:
        return np.array([int((s >= t).sum()) for s in punt_por_imagen])

    def resumen(p: np.ndarray, idx: np.ndarray | None = None) -> dict:
        g = gts if idx is None else gts[idx]
        pp = p if idx is None else p[idx]
        total_gt, total_pred = int(g.sum()), int(pp.sum())
        nz = g > 0
        mape = float(np.mean(np.abs(pp[nz] - g[nz]) / g[nz])) if nz.any() else None
        return {
            "total_gt": total_gt,
            "total_pred": total_pred,
            "error_conteo_total": round(abs(total_pred - total_gt) / total_gt, 4)
            if total_gt
            else None,
            "MAPE_por_imagen": round(mape, 4) if mape is not None else None,
        }

    def con_conf(t: float, idx: np.ndarray | None = None) -> dict:
        return {"conf": round(float(t), 3), **resumen(preds(t), idx)}

    def mejor_en(idx: np.ndarray | None) -> dict | None:
        cands = [c for c in (con_conf(t, idx) for t in thrs)
                 if c["error_conteo_total"] is not None]
        return min(cands, key=lambda s: (s["error_conteo_total"], s["MAPE_por_imagen"] or 1)) \
            if cands else None

    barrido = [con_conf(t) for t in thrs]
    mejor = mejor_en(None)

    honesto = None
    ia, ib = _pliegues(grupos)
    if ia and ib:
        a, b = np.array(ia), np.array(ib)
        ma, mb = mejor_en(a), mejor_en(b)
        if ma and mb:
            mezcla = np.empty(len(gts), dtype=int)
            mezcla[b] = preds(ma["conf"])[b]
            mezcla[a] = preds(mb["conf"])[a]
            honesto = {
                "conf_de_A_medido_en_B": ma["conf"],
                "conf_de_B_medido_en_A": mb["conf"],
                **resumen(mezcla),
            }
    return mejor or {}, honesto, barrido


# --------------------------------------------------------------------------- main


def _gt_cajas(img: Path, w: int, h: int) -> np.ndarray:
    """Lee la verdad de terreno del .txt y la pasa a xyxy en pixeles.

    Acepta las DOS formas que hay en este corpus:
      caja      -> `clase cx cy bw bh`                    (5 columnas)
      poligono  -> `clase x1 y1 x2 y2 ... xn yn`  (impar, >= 7 columnas)

    El poligono se convierte en su caja envolvente, que es lo que hace ultralytics
    al cargar un dataset de segmentacion como si fuera de deteccion.

    Antes esta funcion hacia `p[1:5]` a secas. Con un poligono eso leia los dos
    primeros VERTICES como si fueran centro y tamano, y no daba error: producia
    cajas disparatadas en silencio. Afecta de lleno a realdata/newfarms/lasuiza,
    donde 61 de 61 lineas del split de test son poligonos.
    """
    lbl = img.parent.parent / "labels" / (img.stem + ".txt")
    if not lbl.exists():
        return np.empty((0, 4))
    filas = []
    for linea in lbl.read_text().splitlines():
        p = linea.split()
        if len(p) == 5:
            cx, cy, bw, bh = (float(v) for v in p[1:5])
            x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
        elif len(p) >= 7 and len(p) % 2 == 1:
            vs = [float(v) for v in p[1:]]
            xs, ys = vs[0::2], vs[1::2]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        else:
            continue
        filas.append([x1 * w, y1 * h, x2 * w, y2 * h])
    return np.array(filas) if filas else np.empty((0, 4))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--pesos", nargs="*", type=float, default=None)
    ap.add_argument("--data", required=True)
    ap.add_argument("--imgsz", type=int, default=768)
    ap.add_argument("--name", required=True)
    ap.add_argument("--iou-fusion", type=float, default=0.55)
    ap.add_argument("--score", choices=("presentes", "todos"), default="presentes")
    ap.add_argument("--conf-min", type=float, default=0.001,
                    help="suelo de confianza (0.001 = el que usa ultralytics val)")
    ap.add_argument("--max-det", type=int, default=300,
                    help="detecciones por imagen (300 = el que usa ultralytics val)")
    ap.add_argument("--iou-nms", type=float, default=0.7,
                    help="IoU del NMS de cada modelo (0.7 = el de ultralytics val)")
    ap.add_argument("--validar", type=float, default=None,
                    help="mAP50 esperado (de ultralytics) para comprobar el evaluador")
    args = ap.parse_args()

    pesos = args.pesos or [1.0] * len(args.weights)
    if len(pesos) != len(args.weights):
        sys.exit("--pesos debe tener tantos valores como --weights")

    imgs, grupos = _val_images(args.data)
    gts_conteo = np.array([_gt_count(p) for p in imgs])
    print(f"{len(imgs)} imagenes, {int(gts_conteo.sum())} plantas reales", flush=True)

    # (modelo, imagen) -> cajas y confianzas, en pixeles
    por_modelo_por_img: list[list[tuple[np.ndarray, np.ndarray]]] = []
    formas: list[tuple[int, int]] = []
    for wi, w in enumerate(args.weights):
        modelo = YOLO(w)
        fila = []
        for salida in modelo.predict([str(p) for p in imgs], conf=args.conf_min,
                                     imgsz=args.imgsz, device=0, stream=True,
                                     verbose=False, max_det=args.max_det,
                                     iou=args.iou_nms):
            r: Any = salida
            b = r.boxes
            cajas = b.xyxy.cpu().numpy() if b is not None and len(b) else np.empty((0, 4))
            punt = b.conf.cpu().numpy() if b is not None and len(b) else np.empty(0)
            fila.append((cajas, punt))
            if wi == 0:
                formas.append(r.orig_shape)  # (alto, ancho)
        por_modelo_por_img.append(fila)
        print(f"  predicho con {Path(w).name}", flush=True)

    fusionadas: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(len(imgs)):
        entrada = [por_modelo_por_img[m][i] for m in range(len(args.weights))]
        fusionadas.append(
            entrada[0] if len(entrada) == 1
            else fusionar(entrada, pesos, args.iou_fusion, args.score)
        )

    gts_cajas = [_gt_cajas(p, formas[i][1], formas[i][0]) for i, p in enumerate(imgs)]
    mapa, precision, recall = ap50(fusionadas, gts_cajas)

    mejor, honesto, barrido = metricas_conteo(
        [s for _, s in fusionadas], gts_conteo, grupos
    )

    aviso_validacion = None
    if args.validar is not None:
        d = abs(mapa - args.validar)
        aviso_validacion = (
            f"mAP50 propio {mapa:.4f} vs ultralytics {args.validar:.4f} (dif {d:.4f}): "
            + ("COHERENTE" if d <= 0.03 else "NO COINCIDE — no usar estas cifras")
        )
        print(aviso_validacion, flush=True)

    registro = {
        "fecha": dt.datetime.now().isoformat(timespec="seconds"),
        "weights": [str(w) for w in args.weights],
        "pesos": pesos,
        "modo_score": args.score,
        "iou_fusion": args.iou_fusion,
        "data": str(args.data),
        "imgsz": args.imgsz,
        "n_imagenes": len(imgs),
        "medido_con": "deep/eval_ensemble.py (evaluador propio, NO ultralytics)",
        "validacion": aviso_validacion,
        "metrics": {"mAP50": round(mapa, 4),
                    "precision_mejor_F1": round(precision, 4),
                    "recall_mejor_F1": round(recall, 4)},
        "mejor": mejor,
        "honesto": honesto,
        "barrido": barrido,
    }
    out = ROOT / "real_eval" / f"{args.name}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(registro, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: registro[k] for k in ("metrics", "mejor", "honesto")},
                     indent=2, ensure_ascii=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
