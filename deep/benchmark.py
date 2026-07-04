"""Benchmark honesto de precision sobre datos sinteticos con verdad de terreno.

Mide el error de conteo (MAE, RMSE, MAPE) y la localizacion (precision/recall/F1)
a nivel de MACOLLA, para el pipeline clasico y (opcionalmente) para el modelo de
deep learning. Los datos son SINTETICOS con ground truth exacto: el proposito es
medir de forma reproducible, no afirmar precision sobre banano real de campo.

    python deep/benchmark.py --n 20 --size 1024
    python deep/benchmark.py --n 20 --size 1024 --weights models/banano_seg_synth_v1.pt

Genera un JSON con las metricas y las imprime en una tabla.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

import imageio.v2 as imageio
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from banano.config import PipelineConfig  # noqa: E402
from banano.geo import Raster  # noqa: E402
from banano.metrics import match_points, prf  # noqa: E402
from banano.ortho import process_orthomosaic  # noqa: E402
from banano.synth import synth_plantation  # noqa: E402


def _agg(counts_gt, counts_pred, f1s):
    gt = np.array(counts_gt, float)
    pred = np.array(counts_pred, float)
    err = pred - gt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mape = float(np.mean(np.abs(err) / np.clip(gt, 1, None)) * 100)
    bias = float(np.mean(err))
    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape_%": round(mape, 2),
        "sesgo": round(bias, 2),
        "f1_localizacion": round(float(np.mean(f1s)), 3),
        "conteo_total_gt": int(gt.sum()),
        "conteo_total_pred": int(pred.sum()),
        "error_conteo_total_%": round(abs(pred.sum() - gt.sum()) / max(1, gt.sum()) * 100, 2),
    }


def run_bench(n, size, gsd, weights=None, model_conf=0.6, seed0=900000):
    tmp = tempfile.mkdtemp()
    cfg_classical = PipelineConfig(gsd_cm=gsd, tile=640, overlap=96).validate()
    cfg_model = None
    if weights:
        cfg_model = PipelineConfig(
            gsd_cm=gsd, tile=640, overlap=96,
            model_weights=weights, model_conf=model_conf,
        ).validate()

    res = {"classical": {"gt": [], "pred": [], "f1": []},
           "model": {"gt": [], "pred": [], "f1": []}}

    for i in range(n):
        seed = seed0 + i
        spacing = float(np.random.default_rng(seed).uniform(2.3, 2.9))
        img, _gt_ps, gt_mats, _ = synth_plantation(H=size, W=size, gsd_cm=gsd,
                                                   spacing_m=spacing, seed=seed)
        path = os.path.join(tmp, f"b{i}.png")
        imageio.imwrite(path, img)
        n_gt = len(gt_mats)

        for name, cfg in (("classical", cfg_classical), ("model", cfg_model)):
            if cfg is None:
                continue
            raster = Raster(path, gsd_cm=gsd)
            r = process_orthomosaic(raster, config=cfg, progress=None)
            raster.close()
            centroids = np.array([m["centroid"] for m in r.mats]) if r.mats else np.empty((0, 2))
            tol = r.params.get("mat_eps_px", 25)
            tp, fp, fn, _ = match_points(centroids, gt_mats, tol)
            _, _, f1 = prf(tp, fp, fn)
            res[name]["gt"].append(n_gt)
            res[name]["pred"].append(r.n_mats)
            res[name]["f1"].append(f1)

    out = {"n_imagenes": n, "size_px": size, "gsd_cm": gsd}
    out["classical"] = _agg(res["classical"]["gt"], res["classical"]["pred"], res["classical"]["f1"])
    if weights:
        out["model"] = _agg(res["model"]["gt"], res["model"]["pred"], res["model"]["f1"])
        out["model"]["weights"] = weights
        out["model"]["conf"] = model_conf
    return out


def _print_table(out):
    print(f"\n=== Benchmark de MACOLLAS ({out['n_imagenes']} imagenes {out['size_px']}px, "
          f"GSD {out['gsd_cm']} cm/px) ===")
    hdr = f"{'metodo':<12}{'MAPE%':>8}{'MAE':>7}{'RMSE':>7}{'sesgo':>7}{'F1':>7}{'err_total%':>12}"
    print(hdr)
    print("-" * len(hdr))
    for name in ("classical", "model"):
        if name not in out:
            continue
        m = out[name]
        print(f"{name:<12}{m['mape_%']:>8}{m['mae']:>7}{m['rmse']:>7}{m['sesgo']:>7}"
              f"{m['f1_localizacion']:>7}{m['error_conteo_total_%']:>12}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--gsd", type=float, default=3.0)
    ap.add_argument("--weights", default=None, help="Pesos YOLOv8-seg (opcional)")
    ap.add_argument("--model-conf", type=float, default=0.6)
    ap.add_argument("--out", default="benchmark_results.json")
    args = ap.parse_args()

    out = run_bench(args.n, args.size, args.gsd, args.weights, args.model_conf)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    _print_table(out)
    print(f"\nJSON: {args.out}")


if __name__ == "__main__":
    main()
