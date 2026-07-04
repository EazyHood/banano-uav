"""Demo de punta a punta sobre una plantacion sintetica.

Genera una imagen con verdad de terreno conocida, corre el pipeline y evalua
precision/recall tanto a nivel de pseudotallo como de macolla. Guarda las
imagenes de entrada y resultado en ../outputs.

Uso:
    python scripts/demo.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import imageio.v2 as imageio  # noqa: E402

from banano.metrics import match_points, prf  # noqa: E402
from banano.pipeline import detect_banana  # noqa: E402
from banano.synth import synth_plantation  # noqa: E402
from banano.visualize import overlay, save_score_map  # noqa: E402


def _np_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"No serializable: {type(o)}")


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outdir = os.path.join(root, "outputs")
    os.makedirs(outdir, exist_ok=True)

    gsd = 3.0  # cm/pixel
    img, gt_ps, gt_mats, meta = synth_plantation(H=900, W=900, gsd_cm=gsd, seed=7)
    imageio.imwrite(os.path.join(outdir, "synthetic_input.png"), img)

    res = detect_banana(img, gsd_cm=gsd)
    overlay(img, res, out_path=os.path.join(outdir, "detection_overlay.png"))
    save_score_map(res, os.path.join(outdir, "radial_score.png"))

    tol_ps = res.params["pseudostem_min_dist"]
    tol_mat = res.params["mat_eps_px"]
    tp, fp, fn, _ = match_points(res.centers, gt_ps, tol_ps)
    p, r, f = prf(tp, fp, fn)

    mat_centroids = (
        np.array([m["centroid"] for m in res.mats]) if res.mats else np.empty((0, 2))
    )
    tpm, fpm, fnm, _ = match_points(mat_centroids, gt_mats, tol_mat)
    pm, rm, fm = prf(tpm, fpm, fnm)

    report = {
        "verdad_terreno": {
            "pseudotallos": int(len(gt_ps)),
            "macollas": int(len(gt_mats)),
        },
        "detectado": {
            "pseudotallos": res.n_pseudostems,
            "macollas": res.n_mats,
        },
        "pseudotallos": {
            "precision": round(p, 3),
            "recall": round(r, 3),
            "f1": round(f, 3),
        },
        "macollas": {
            "precision": round(pm, 3),
            "recall": round(rm, 3),
            "f1": round(fm, 3),
        },
        "resumen": res.summary(),
        "params": {
            k: (list(np.round(v, 1)) if isinstance(v, (list, np.ndarray)) else v)
            for k, v in res.params.items()
        },
    }

    with open(os.path.join(outdir, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=_np_default)

    print(json.dumps(report, indent=2, ensure_ascii=False, default=_np_default))
    print(f"\nImagenes guardadas en: {outdir}")


if __name__ == "__main__":
    main()
