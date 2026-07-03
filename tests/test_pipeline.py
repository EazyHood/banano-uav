"""Pruebas basicas: el pipeline detecta banano sintetico razonablemente bien."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from banano.metrics import match_points, prf  # noqa: E402
from banano.pipeline import detect_banana  # noqa: E402
from banano.synth import synth_plantation  # noqa: E402


def _run(seed):
    gsd = 3.0
    img, gt_ps, gt_mats, _ = synth_plantation(H=700, W=700, gsd_cm=gsd, seed=seed)
    res = detect_banana(img, gsd_cm=gsd)
    return res, gt_ps, gt_mats


def test_detects_something():
    res, gt_ps, gt_mats = _run(1)
    assert res.n_pseudostems > 0
    assert res.n_mats > 0


def test_mat_count_in_range():
    # El conteo de macollas debe quedar dentro de +-50% de la verdad de terreno.
    res, gt_ps, gt_mats = _run(2)
    lo, hi = 0.5 * len(gt_mats), 1.5 * len(gt_mats)
    assert lo <= res.n_mats <= hi, (res.n_mats, len(gt_mats))


def test_pseudostem_recall():
    # Recall de pseudotallos aceptable en imagen limpia.
    res, gt_ps, gt_mats = _run(3)
    tol = res.params["mat_eps_px"]
    tp, fp, fn, _ = match_points(res.centers, gt_ps, tol)
    _, recall, _ = prf(tp, fp, fn)
    assert recall >= 0.4, recall


def test_mask_covers_vegetation():
    res, _, _ = _run(4)
    cover = float(res.mask.mean())
    # El dosel debe cubrir una fraccion no trivial pero no toda la imagen.
    assert 0.05 < cover < 0.95, cover


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("OK", name)
