"""Pruebas del camino de deep learning.

Se saltan solas si falta ultralytics o los pesos del modelo (no bloquean el CI
base, que no instala el extra [deep]).
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("ultralytics")

WEIGHTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "banano_seg_synth_v1.pt",
)
pytestmark = pytest.mark.skipif(not os.path.exists(WEIGHTS), reason="pesos no disponibles")

from banano.model import BananaModel  # noqa: E402
from banano.synth import synth_plantation  # noqa: E402

REAL_WEIGHTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "banano_real_v1.pt",
)


@pytest.mark.skipif(not os.path.exists(REAL_WEIGHTS), reason="modelo real no disponible")
def test_real_model_loads_and_runs():
    # el modelo de DETECCION real debe cargar y devolver centroides validos
    m = BananaModel(REAL_WEIGHTS, conf=0.35, imgsz=640)
    img, _, _, _ = synth_plantation(H=640, W=640, gsd_cm=3.0, seed=1234)
    pred = m.predict_mats(img)
    assert pred["centroids"].ndim == 2 and pred["centroids"].shape[1] == 2
    assert pred["n"] == len(pred["centroids"]) >= 0


def test_model_predict_mats():
    m = BananaModel(WEIGHTS, conf=0.5, imgsz=640)
    img, _, gt_mats, _ = synth_plantation(H=640, W=640, gsd_cm=3.0, seed=770001)
    pred = m.predict_mats(img)
    assert pred["centroids"].shape[1] == 2
    assert pred["n"] == len(pred["centroids"])
    # el conteo debe estar en un rango razonable de la verdad de terreno
    assert 0.5 * len(gt_mats) <= pred["n"] <= 1.6 * len(gt_mats), (pred["n"], len(gt_mats))


def test_model_ortho_path(tmp_path):
    import imageio.v2 as imageio

    from banano.config import PipelineConfig
    from banano.geo import Raster
    from banano.ortho import process_orthomosaic

    img, _, gt_mats, _ = synth_plantation(H=900, W=900, gsd_cm=3.0, seed=770002)
    p = os.path.join(str(tmp_path), "o.png")
    imageio.imwrite(p, img)
    cfg = PipelineConfig(gsd_cm=3.0, tile=640, overlap=96,
                         model_weights=WEIGHTS, model_conf=0.5).validate()
    raster = Raster(p, gsd_cm=3.0)
    res = process_orthomosaic(raster, config=cfg)
    raster.close()
    assert res.n_mats > 0
    assert 0.5 * len(gt_mats) <= res.n_mats <= 1.6 * len(gt_mats), (res.n_mats, len(gt_mats))
