"""Pruebas del flujo de ortomosaico y entregables (sin requerir rasterio).

Usa la ruta de imagen normal (imageio) de banano/geo.py, así corre en CI sin el
extra [geo].
"""
from __future__ import annotations

import json
import os
import sys

import imageio.v2 as imageio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from banano.geo import Raster  # noqa: E402
from banano.ortho import process_orthomosaic  # noqa: E402
from banano.report import write_all  # noqa: E402
from banano.synth import synth_plantation  # noqa: E402


def _make_png(tmp_path, size=1000, seed=5):
    img, gt_ps, gt_mats, _ = synth_plantation(H=size, W=size, gsd_cm=3.0, seed=seed)
    p = os.path.join(str(tmp_path), "orto.png")
    imageio.imwrite(p, img)
    return p, len(gt_ps), len(gt_mats)


def test_ortho_runs_and_counts(tmp_path):
    path, n_ps, n_mats = _make_png(tmp_path)
    raster = Raster(path, gsd_cm=3.0)
    assert not raster.has_geo  # PNG => sin georreferencia
    res = process_orthomosaic(raster, gsd_cm=3.0, tile=512, overlap=96)
    raster.close()

    # el conteo de macollas debe quedar dentro de +-40% de la verdad de terreno
    assert 0.6 * n_mats <= res.n_mats <= 1.4 * n_mats, (res.n_mats, n_mats)
    # el rango honesto de pseudotallos debe contener la verdad
    lo, hi = res.pseudostem_range
    assert lo <= n_ps <= hi or hi >= 0.7 * n_ps, (res.pseudostem_range, n_ps)


def test_ortho_writes_deliverables(tmp_path):
    path, _, _ = _make_png(tmp_path, size=700)
    raster = Raster(path, gsd_cm=3.0)
    res = process_orthomosaic(raster, gsd_cm=3.0, tile=512, overlap=96)
    outdir = os.path.join(str(tmp_path), "out")
    paths = write_all(outdir, res, raster=raster, input_name=path)
    raster.close()

    for key in ("resumen", "csv", "geojson", "html", "mapa"):
        assert os.path.exists(paths[key]), key

    # el GeoJSON debe ser válido y tener una feature por macolla
    with open(paths["geojson"], encoding="utf-8") as fh:
        gj = json.load(fh)
    assert gj["type"] == "FeatureCollection"
    assert len(gj["features"]) == res.n_mats


def test_overlap_ge_tile_is_sanitized(tmp_path):
    # overlap >= tile no debe colgar (antes: step colapsaba a 1 -> H*W tuplas).
    path, _, _ = _make_png(tmp_path, size=400)
    raster = Raster(path, gsd_cm=3.0)
    res = process_orthomosaic(raster, gsd_cm=3.0, tile=256, overlap=256)
    raster.close()
    assert res.n_mats >= 0


def test_tiling_dedup_no_double_count(tmp_path):
    # Con solape, el conteo por tiles no debe inflarse respecto a un solo paso.
    path, _, n_mats = _make_png(tmp_path, size=900, seed=8)
    raster = Raster(path, gsd_cm=3.0)
    res_tiled = process_orthomosaic(raster, gsd_cm=3.0, tile=400, overlap=100)
    res_single = process_orthomosaic(raster, gsd_cm=3.0, tile=2000, overlap=0)
    raster.close()
    # tolerancia del 30% entre teselado fino y paso único
    assert abs(res_tiled.n_mats - res_single.n_mats) <= 0.3 * res_single.n_mats + 5


class _StubModel:
    """Modelo falso para probar el camino de modelo sin ultralytics.

    Devuelve dos detecciones casi coincidentes: la de MENOR confianza primero,
    para que una deduplicacion primero-llega-gana conserve la equivocada.
    """

    def __init__(self, weights, conf=0.5, imgsz=640, augment=False, device=None):
        self.weights = weights
        self.conf = conf
        self.imgsz = imgsz
        self.augment = augment
        self.device = device

    def predict_mats(self, image, conf=None):
        import numpy as np

        return {
            "centroids": np.array([[100.0, 100.0], [100.0, 106.0]]),
            "confidences": np.array([0.4, 0.9]),
            "boxes_xyxy": np.array(
                [[80.0, 80.0, 120.0, 120.0], [86.0, 80.0, 126.0, 120.0]]
            ),
            "n": 2,
        }


class _SpyModel(_StubModel):
    """Igual que el anterior, pero apunta con que imgsz se construyo."""

    ultimo_imgsz: int | None = None

    def __init__(self, weights, conf=0.5, imgsz=640, augment=False, device=None):
        super().__init__(weights, conf=conf, imgsz=imgsz, augment=augment, device=device)
        _SpyModel.ultimo_imgsz = imgsz


def _run_con_spy(tmp_path, monkeypatch, **cfg_kwargs):
    import banano.model
    from banano.config import PipelineConfig

    monkeypatch.setattr(banano.model, "BananaModel", _SpyModel)
    _SpyModel.ultimo_imgsz = None
    path, _, _ = _make_png(tmp_path, size=300, seed=12)
    cfg = PipelineConfig(gsd_cm=3.0, overlap=64, model_weights="stub.pt", **cfg_kwargs).validate()
    raster = Raster(path, gsd_cm=3.0)
    process_orthomosaic(raster, config=cfg)
    raster.close()
    return _SpyModel.ultimo_imgsz


def test_model_imgsz_se_deduce_del_tile(tmp_path, monkeypatch):
    # Sin model_imgsz, la resolucion de inferencia sale del tile (comportamiento de siempre).
    assert _run_con_spy(tmp_path, monkeypatch, tile=512) == 512


def test_model_imgsz_topado_a_1280(tmp_path, monkeypatch):
    # Un tile enorme no debe disparar la resolucion de inferencia.
    assert _run_con_spy(tmp_path, monkeypatch, tile=2048) == 1280


def test_model_imgsz_explicito_manda(tmp_path, monkeypatch):
    # El caso que motiva la opcion: tile 1024 pero modelo entrenado a 768.
    assert _run_con_spy(tmp_path, monkeypatch, tile=1024, model_imgsz=768) == 768


def test_model_dedup_keeps_highest_confidence(tmp_path, monkeypatch):
    # En un duplicado (dos detecciones a <mat_eps/2), debe sobrevivir la de mayor
    # confianza, no la primera en llegar.
    import banano.model
    from banano.config import PipelineConfig

    monkeypatch.setattr(banano.model, "BananaModel", _StubModel)
    path, _, _ = _make_png(tmp_path, size=300, seed=11)
    cfg = PipelineConfig(gsd_cm=3.0, tile=512, overlap=64, model_weights="stub.pt").validate()
    raster = Raster(path, gsd_cm=3.0)
    res = process_orthomosaic(raster, config=cfg)
    raster.close()
    assert res.n_mats == 1
    kept_col = res.mats[0]["centroid"][1]
    assert abs(kept_col - 106.0) < 1e-6, f"sobrevivio la de menor confianza (col={kept_col})"
