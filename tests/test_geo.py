"""Pruebas del soporte geoespacial y de casos límite corregidos tras la revisión.

Las que necesitan rasterio se saltan solas si el extra [geo] no está instalado.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from banano.geo import Raster, _normalize_rgb  # noqa: E402
from banano.ortho import _tile_starts, process_orthomosaic  # noqa: E402
from banano.synth import synth_plantation  # noqa: E402

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402


def _write_tif(path, data_bands, crs, transform):
    n = data_bands.shape[0]
    with rasterio.open(
        path, "w", driver="GTiff", height=data_bands.shape[1], width=data_bands.shape[2],
        count=n, dtype="uint8", crs=crs, transform=transform,
    ) as ds:
        ds.write(data_bands)


def test_gsd_geographic_only_scales_longitude(tmp_path):
    # A lat 40, un píxel de 0.03 m equiv. en grados: el GSD promedio correcto usa
    # cos(lat) SOLO en longitud.
    lat, deg, n = 40.0, 0.03 / 111320.0, 120
    path = os.path.join(str(tmp_path), "geo4326.tif")
    _write_tif(path, np.zeros((3, n, n), np.uint8) + 30, "EPSG:4326",
               from_origin(-74.0, lat + n * deg, deg, deg))
    r = Raster(path)
    expected = 0.5 * (0.03 * math.cos(math.radians(lat)) + 0.03) * 100
    assert abs(r.gsd_cm - expected) < 0.05, (r.gsd_cm, expected)
    r.close()


def test_two_band_raster_does_not_crash(tmp_path):
    img, _, _, _ = synth_plantation(H=300, W=300, gsd_cm=3.0, seed=3)
    gray = img.mean(-1).astype(np.uint8)
    two = np.stack([gray, np.full_like(gray, 255)], 0)  # gris + alfa
    path = os.path.join(str(tmp_path), "two.tif")
    _write_tif(path, two, "EPSG:32618", from_origin(5e5, 5e5, 0.03, 0.03))
    r = Raster(path)
    assert r.read_window(0, 0, 100, 100).shape[-1] == 3
    res = process_orthomosaic(r, tile=256, overlap=64)  # no debe lanzar
    assert res.n_mats >= 0
    r.close()


def test_projected_gsd_auto(tmp_path):
    img, _, _, _ = synth_plantation(H=300, W=300, gsd_cm=3.0, seed=1)
    path = os.path.join(str(tmp_path), "utm.tif")
    _write_tif(path, np.moveaxis(img, -1, 0), "EPSG:32618", from_origin(5e5, 5e5, 0.03, 0.03))
    r = Raster(path)
    assert abs(r.gsd_cm - 3.0) < 0.01  # 0.03 m/px -> 3 cm/px
    assert r.has_geo
    r.close()


def test_normalize_rgb_channels_and_dtype():
    assert _normalize_rgb(np.zeros((4, 4), np.uint8)).shape == (4, 4, 3)          # 2D
    assert _normalize_rgb(np.zeros((4, 4, 2), np.uint8)).shape == (4, 4, 3)       # gris+alfa
    assert _normalize_rgb(np.zeros((4, 4, 5), np.uint8)).shape == (4, 4, 3)       # >3 bandas
    u16 = (np.ones((4, 4, 3), np.uint16) * 512)
    assert _normalize_rgb(u16).dtype == np.uint8 and _normalize_rgb(u16)[0, 0, 0] == 2


def test_tile_starts_no_redundant_final():
    # El último tile llega al borde y no hay una tesela redundante extra.
    starts = _tile_starts(1000, 400, 300)  # 0, 300, 600 (600+400>=1000 -> para)
    assert starts == [0, 300, 600]
    # caso donde un tile completo justo alcanza el borde
    assert _tile_starts(800, 400, 400) == [0, 400]
