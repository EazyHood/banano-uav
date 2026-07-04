"""Pruebas unitarias de los modulos base."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from banano import indices
from banano.centers import distance_peaks, merge_candidates
from banano.grid import estimate_grid
from banano.mats import cluster_mats, planting_regularity
from banano.metrics import match_points, prf
from banano.radial import frst
from banano.segment import illumination_correct, vegetation_mask


def _green_img():
    img = np.zeros((64, 64, 3), np.uint8)
    img[..., 0] = 40  # R
    img[..., 1] = 180  # G (verde)
    img[..., 2] = 30  # B
    return img


def test_indices_shapes_and_range():
    img = _green_img()
    for name in indices.INDEX_FUNCS:
        v = indices.compute(img, name)
        assert v.shape == (64, 64)
        assert np.all(np.isfinite(v))


def test_to_rgb_float_channels():
    assert indices.to_rgb_float(np.zeros((8, 8))).shape == (8, 8, 3)  # 2D
    assert indices.to_rgb_float(np.zeros((8, 8, 2), np.uint8)).shape == (8, 8, 3)
    assert indices.to_rgb_float(np.zeros((8, 8, 4), np.uint8)).shape == (8, 8, 3)
    out = indices.to_rgb_float((np.ones((4, 4, 3)) * 255).astype(np.uint8))
    assert out.max() <= 1.0


def test_vegetation_mask_detects_green():
    img = np.zeros((40, 40, 3), np.uint8)
    img[10:30, 10:30] = [40, 200, 30]  # parche verde sobre negro
    mask, diag = vegetation_mask(img)
    assert mask[20, 20]  # el centro verde es vegetacion
    assert "veg_fraction" in diag


def test_illumination_correct_preserves_shape():
    img = _green_img()
    out = illumination_correct(img)
    assert out.shape == (64, 64, 3)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_frst_zero_on_flat():
    flat = np.ones((50, 50), np.float32) * 0.5
    s = frst(flat, radii=[5, 8])
    assert s.shape == (50, 50)
    assert float(s.max()) == 0.0  # sin gradientes -> sin respuesta


def test_distance_peaks_on_blobs():
    mask = np.zeros((80, 80), bool)
    mask[20:30, 20:30] = True
    mask[50:60, 50:60] = True
    peaks, dt, labels = distance_peaks(mask, min_distance=5)
    assert len(peaks) >= 2


def test_merge_candidates_dedup():
    a = np.array([[10.0, 10.0], [10.5, 10.5]])  # muy cerca -> se funden
    b = np.array([[50.0, 50.0]])
    merged = merge_candidates([a, b], merge_dist=3.0)
    assert len(merged) == 2


def test_cluster_mats_groups():
    pts = np.array([[10, 10], [12, 11], [60, 60]], float)  # 2 juntos + 1 lejos
    labels, mats = cluster_mats(pts, mat_radius_px=5)
    assert len(mats) == 2


def test_planting_regularity_grid():
    xs, ys = np.meshgrid(np.arange(0, 100, 10), np.arange(0, 100, 10))
    pts = np.stack([ys.ravel(), xs.ravel()], 1).astype(float)
    reg = planting_regularity(pts)
    assert reg["score"] > 0.8  # malla perfectamente regular


def test_estimate_grid_on_regular():
    mask = np.zeros((300, 300), bool)
    for y in range(30, 300, 40):
        for x in range(30, 300, 40):
            mask[y - 3:y + 3, x - 3:x + 3] = True
    g = estimate_grid(mask, min_spacing_px=15)
    assert g["spacing_px"] is not None
    assert abs(g["spacing_px"] - 40) < 8  # deberia recuperar ~40 px


def test_metrics_perfect_and_empty():
    gt = np.array([[10, 10], [20, 20]], float)
    tp, fp, fn, _ = match_points(gt, gt, 2)
    assert (tp, fp, fn) == (2, 0, 0)
    p, r, f = prf(2, 0, 0)
    assert (p, r, f) == (1.0, 1.0, 1.0)
    assert match_points(np.empty((0, 2)), gt, 2) == (0, 0, 2, [])
