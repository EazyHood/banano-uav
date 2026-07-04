"""Generador de ortomosaico sintetico de plantacion de banano.

Sirve para (a) demostrar el pipeline de punta a punta y (b) auto-probarlo con
verdad de terreno conocida (posiciones reales de pseudotallos y macollas).

Cada pseudotallo se dibuja como una ROSETA de hojas alargadas que irradian de un
centro (la firma radial que el detector busca). Las macollas tienen 1-3
pseudotallos. Se anaden malezas de textura fina y un camino de suelo para poner
a prueba la segmentacion.
"""

from __future__ import annotations

import numpy as np


def _rng(seed):
    return np.random.default_rng(seed)


def _rosette_patch(P, n_leaves, leaf_len, leaf_w, rng):
    """Genera un parche PxP con alfa [0,1] de una roseta de hojas."""
    yy, xx = np.mgrid[0:P, 0:P].astype(np.float32)
    cy = cx = P / 2.0
    canvas = np.zeros((P, P), np.float32)
    base = rng.uniform(0, 2 * np.pi)
    angles = np.linspace(0, 2 * np.pi, n_leaves, endpoint=False) + base
    for a in angles:
        dx = xx - cx
        dy = yy - cy
        u = dx * np.cos(a) + dy * np.sin(a)  # a lo largo de la hoja
        v = -dx * np.sin(a) + dy * np.cos(a)  # perpendicular
        along = np.exp(-0.5 * ((u - leaf_len * 0.45) / (leaf_len * 0.55)) ** 2)
        across = np.exp(-0.5 * (v / leaf_w) ** 2)
        leaf = along * across * (u > 0)
        canvas = np.maximum(canvas, leaf)
    return np.clip(canvas, 0.0, 1.0)


def _stamp_rosette(img, cy, cx, P, leaf_len, leaf_w, rng, inst_map=None, inst_id=0):
    H, W, _ = img.shape
    n_leaves = int(rng.integers(6, 11))
    canvas = _rosette_patch(P, n_leaves, leaf_len, leaf_w, rng)

    yy, xx = np.mgrid[0:P, 0:P].astype(np.float32)
    center = np.exp(-0.5 * (((yy - P / 2) ** 2 + (xx - P / 2) ** 2) / (0.12 * P) ** 2))
    green = np.array([0.20, 0.55, 0.18], np.float32) + rng.normal(0, 0.03, 3).astype(np.float32)

    y0 = int(round(cy - P / 2))
    x0 = int(round(cx - P / 2))
    y1, x1 = y0 + P, x0 + P
    iy0, ix0 = max(0, y0), max(0, x0)
    iy1, ix1 = min(H, y1), min(W, x1)
    if iy1 <= iy0 or ix1 <= ix0:
        return
    ry0, rx0 = iy0 - y0, ix0 - x0
    a2 = canvas[ry0 : ry0 + (iy1 - iy0), rx0 : rx0 + (ix1 - ix0)]
    a = a2[..., None]
    ctr = center[ry0 : ry0 + (iy1 - iy0), rx0 : rx0 + (ix1 - ix0)][..., None]
    col = green[None, None, :] * (0.8 + 0.4 * a) + ctr * 0.18
    region = img[iy0:iy1, ix0:ix1]
    img[iy0:iy1, ix0:ix1] = (1 - a) * region + a * col
    if inst_map is not None and inst_id:
        sub = inst_map[iy0:iy1, ix0:ix1]
        sub[a2 > 0.15] = inst_id  # asigna estos pixeles a la macolla inst_id


def _add_weeds(img, rng, px_per_m):
    """Malezas: verde de textura FINA (pixeles sueltos) para retar al filtro."""
    H, W, _ = img.shape
    n = int(0.0010 * H * W)
    ys = rng.integers(0, H, n)
    xs = rng.integers(0, W, n)
    green = np.array([0.30, 0.45, 0.20], np.float32)
    img[ys, xs] = np.clip(green + rng.normal(0, 0.05, (n, 3)).astype(np.float32), 0, 1)


def _add_road(img, rng):
    H, W, _ = img.shape
    y = int(rng.integers(int(0.3 * H), int(0.65 * H)))
    w = int(rng.integers(int(0.03 * H), int(0.06 * H)))
    band = np.array([0.5, 0.48, 0.45], np.float32) + rng.normal(0, 0.01, (w, W, 3)).astype(
        np.float32
    )
    img[y : y + w, :] = np.clip(band, 0, 1)


def synth_plantation(
    H=900,
    W=900,
    gsd_cm=3.0,
    spacing_m=2.6,
    seed=7,
    weeds=True,
    road=True,
):
    """Crea una plantacion sintetica de banano.

    Devuelve (imagen_uint8, gt_pseudostems, gt_mats, meta) donde los ground truth
    son arreglos (N, 2) en (fila, columna).
    """
    rng = _rng(seed)
    px_per_m = 100.0 / gsd_cm
    spacing_px = spacing_m * px_per_m

    soil = np.array([0.45, 0.32, 0.22], np.float32)
    img = np.ones((H, W, 3), np.float32) * soil
    img = np.clip(img + rng.normal(0, 0.02, (H, W, 3)).astype(np.float32), 0, 1)

    leaf_len = 0.9 * px_per_m
    leaf_w = 0.12 * px_per_m
    P = int(2.4 * leaf_len)

    gt_pseudo = []
    gt_mats = []

    ys = np.arange(spacing_px, H - spacing_px, spacing_px)
    xs = np.arange(spacing_px, W - spacing_px, spacing_px)
    for gy in ys:
        for gx in xs:
            jy = gy + rng.normal(0, 0.10 * spacing_px)
            jx = gx + rng.normal(0, 0.10 * spacing_px)
            n_ps = int(rng.choice([1, 2, 3], p=[0.25, 0.35, 0.40]))
            members = []
            for k in range(n_ps):
                if k == 0:
                    py, px = jy, jx
                else:
                    a = rng.uniform(0, 2 * np.pi)
                    off = 0.5 * px_per_m
                    py = jy + off * np.sin(a)
                    px = jx + off * np.cos(a)
                _stamp_rosette(img, py, px, P, leaf_len, leaf_w, rng)
                gt_pseudo.append((py, px))
                members.append((py, px))
            gt_mats.append(tuple(np.mean(members, axis=0)))

    if weeds:
        _add_weeds(img, rng, px_per_m)
    if road:
        _add_road(img, rng)

    img = np.clip(img, 0, 1)
    return (
        (img * 255).astype(np.uint8),
        np.array(gt_pseudo, dtype=float),
        np.array(gt_mats, dtype=float),
        {"gsd_cm": gsd_cm, "spacing_m": spacing_m},
    )


def synth_plantation_labeled(
    H=1024, W=1024, gsd_cm=3.0, spacing_m=2.6, seed=7, weeds=True, road=True
):
    """Como synth_plantation pero ademas devuelve un mapa de instancias por macolla.

    Devuelve (img_uint8, instance_map, gt_pseudo, gt_mats, meta) donde instance_map
    es int32 HxW con 0=fondo y 1..N = id de cada macolla. Sirve para generar el
    dataset de entrenamiento de segmentacion de instancias (YOLOv8-seg).
    """
    rng = _rng(seed)
    px_per_m = 100.0 / gsd_cm
    spacing_px = spacing_m * px_per_m

    soil = np.array([0.45, 0.32, 0.22], np.float32)
    img = np.ones((H, W, 3), np.float32) * soil
    img = np.clip(img + rng.normal(0, 0.02, (H, W, 3)).astype(np.float32), 0, 1)
    inst = np.zeros((H, W), np.int32)

    leaf_len = 0.9 * px_per_m
    leaf_w = 0.12 * px_per_m
    P = int(2.4 * leaf_len)

    gt_pseudo, gt_mats = [], []
    ys = np.arange(spacing_px, H - spacing_px, spacing_px)
    xs = np.arange(spacing_px, W - spacing_px, spacing_px)
    mat_id = 0
    for gy in ys:
        for gx in xs:
            jy = gy + rng.normal(0, 0.10 * spacing_px)
            jx = gx + rng.normal(0, 0.10 * spacing_px)
            n_ps = int(rng.choice([1, 2, 3], p=[0.25, 0.35, 0.40]))
            mat_id += 1
            members = []
            for k in range(n_ps):
                if k == 0:
                    py, px = jy, jx
                else:
                    a = rng.uniform(0, 2 * np.pi)
                    off = 0.5 * px_per_m
                    py, px = jy + off * np.sin(a), jx + off * np.cos(a)
                _stamp_rosette(img, py, px, P, leaf_len, leaf_w, rng, inst_map=inst, inst_id=mat_id)
                gt_pseudo.append((py, px))
                members.append((py, px))
            gt_mats.append(tuple(np.mean(members, axis=0)))

    if weeds:
        _add_weeds(img, rng, px_per_m)
    if road:
        _add_road(img, rng)

    img = np.clip(img, 0, 1)
    return (
        (img * 255).astype(np.uint8),
        inst,
        np.array(gt_pseudo, dtype=float),
        np.array(gt_mats, dtype=float),
        {"gsd_cm": gsd_cm, "spacing_m": spacing_m},
    )
