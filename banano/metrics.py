"""Metricas de deteccion por emparejamiento de puntos.

Compara centros detectados contra posiciones verdaderas (ground truth) usando
emparejamiento voraz por vecino mas cercano con una tolerancia en pixeles.
"""

from __future__ import annotations

import numpy as np


def match_points(pred, gt, tol):
    """Empareja predicciones con ground truth dentro de una tolerancia (px).

    Devuelve (tp, fp, fn, matches) con matches como lista de pares (i_pred, j_gt).
    """
    pred = np.asarray(pred, dtype=float).reshape(-1, 2)
    gt = np.asarray(gt, dtype=float).reshape(-1, 2)

    if len(gt) == 0:
        return 0, len(pred), 0, []
    if len(pred) == 0:
        return 0, 0, len(gt), []

    used = np.zeros(len(gt), dtype=bool)
    tp = 0
    matches = []
    for i, p in enumerate(pred):
        d = np.linalg.norm(gt - p, axis=1)
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= tol:
            used[j] = True
            tp += 1
            matches.append((i, j))
    fp = len(pred) - tp
    fn = int((~used).sum())
    return tp, fp, fn, matches


def prf(tp, fp, fn):
    """Precision, recall y F1."""
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f
