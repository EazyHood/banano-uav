"""Agrupamiento de centros de pseudotallo en macollas (mats) de banano.

Una macolla es una planta madre con sus hijuelos: normalmente 1-3 pseudotallos
muy juntos. Agrupamos los centros detectados por proximidad para reportar el
conteo en la unidad agronomica correcta. Asi el usuario obtiene DOS numeros:
  - n_pseudostems: pseudotallos individuales (por simetria radial).
  - n_mats: macollas (grupos), la unidad de siembra/manejo.
"""

from __future__ import annotations

import numpy as np

try:
    from sklearn.cluster import DBSCAN

    _HAS_SK = True
except Exception:  # pragma: no cover
    _HAS_SK = False


def _greedy_cluster(centers, radius):
    """Agrupamiento aglomerativo simple si no hay scikit-learn."""
    n = centers.shape[0]
    labels = -np.ones(n, dtype=int)
    cur = 0
    for i in range(n):
        if labels[i] != -1:
            continue
        labels[i] = cur
        d = np.linalg.norm(centers - centers[i], axis=1)
        labels[(d <= radius) & (labels == -1)] = cur
        cur += 1
    return labels


def cluster_mats(centers, mat_radius_px):
    """Agrupa centros (N,2 en fila,columna) en macollas.

    Devuelve (labels, mats) donde mats es una lista de dicts con centroide,
    numero de pseudotallos y miembros.
    """
    centers = np.asarray(centers, dtype=float).reshape(-1, 2)
    if centers.shape[0] == 0:
        return np.array([], dtype=int), []

    eps = max(1.0, float(mat_radius_px))
    if _HAS_SK:
        labels = DBSCAN(eps=eps, min_samples=1).fit_predict(centers)
    else:
        labels = _greedy_cluster(centers, eps)

    mats = []
    for lb in np.unique(labels):
        members = centers[labels == lb]
        mats.append(
            {
                "label": int(lb),
                "centroid": members.mean(axis=0),
                "n_pseudostems": int(members.shape[0]),
                "members": members,
            }
        )
    return labels, mats


def planting_regularity(mat_centroids):
    """Mide que tan regular es la malla de siembra (0-1, mayor = mas regular).

    Plantaciones comerciales de banano se siembran en malla regular; una alta
    regularidad es evidencia adicional de que el area ES un cultivo de banano.
    Se basa en la variabilidad de la distancia al vecino mas cercano.
    """
    c = np.asarray(mat_centroids, dtype=float).reshape(-1, 2)
    if c.shape[0] < 3:
        return {"score": 0.0, "median_spacing_px": None, "cv": None}
    nn = []
    for i in range(c.shape[0]):
        d = np.linalg.norm(c - c[i], axis=1)
        d[i] = np.inf
        nn.append(d.min())
    nn = np.asarray(nn)
    median = float(np.median(nn))
    cv = float(np.std(nn) / (np.mean(nn) + 1e-9))  # coef. de variacion
    score = float(np.clip(1.0 - cv, 0.0, 1.0))
    return {"score": score, "median_spacing_px": median, "cv": cv}
