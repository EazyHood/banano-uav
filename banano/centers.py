"""Deteccion de centros por transformada de distancia + watershed, y fusion.

La revision adversarial senalo que la FRST sola (simetria radial) sobre-cuenta
plantas adultas y no tiene base fisica solida en la roseta cenital del banano. El
metodo estandar para "separar objetos convexos que se solapan" es:

    transformada de distancia de la mascara -> maximos locales (marcadores)
    -> watershed.

Los maximos de la transformada de distancia son centros de macolla robustos (uno
por mancha del dosel). La FRST aporta candidatos mas finos (pseudotallos). Se
fusionan ambos conjuntos y se deduplican. Asi, FRST pasa de ser "el detector" a
ser un generador de candidatos, como recomendo la revision.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


def distance_peaks(mask, min_distance, abs_threshold=None):
    """Centros por maximos de la transformada de distancia de la mascara.

    Devuelve (peaks, distance_map, labels) con peaks en (fila, columna) y labels
    la segmentacion watershed (0 = fondo).

    El umbral es ABSOLUTO (en px de la transformada de distancia), no relativo al
    maximo global: asi un blob grande no eleva el umbral y suprime plantas pequenas
    en otras partes del tile. Por defecto ~medio min_distance.
    """
    mask = np.asarray(mask, dtype=bool)
    dt = ndi.distance_transform_edt(mask)
    if dt.max() <= 0:
        return np.empty((0, 2), int), dt, np.zeros(mask.shape, int)

    if abs_threshold is None:
        abs_threshold = max(1.0, 0.7 * float(min_distance))

    peaks = peak_local_max(
        dt,
        min_distance=max(1, int(min_distance)),
        threshold_abs=float(abs_threshold),
        labels=mask,
        exclude_border=False,
    )
    markers = np.zeros(mask.shape, dtype=int)
    for i, (y, x) in enumerate(peaks, start=1):
        markers[y, x] = i
    labels = watershed(-dt, markers, mask=mask)
    return peaks, dt, labels


def merge_candidates(point_arrays, merge_dist):
    """Une varios conjuntos de puntos y deduplica los mas cercanos que merge_dist.

    Da prioridad al orden de entrada (el primer arreglo "gana" en empates).
    """
    arrays = [np.asarray(p, dtype=float).reshape(-1, 2) for p in point_arrays]
    arrays = [a for a in arrays if a.shape[0] > 0]
    if not arrays:
        return np.empty((0, 2), dtype=float)
    pts = np.vstack(arrays)

    kept = []
    for p in pts:
        if all(np.hypot(p[0] - k[0], p[1] - k[1]) > merge_dist for k in kept):
            kept.append(p)
    return np.array(kept, dtype=float)
