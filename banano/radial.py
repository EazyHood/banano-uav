"""Deteccion de centros de pseudotallo por simetria radial.

Idea clave del proyecto: vista desde arriba (nadir), una planta de banano es una
ROSETA: las hojas irradian desde un punto central y sus nervaduras/bordes apuntan
hacia (o desde) ese centro. Aunque tres pseudotallos crezcan juntos en una macolla
y sus doseles se solapen, cada uno mantiene su propio centro de simetria radial.

Se usa la Fast Radial Symmetry Transform (FRST; Loy & Zelinsky, 2003) para votar
por esos centros a varios radios, y luego supresion de no-maximos para extraer los
picos. Esto localiza pseudotallos individuales incluso dentro de un grupo, algo que
la deteccion de copas por "manchas" no logra.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max


def _gradients(gray):
    gx = ndi.sobel(gray.astype(np.float32), axis=1, mode="reflect")
    gy = ndi.sobel(gray.astype(np.float32), axis=0, mode="reflect")
    mag = np.hypot(gx, gy)
    return gx, gy, mag


def frst(gray, radii, alpha=2.0, beta=0.25, mode="both", grad_frac=0.15):
    """Fast Radial Symmetry Transform.

    Parametros
    ----------
    gray : imagen 2D (se usa el canal de intensidad).
    radii : iterable de radios en px a los que buscar simetria.
    alpha : exponente de radialidad (mayor = mas estricto). 2 es tipico.
    beta : factor de suavizado gaussiano relativo al radio.
    mode : 'bright' (centro claro), 'dark' (centro oscuro) o 'both'.
    grad_frac : umbral de magnitud de gradiente como fraccion del maximo.

    Devuelve un mapa de respuesta S (mismo tamano que gray); picos = centros.
    """
    gray = np.asarray(gray, dtype=np.float32)
    H, W = gray.shape
    gx, gy, mag = _gradients(gray)

    mmax = float(mag.max())
    if mmax <= 0:
        return np.zeros((H, W), np.float32)
    thresh = grad_frac * mmax

    ys, xs = np.nonzero(mag > thresh)
    if ys.size == 0:
        return np.zeros((H, W), np.float32)
    gmag = mag[ys, xs]
    ux = gx[ys, xs] / gmag
    uy = gy[ys, xs] / gmag

    radii = [int(round(r)) for r in radii if r >= 1]
    S = np.zeros((H, W), np.float32)

    for n in radii:
        On = np.zeros((H, W), np.float32)
        Mn = np.zeros((H, W), np.float32)

        pos_x = np.clip(xs + np.round(ux * n).astype(np.int64), 0, W - 1)
        pos_y = np.clip(ys + np.round(uy * n).astype(np.int64), 0, H - 1)
        neg_x = np.clip(xs - np.round(ux * n).astype(np.int64), 0, W - 1)
        neg_y = np.clip(ys - np.round(uy * n).astype(np.int64), 0, H - 1)

        if mode in ("bright", "both"):
            np.add.at(On, (pos_y, pos_x), 1.0)
            np.add.at(Mn, (pos_y, pos_x), gmag)
        if mode in ("dark", "both"):
            np.add.at(On, (neg_y, neg_x), -1.0)
            np.add.at(Mn, (neg_y, neg_x), -gmag)

        kappa = 9.9 if n > 1 else 8.0
        On = np.clip(np.abs(On), None, kappa)
        Mn = np.abs(Mn)
        Fn = (On / kappa) ** alpha * (Mn / kappa)
        S += ndi.gaussian_filter(Fn, sigma=max(0.5, beta * n))

    return S / max(1, len(radii))


def detect_centers(
    gray,
    radii,
    mask=None,
    mode="both",
    min_distance=None,
    rel_threshold=0.20,
):
    """Detecta centros de pseudotallo.

    Devuelve (centers, score_map) donde centers es (N, 2) en (fila, columna).
    """
    S = frst(gray, radii, mode=mode)
    if mask is not None:
        S = S * mask.astype(np.float32)

    if S.max() <= 0:
        return np.empty((0, 2), dtype=int), S

    if min_distance is None:
        min_distance = max(1, int(min(radii)))

    # Umbral robusto: relativo a un percentil alto (no al maximo global), para que
    # un solo punto muy brillante no suba el umbral y suprima plantas mas debiles.
    # Esto hace la deteccion estable al tamano del tile.
    pos = S[S > 0]
    ref = float(np.percentile(pos, 99)) if pos.size else float(S.max())
    peaks = peak_local_max(
        S,
        min_distance=int(min_distance),
        threshold_abs=float(rel_threshold) * ref,
        exclude_border=False,
    )
    return peaks, S
