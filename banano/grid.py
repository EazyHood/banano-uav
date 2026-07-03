"""Estimacion del marco de siembra (prior geometrico).

El banano comercial se siembra en marco regular (~2-3 m). Ese patron periodico es
la informacion "gratis" mas potente para el conteo: fija el paso entre plantas y
la orientacion de las hileras de forma robusta, sin datos etiquetados. Con el se
derivan la distancia de supresion de no-maximos y el ``eps`` de agrupamiento, y
sirve como escala fisica cuando NO se conoce el GSD.

Se estima por autocorrelacion 2D (via FFT) de la mascara de dosel: el primer pico
SECUNDARIO (fuera del lobulo central) da el vector al vecino mas cercano en la
reticula. Se excluye un radio central para no confundir el lobulo central (ancho
como una planta) con el paso del marco.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max


def estimate_grid(mask, min_spacing_px=15, max_spacing_frac=0.4):
    """Estima el paso y la orientacion del marco de siembra.

    Devuelve dict con:
      spacing_px      : distancia estimada entre plantas vecinas (px).
      orientation_deg : orientacion del vector al vecino mas cercano (grados).
      strength        : prominencia del pico periodico (0-1); < ~0.05 = poco fiable.
    """
    m = np.asarray(mask, dtype=np.float32)
    none = {"spacing_px": None, "orientation_deg": None, "strength": 0.0}
    if m.sum() == 0:
        return none

    m = m - m.mean()
    F = np.fft.fft2(m)
    ac = np.fft.fftshift(np.fft.ifft2(F * np.conj(F)).real)
    ac = ndi.gaussian_filter(ac, 1.0)

    H, W = ac.shape
    cy, cx = H // 2, W // 2
    center = float(ac[cy, cx]) + 1e-9
    maxr = max_spacing_frac * min(H, W)

    # picos locales (no solo el argmax del anillo: buscamos maximos genuinos)
    peaks = peak_local_max(
        ac,
        min_distance=max(3, int(min_spacing_px // 2)),
        threshold_rel=0.02,
        exclude_border=True,
    )

    cand = []
    for py, px in peaks:
        rr = float(np.hypot(py - cy, px - cx))
        if rr < min_spacing_px or rr > maxr:
            continue
        cand.append((float(ac[py, px]), py, px, rr))

    if not cand:
        return none

    # entre los picos fuertes (>= 60% del mas fuerte), toma el de MENOR radio:
    # el paso fundamental del marco, no un armonico (2x, diagonal, ...).
    vmax = max(c[0] for c in cand)
    strong = [c for c in cand if c[0] >= 0.6 * vmax]
    val, py, px, rr = min(strong, key=lambda c: c[3])
    return {
        "spacing_px": rr,
        "orientation_deg": float(np.degrees(np.arctan2(py - cy, px - cx))),
        "strength": float(np.clip(val / center, 0.0, 1.0)),
    }
