"""Indices de vegetacion a partir de imagenes RGB de dron.

Todos los indices asumen un arreglo HxWx3 en orden RGB. La entrada se normaliza
a [0, 1]. Referencias: Woebbecke 1995 (ExG), Meyer 2008 (ExGR), Gitelson 2002
(VARI), Louhaichi 2001 (GLI), Tucker 1979 (NGRDI).
"""

from __future__ import annotations

import numpy as np


def to_rgb_float(img: np.ndarray) -> np.ndarray:
    """Convierte cualquier imagen a RGB float32 en [0, 1]."""
    img = np.asarray(img)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[-1] == 2:  # gris+alfa u otra 2-banda: usa la 1a como intensidad
        img = np.repeat(img[..., :1], 3, axis=-1)
    if img.shape[-1] > 3:  # descarta canal alfa / bandas extra
        img = img[..., :3]
    img = img.astype(np.float32)
    if img.max() > 1.5:
        img = img / 255.0
    return np.clip(img, 0.0, 1.0)


def _channels(rgb):
    return rgb[..., 0], rgb[..., 1], rgb[..., 2]


def excess_green(rgb):
    """ExG = 2G - R - B. Realza vegetacion frente a suelo."""
    r, g, b = _channels(rgb)
    return 2.0 * g - r - b


def excess_red(rgb):
    """ExR = 1.4R - G."""
    r, g, b = _channels(rgb)
    return 1.4 * r - g


def exgr(rgb):
    """ExGR = ExG - ExR. Mejor separacion vegetacion/suelo que ExG solo."""
    return excess_green(rgb) - excess_red(rgb)


def vari(rgb, eps=1e-6):
    """VARI = (G - R) / (G + R - B). Robusto a iluminacion."""
    r, g, b = _channels(rgb)
    return (g - r) / (g + r - b + eps)


def gli(rgb, eps=1e-6):
    """Green Leaf Index."""
    r, g, b = _channels(rgb)
    return (2.0 * g - r - b) / (2.0 * g + r + b + eps)


def ngrdi(rgb, eps=1e-6):
    """Normalized Green-Red Difference Index."""
    r, g, b = _channels(rgb)
    return (g - r) / (g + r + eps)


def tgi(rgb):
    """Triangular Greenness Index = G - 0.39R - 0.61B.

    Usado por Neupane et al. (2019) para realzar hojas jovenes ricas en clorofila
    antes de detectar banano en ortomosaicos de dron; sube el recall.
    """
    r, g, b = _channels(rgb)
    return g - 0.39 * r - 0.61 * b


INDEX_FUNCS = {
    "exg": excess_green,
    "exgr": exgr,
    "vari": vari,
    "gli": gli,
    "ngrdi": ngrdi,
    "tgi": tgi,
}


def compute(rgb, name="exgr"):
    """Calcula un indice por nombre sobre una imagen RGB (se normaliza sola)."""
    rgb = to_rgb_float(rgb)
    if name not in INDEX_FUNCS:
        raise ValueError(f"Indice desconocido: {name}. Opciones: {list(INDEX_FUNCS)}")
    return INDEX_FUNCS[name](rgb)
