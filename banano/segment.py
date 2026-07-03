"""Segmentacion del dosel de banano en imagenes RGB de dron.

Estrategia (no requiere datos etiquetados), endurecida tras revision adversarial:

  1. Correccion de iluminacion (retinex simple): divide por una version muy
     suavizada del brillo para aplanar vinetado y variacion de luz a lo largo del
     vuelo. Evita que Otsu global falle por iluminacion no uniforme.
  2. Mascara de vegetacion: ExGR con umbral ADAPTATIVO (local) + piso global; se
     diagnostica bimodalidad (dosel cerrado o suelo desnudo -> aviso).
  3. Refinamiento por textura a escala de hoja (banano = hojas grandes,
     brillantes, nervadas -> alta desviacion local a esa escala; cesped/maleza
     fina se descartan).
  4. Filtrado morfologico por area y excentricidad (quita ruido puntual y
     estructuras lineales tipo camino).

Es una linea base explicable que funciona sin anotaciones. Para produccion, ver
el paquete ``deep`` (YOLOv8-seg / ALSS-YOLO-Seg).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_local, threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import disk

from .indices import exgr, to_rgb_float


def illumination_correct(rgb, sigma_frac=0.25):
    """Aplana la iluminacion de gran escala (vinetado, hora solar, nubes)."""
    rgb = to_rgb_float(rgb)
    bright = rgb.mean(axis=-1)
    sigma = max(5.0, sigma_frac * min(rgb.shape[:2]))
    background = ndi.gaussian_filter(bright, sigma)
    background = np.clip(background, 1e-3, None)
    gain = float(bright.mean()) / background
    return np.clip(rgb * gain[..., None], 0.0, 1.0)


def _odd(n):
    n = int(n)
    return n if n % 2 == 1 else n + 1


def vegetation_mask(rgb, block_frac=0.12):
    """Mascara de vegetacion con ExGR + umbral local adaptativo + piso global.

    Devuelve (mask, diag). diag incluye la fraccion de vegetacion, el umbral de
    Otsu global y una bandera de bimodalidad (False = histograma unimodal =
    posible fallo silencioso: dosel cerrado o suelo desnudo).
    """
    rgb = to_rgb_float(rgb)
    idx = exgr(rgb)

    # umbral global de referencia + chequeo de bimodalidad
    try:
        t_global = float(threshold_otsu(idx[np.isfinite(idx)]))
    except ValueError:
        t_global = 0.0
    veg_frac_global = float((idx > max(t_global, 0.0)).mean())
    bimodal = 0.03 < veg_frac_global < 0.97

    # umbral local adaptativo (robusto a iluminacion residual)
    block = _odd(max(15, int(block_frac * min(rgb.shape[:2]))))
    try:
        t_local = threshold_local(idx, block_size=block, method="gaussian", offset=0.0)
    except Exception:
        t_local = t_global
    mask = (idx > t_local) & (idx > 0.0)  # nunca por debajo del suelo

    diag = {
        "veg_fraction": float(mask.mean()),
        "otsu_threshold": t_global,
        "bimodal": bool(bimodal),
    }
    return mask, diag


def local_std(gray, size):
    """Desviacion estandar local en ventana cuadrada (separable, rapida)."""
    gray = gray.astype(np.float32)
    size = max(3, int(size))
    mean = ndi.uniform_filter(gray, size)
    sq = ndi.uniform_filter(gray * gray, size)
    var = np.clip(sq - mean * mean, 0.0, None)
    return np.sqrt(var)


def _morph_filter(mask, leaf_scale_px, min_area, max_ecc=0.985):
    """Cierre/apertura + descarta objetos pequenos o muy lineales (excentricos)."""
    footprint = disk(max(1, int(leaf_scale_px // 8)))
    mask = ndi.binary_closing(mask, structure=footprint)
    mask = ndi.binary_opening(mask, structure=disk(max(1, int(leaf_scale_px // 16))))

    lbl = label(mask)
    keep = np.zeros(mask.shape, dtype=bool)
    for r in regionprops(lbl):
        if r.area < min_area:
            continue
        if r.eccentricity > max_ecc and r.area < 8 * min_area:
            continue  # estructura lineal pequena (borde de camino, surco)
        keep[lbl == r.label] = True
    return keep


def banana_canopy_mask(rgb, leaf_scale_px=25, texture_percentile=40, min_mat_area_px=200):
    """Devuelve (mask, diag) del dosel de banano.

    Parametros
    ----------
    leaf_scale_px : ventana (px) al tamano aproximado de una hoja de banano.
    texture_percentile : descarta el N% de vegetacion mas lisa (cesped/maleza).
    min_mat_area_px : area minima de un objeto de banano en px.
    """
    rgb = to_rgb_float(rgb)
    veg, diag = vegetation_mask(rgb)

    gray = rgb.mean(axis=-1)
    std = local_std(gray, leaf_scale_px)
    std_veg = std[veg]
    tex_t = float(np.percentile(std_veg, texture_percentile)) if std_veg.size else 0.0
    textured = std > tex_t

    mask = _morph_filter(veg & textured, leaf_scale_px, int(min_mat_area_px))
    diag["canopy_fraction"] = float(mask.mean())
    return mask, diag
