"""Pipeline completo de identificacion de banano en imagenes de dron.

Flujo (endurecido tras revision adversarial del estado del arte):

  1. Correccion de iluminacion.
  2. Mascara de dosel de banano (ExGR adaptativo + textura + morfologia).
  3. Estimacion del marco de siembra por autocorrelacion (prior geometrico).
  4. Deteccion de centros: transformada de distancia + watershed (robusto) FUSIONADO
     con Fast Radial Symmetry Transform (candidatos finos).
  5. Agrupamiento en macollas (DBSCAN) con distancias derivadas del marco/GSD.
  6. Guardarrail de GSD y reporte honesto: rango de pseudotallos + cobertura.

El GSD (cm/pixel) fija los tamanos en metros. Si no se conoce, el marco de siembra
estimado aporta una escala fisica aproximada (asumiendo separacion tipica ~2.5 m).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .centers import distance_peaks, merge_candidates
from .config import PipelineConfig
from .errors import InputError
from .grid import estimate_grid
from .indices import to_rgb_float
from .mats import cluster_mats, planting_regularity
from .radial import detect_centers
from .segment import banana_canopy_mask, illumination_correct

logger = logging.getLogger(__name__)

# Compatibilidad: constantes historicas (ahora viven en config.py como defaults).
ASSUMED_SPACING_M = PipelineConfig.__dataclass_fields__["assumed_spacing_m"].default
LEAF_LEN_M = PipelineConfig.__dataclass_fields__["leaf_len_m"].default
PSEUDOSTEM_SEP_M = PipelineConfig.__dataclass_fields__["pseudostem_sep_m"].default
GRID_MIN_STRENGTH = PipelineConfig.__dataclass_fields__["grid_min_strength"].default


@dataclass
class DetectionResult:
    mask: np.ndarray
    centers: np.ndarray  # (N, 2) fila, columna — candidatos de pseudotallo
    score_map: np.ndarray
    labels: np.ndarray
    mats: list
    grid: dict = field(default_factory=dict)
    regularity: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    params: dict = field(default_factory=dict)

    @property
    def n_pseudostems(self) -> int:
        return int(self.centers.shape[0])

    @property
    def n_mats(self) -> int:
        return len(self.mats)

    @property
    def pseudostem_range(self):
        """Rango honesto: minimo = nº de macollas, maximo = nº de candidatos."""
        return (self.n_mats, self.n_pseudostems)

    def summary(self) -> dict:
        return {
            "n_macollas": self.n_mats,
            "pseudotallos_estimados": self.n_pseudostems,
            "pseudotallos_rango": list(self.pseudostem_range),
            "pseudotallos_por_macolla": round(self.n_pseudostems / self.n_mats, 2)
            if self.n_mats
            else 0.0,
            "cobertura_dosel_%": round(100.0 * float(self.mask.mean()), 1),
            "marco_px": round(self.grid.get("spacing_px") or 0.0, 1),
            "regularidad_siembra": round(self.regularity.get("score", 0.0), 3),
            "avisos": list(self.warnings),
        }


def _radii_from_scale(px_per_m):
    """Radios FRST (px) segun tamanos fisicos de copa (0.4-1.4 m de radio)."""
    if not px_per_m:
        return [8, 14, 22]
    return [max(2, round(r * px_per_m)) for r in (0.4, 0.7, 1.0, 1.4)]


def _validate_image(rgb):
    """Valida y normaliza la imagen de entrada. Lanza InputError si es invalida."""
    if rgb is None:
        raise InputError("La imagen de entrada es None.")
    arr = np.asarray(rgb)
    if arr.size == 0:
        raise InputError("La imagen de entrada esta vacia.")
    if arr.ndim not in (2, 3):
        raise InputError(f"La imagen debe ser 2D o 3D (HxW o HxWxC), no {arr.ndim}D.")
    if arr.ndim == 3 and arr.shape[-1] < 1:
        raise InputError("La imagen 3D no tiene canales.")
    if min(arr.shape[:2]) < 8:
        raise InputError(f"La imagen es demasiado pequena: {arr.shape[:2]} (minimo 8x8).")
    # los valores no-finitos (NaN/inf) se recortan mas adelante en to_rgb_float
    return arr


def detect_banana(
    rgb,
    gsd_cm=None,
    mode="both",
    use_mask=True,
    rel_threshold=0.30,
    use_grid=True,
    config: PipelineConfig | None = None,
):
    """Detecta cultivo de banano en una imagen RGB de dron.

    Parametros
    ----------
    rgb : imagen HxWx3 (uint8 o float).
    gsd_cm : resolucion en cm/pixel. Muy recomendable; sin el se intenta derivar la
        escala del marco de siembra estimado.
    mode : 'bright' | 'dark' | 'both' para la simetria radial.
    use_mask : restringe la deteccion al dosel de banano segmentado.
    rel_threshold : umbral relativo de los picos de simetria (0-1).
    use_grid : usar el marco de siembra estimado para fijar distancias.
    config : PipelineConfig opcional; si se pasa, tiene prioridad sobre los kwargs.

    Lanza
    -----
    InputError : si la imagen es invalida.
    ConfigError : si la configuracion es invalida.
    """
    if config is None:
        config = PipelineConfig(
            gsd_cm=gsd_cm,
            mode=mode,
            use_mask=use_mask,
            rel_threshold=rel_threshold,
            use_grid=use_grid,
        ).validate()

    _validate_image(rgb)
    rgb = to_rgb_float(rgb)
    warnings: list[str] = []
    gsd_cm = config.gsd_cm

    # 1. iluminacion + intensidad
    rgb_corr = illumination_correct(rgb)
    gray = rgb_corr.mean(axis=-1)

    # guardarrail de GSD
    if gsd_cm and gsd_cm > 5:
        warnings.append(
            "GSD > 5 cm/px: el conteo individual NO es fiable. Reporta cobertura de dosel."
        )
    elif gsd_cm and gsd_cm > 3:
        warnings.append("GSD marginal (3-5 cm/px): conteo con incertidumbre alta.")

    px_per_m = (100.0 / gsd_cm) if (gsd_cm and gsd_cm > 0) else None
    leaf_scale_px = (
        max(5, int(config.leaf_len_m * 0.667 * px_per_m))
        if px_per_m
        else max(9, min(gray.shape) // 60)
    )

    # 2. mascara de dosel de banano
    if config.use_mask:
        mask, seg_diag = banana_canopy_mask(rgb_corr, leaf_scale_px=leaf_scale_px)
    else:
        mask, seg_diag = np.ones(gray.shape, bool), {}
    if seg_diag and not seg_diag.get("bimodal", True):
        warnings.append(
            "Histograma de vegetacion unimodal (dosel cerrado o suelo desnudo): "
            "la segmentacion puede ser poco fiable."
        )

    # 3. marco de siembra (prior geometrico)
    grid = (
        estimate_grid(mask, min_spacing_px=max(10, int(1.5 * leaf_scale_px)))
        if config.use_grid
        else {"spacing_px": None, "strength": 0.0}
    )
    grid_strength = float(grid.get("strength") or 0.0)
    grid_spacing = grid.get("spacing_px")
    grid_ok = bool(config.use_grid and grid_strength >= config.grid_min_strength and grid_spacing)

    if px_per_m is None and grid_ok and grid_spacing is not None:
        px_per_m = grid_spacing / config.assumed_spacing_m
        warnings.append(
            f"GSD no provisto: escala fisica estimada del marco de siembra "
            f"(~{grid_spacing:.0f} px = {config.assumed_spacing_m} m)."
        )

    # 4. distancias derivadas
    if px_per_m:
        pseudostem_min_dist = max(3, int(config.pseudostem_sep_m * px_per_m))
        mat_eps_px = max(4, int(1.0 * px_per_m))
    elif grid_ok and grid_spacing is not None:
        pseudostem_min_dist = max(3, int(0.2 * grid_spacing))
        mat_eps_px = max(4, int(0.4 * grid_spacing))
    else:
        pseudostem_min_dist, mat_eps_px = 12, 20
        warnings.append("Sin GSD ni marco fiable: distancias por defecto en px (revisa el conteo).")
    radii = _radii_from_scale(px_per_m)

    # 5. deteccion de centros: FRST (candidatos finos) + distancia/watershed (robusto)
    seg_mask = mask if config.use_mask else None
    frst_peaks, score = detect_centers(
        gray,
        radii,
        mask=seg_mask,
        mode=config.mode,
        min_distance=pseudostem_min_dist,
        rel_threshold=config.rel_threshold,
    )
    dt_peaks, _dt, _wlabels = distance_peaks(mask, min_distance=pseudostem_min_dist)
    centers = merge_candidates([dt_peaks, frst_peaks], merge_dist=0.8 * pseudostem_min_dist)

    # 6. agrupamiento en macollas
    labels, mats = cluster_mats(centers, mat_eps_px)
    mat_centroids = np.array([m["centroid"] for m in mats]) if mats else np.empty((0, 2))
    reg = planting_regularity(mat_centroids)

    logger.debug(
        "detect_banana: %d macollas, %d pseudotallos (gsd=%s, grid_used=%s)",
        len(mats),
        int(centers.shape[0]),
        gsd_cm,
        bool(grid_ok),
    )

    return DetectionResult(
        mask=mask,
        centers=centers,
        score_map=score,
        labels=labels,
        mats=mats,
        grid=grid,
        regularity=reg,
        warnings=warnings,
        params={
            "gsd_cm": gsd_cm,
            "px_per_m": round(px_per_m, 2) if px_per_m else None,
            "leaf_scale_px": int(leaf_scale_px),
            "pseudostem_min_dist": int(pseudostem_min_dist),
            "mat_eps_px": int(mat_eps_px),
            "radii": [int(r) for r in radii],
            "mode": config.mode,
            "rel_threshold": float(config.rel_threshold),
            "grid_used": bool(grid_ok),
        },
    )
