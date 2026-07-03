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

from dataclasses import dataclass, field

import numpy as np

from .centers import distance_peaks, merge_candidates
from .grid import estimate_grid
from .indices import to_rgb_float
from .mats import cluster_mats, planting_regularity
from .radial import detect_centers
from .segment import banana_canopy_mask, illumination_correct

# Supuestos agronomicos por defecto (banano comercial adulto).
ASSUMED_SPACING_M = 2.5   # separacion tipica entre plantas
LEAF_LEN_M = 0.9          # largo de hoja
PSEUDOSTEM_SEP_M = 0.5    # separacion entre pseudotallos dentro de una macolla
GRID_MIN_STRENGTH = 0.05  # por debajo, el marco estimado no es fiable


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


def detect_banana(
    rgb,
    gsd_cm=None,
    mode="both",
    use_mask=True,
    rel_threshold=0.30,
    use_grid=True,
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
    """
    rgb = to_rgb_float(rgb)
    warnings = []

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

    # escala inicial para segmentacion
    px_per_m = (100.0 / gsd_cm) if (gsd_cm and gsd_cm > 0) else None
    leaf_scale_px = (
        max(5, int(0.6 * px_per_m)) if px_per_m else max(9, min(gray.shape) // 60)
    )

    # 2. mascara de dosel de banano
    if use_mask:
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
        if use_grid
        else {"spacing_px": None, "strength": 0.0}
    )
    grid_ok = use_grid and grid.get("strength", 0.0) >= GRID_MIN_STRENGTH and grid.get("spacing_px")

    # si no hay GSD pero hay marco fiable, deriva la escala fisica del marco
    if px_per_m is None and grid_ok:
        px_per_m = grid["spacing_px"] / ASSUMED_SPACING_M
        warnings.append(
            f"GSD no provisto: escala fisica estimada del marco de siembra "
            f"(~{grid['spacing_px']:.0f} px = {ASSUMED_SPACING_M} m)."
        )

    # 4. distancias derivadas
    #    - min_dist entre pseudotallos ~ separacion dentro de la macolla (0.5 m)
    #    - eps agrupa pseudotallos de una misma macolla (~0.75 m) sin encadenar
    #      macollas vecinas del marco (~2.5 m)
    if px_per_m:
        pseudostem_min_dist = max(3, int(PSEUDOSTEM_SEP_M * px_per_m))
        mat_eps_px = max(4, int(1.0 * px_per_m))
    elif grid_ok:
        pseudostem_min_dist = max(3, int(0.2 * grid["spacing_px"]))
        mat_eps_px = max(4, int(0.4 * grid["spacing_px"]))
    else:
        pseudostem_min_dist, mat_eps_px = 12, 20
        warnings.append(
            "Sin GSD ni marco fiable: distancias por defecto en px (revisa el conteo)."
        )
    radii = _radii_from_scale(px_per_m)

    # 5. deteccion de centros: FRST (candidatos finos) + distancia/watershed (robusto)
    seg_mask = mask if use_mask else None
    frst_peaks, score = detect_centers(
        gray, radii, mask=seg_mask, mode=mode,
        min_distance=pseudostem_min_dist, rel_threshold=rel_threshold,
    )
    dt_peaks, _dt, _wlabels = distance_peaks(
        mask, min_distance=pseudostem_min_dist
    )
    centers = merge_candidates([dt_peaks, frst_peaks], merge_dist=0.8 * pseudostem_min_dist)

    # 6. agrupamiento en macollas
    labels, mats = cluster_mats(centers, mat_eps_px)
    mat_centroids = np.array([m["centroid"] for m in mats]) if mats else np.empty((0, 2))
    reg = planting_regularity(mat_centroids)

    return DetectionResult(
        mask=mask,
        centers=centers,
        score_map=score,
        labels=labels,
        mats=mats,
        grid=grid,
        regularity=reg,
        warnings=warnings,
        params=dict(
            gsd_cm=gsd_cm,
            px_per_m=round(px_per_m, 2) if px_per_m else None,
            leaf_scale_px=int(leaf_scale_px),
            pseudostem_min_dist=int(pseudostem_min_dist),
            mat_eps_px=int(mat_eps_px),
            radii=[int(r) for r in radii],
            mode=mode,
            rel_threshold=float(rel_threshold),
            grid_used=bool(grid_ok),
        ),
    )
