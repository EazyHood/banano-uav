"""Procesamiento de un ortomosaico completo por tiles, con deduplicacion.

Un ortomosaico real no cabe en memoria ni en un solo paso de deteccion. Este modulo:

  1. Recorre el raster en tiles con solape.
  2. Corre la deteccion en cada tile.
  3. Asigna cada deteccion al tile cuyo NUCLEO la contiene (evita contar dos veces
     las plantas del solape) + una deduplicacion espacial de seguridad.
  4. Reagrupa globalmente en macollas y adjunta lon/lat si hay georreferencia.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .config import PipelineConfig
from .mats import cluster_mats, planting_regularity
from .pipeline import detect_banana

logger = logging.getLogger(__name__)


@dataclass
class OrthoResult:
    mats: list
    pseudostems_px: np.ndarray
    n_mats: int
    n_pseudostems: int
    canopy_fraction: float
    gsd_cm: float | None
    has_geo: bool
    grid: dict = field(default_factory=dict)
    regularity: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    params: dict = field(default_factory=dict)

    @property
    def pseudostem_range(self):
        return (self.n_mats, self.n_pseudostems)

    def summary(self) -> dict:
        return {
            "n_macollas": self.n_mats,
            "pseudotallos_estimados": self.n_pseudostems,
            "pseudotallos_rango": list(self.pseudostem_range),
            "pseudotallos_por_macolla": round(self.n_pseudostems / self.n_mats, 2)
            if self.n_mats
            else 0.0,
            "cobertura_dosel_%": round(100.0 * self.canopy_fraction, 1),
            "gsd_cm": self.gsd_cm,
            "georreferenciado": self.has_geo,
            "regularidad_siembra": round(self.regularity.get("score", 0.0), 3),
            "avisos": list(self.warnings),
        }


def _dedup(pts, dist):
    """Deduplicacion espacial casi lineal por rejilla (celda = dist)."""
    if len(pts) == 0:
        return pts
    dist = max(1.0, float(dist))
    d2 = dist * dist
    grid = defaultdict(list)
    kept = []
    for p in pts:
        gy, gx = int(p[0] // dist), int(p[1] // dist)
        dup = False
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                for q in grid[(gy + dy, gx + dx)]:
                    if (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 < d2:
                        dup = True
                        break
                if dup:
                    break
            if dup:
                break
        if not dup:
            grid[(gy, gx)].append(p)
            kept.append(p)
    return np.array(kept, dtype=float)


def _tile_starts(total, tile, step):
    """Posiciones de inicio de tile a lo largo de un eje.

    Se detiene en cuanto un tile alcanza el borde, de modo que NUNCA emite una
    tesela final redundante (dos tiles cuyo nucleo llega al mismo borde), lo que
    duplicaria el conteo de cobertura en la banda solapada.
    """
    starts = []
    y = 0
    while True:
        starts.append(y)
        if y + tile >= total:
            break
        y += step
    return starts


def _detect_tile_model(img, model, config):
    """Detecta macollas con el modelo y calcula la cobertura de dosel del tile."""
    from .indices import to_rgb_float
    from .segment import banana_canopy_mask, illumination_correct

    pred = model.predict_mats(img)
    centers = pred["centroids"]

    gsd = config.gsd_cm
    px_per_m = (100.0 / gsd) if (gsd and gsd > 0) else None
    leaf_scale_px = (
        max(5, int(config.leaf_len_m * 0.667 * px_per_m))
        if px_per_m
        else max(9, min(img.shape[:2]) // 60)
    )
    try:
        mask, _ = banana_canopy_mask(
            illumination_correct(to_rgb_float(img)), leaf_scale_px=leaf_scale_px
        )
    except Exception:
        mask = np.ones(img.shape[:2], dtype=bool)

    mat_eps = max(4, int(1.0 * px_per_m)) if px_per_m else max(8, min(img.shape[:2]) // 25)
    params = {
        "pseudostem_min_dist": max(3, int(config.pseudostem_sep_m * px_per_m)) if px_per_m else 12,
        "mat_eps_px": mat_eps,
        "model": model.weights,
    }
    return centers, mask, params, {"spacing_px": None, "strength": 0.0}, []


def process_orthomosaic(
    raster,
    gsd_cm=None,
    tile=1024,
    overlap=128,
    mode="both",
    rel_threshold=0.25,
    use_mask=True,
    progress=None,
    config: PipelineConfig | None = None,
):
    """Procesa un objeto Raster completo y devuelve un OrthoResult.

    Un fallo al detectar en un tile individual se registra y se omite ese tile,
    sin abortar todo el ortomosaico (robustez para lotes grandes).
    """
    H, W = raster.shape
    gsd = gsd_cm if gsd_cm else raster.gsd_cm

    if config is None:
        config = PipelineConfig(
            gsd_cm=gsd,
            mode=mode,
            use_mask=use_mask,
            rel_threshold=rel_threshold,
            tile=tile,
            overlap=overlap,
        ).validate()
    else:
        config.gsd_cm = config.gsd_cm if config.gsd_cm else gsd
        config.validate()

    tile, overlap = config.tile, config.overlap
    step = tile - overlap

    # camino de deep learning (opcional): el modelo detecta macollas directamente
    model = None
    if config.model_weights:
        from .model import BananaModel

        model = BananaModel(config.model_weights, conf=config.model_conf, imgsz=min(tile, 1280))
        logger.info("Usando modelo YOLOv8-seg: %s", config.model_weights)

    coords = [(y, x) for y in _tile_starts(H, tile, step) for x in _tile_starts(W, tile, step)]
    logger.info(
        "Ortomosaico %dx%d px en %d tiles (tile=%d, overlap=%d, gsd=%s)",
        W,
        H,
        len(coords),
        tile,
        overlap,
        gsd,
    )
    all_pts = []
    canopy_px = 0
    core_px = 0
    warns = []
    params_sample = None
    grid_sample = {}
    failed_tiles = 0

    for i, (y0, x0) in enumerate(coords):
        th = min(tile, H - y0)
        tw = min(tile, W - x0)
        if th < 32 or tw < 32:
            continue
        try:
            img = raster.read_window(y0, x0, th, tw)
            if model is not None:
                tile_centers, tile_mask, tparams, tgrid, twarns = _detect_tile_model(
                    img, model, config
                )
            else:
                res = detect_banana(img, config=config)
                tile_centers = res.centers
                tile_mask = res.mask
                tparams, tgrid, twarns = res.params, res.grid, res.warnings
        except Exception as e:  # un tile malo no debe tumbar el lote
            failed_tiles += 1
            logger.warning("Tile (%d,%d) fallo y se omite: %s", y0, x0, e)
            if progress:
                progress(i + 1, len(coords))
            continue

        # nucleo del tile (excluye medio solape salvo en bordes de la imagen)
        my0 = 0 if y0 == 0 else overlap // 2
        mx0 = 0 if x0 == 0 else overlap // 2
        my1 = th if (y0 + th >= H) else th - overlap // 2
        mx1 = tw if (x0 + tw >= W) else tw - overlap // 2

        for r, c in tile_centers:
            if my0 <= r < my1 and mx0 <= c < mx1:
                all_pts.append((y0 + r, x0 + c))

        core_mask = tile_mask[my0:my1, mx0:mx1]
        canopy_px += int(core_mask.sum())
        core_px += int(core_mask.size)

        params_sample = tparams
        grid_sample = tgrid
        for w in twarns:
            if w not in warns:
                warns.append(w)
        if progress:
            progress(i + 1, len(coords))

    if failed_tiles:
        warns.append(f"{failed_tiles} tile(s) fallaron y se omitieron (ver el log).")

    pts = np.array(all_pts, dtype=float) if all_pts else np.empty((0, 2))

    min_dist = params_sample["pseudostem_min_dist"] if params_sample else 12
    mat_eps = params_sample["mat_eps_px"] if params_sample else 20

    if model is not None:
        # el modelo ya devuelve macollas: deduplicar por escala de macolla, sin DBSCAN
        pts = _dedup(pts, 0.5 * mat_eps)
        mats = [{"centroid": p, "n_pseudostems": 1, "members": p[None, :]} for p in pts]
        labels = np.arange(len(mats))
        warns.append("Conteo por modelo de deep learning: unidad = macolla.")
    else:
        pts = _dedup(pts, 0.6 * min_dist)
        labels, mats = cluster_mats(pts, mat_eps)
    mat_centroids = np.array([m["centroid"] for m in mats]) if mats else np.empty((0, 2))
    reg = planting_regularity(mat_centroids)

    # georreferencia
    if raster.has_geo and mats:
        res_ll = raster.pixel_to_lonlat(mat_centroids[:, 0], mat_centroids[:, 1])
        if res_ll is not None:
            lon, lat = res_ll
            for i, m in enumerate(mats):
                m["lon"] = float(lon[i])
                m["lat"] = float(lat[i])

    return OrthoResult(
        mats=mats,
        pseudostems_px=pts,
        n_mats=len(mats),
        n_pseudostems=int(pts.shape[0]),
        canopy_fraction=(canopy_px / core_px) if core_px else 0.0,
        gsd_cm=gsd,
        has_geo=raster.has_geo,
        grid=grid_sample,
        regularity=reg,
        warnings=warns,
        params=params_sample or {},
    )
