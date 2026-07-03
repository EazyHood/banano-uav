"""Procesamiento de un ortomosaico completo por tiles, con deduplicacion.

Un ortomosaico real no cabe en memoria ni en un solo paso de deteccion. Este modulo:

  1. Recorre el raster en tiles con solape.
  2. Corre la deteccion en cada tile.
  3. Asigna cada deteccion al tile cuyo NUCLEO la contiene (evita contar dos veces
     las plantas del solape) + una deduplicacion espacial de seguridad.
  4. Reagrupa globalmente en macollas y adjunta lon/lat si hay georreferencia.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .mats import cluster_mats, planting_regularity
from .pipeline import detect_banana


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


def process_orthomosaic(
    raster,
    gsd_cm=None,
    tile=1024,
    overlap=128,
    mode="both",
    rel_threshold=0.25,
    use_mask=True,
    progress=None,
):
    """Procesa un objeto Raster completo y devuelve un OrthoResult."""
    H, W = raster.shape
    gsd = gsd_cm if gsd_cm else raster.gsd_cm
    step = max(1, tile - overlap)

    coords = [(y, x) for y in range(0, H, step) for x in range(0, W, step)]
    all_pts = []
    canopy_px = 0
    core_px = 0
    warns = []
    params_sample = None
    grid_sample = {}

    for i, (y0, x0) in enumerate(coords):
        th = min(tile, H - y0)
        tw = min(tile, W - x0)
        if th < 32 or tw < 32:
            continue
        img = raster.read_window(y0, x0, th, tw)
        res = detect_banana(
            img, gsd_cm=gsd, mode=mode, use_mask=use_mask, rel_threshold=rel_threshold
        )

        # nucleo del tile (excluye medio solape salvo en bordes de la imagen)
        my0 = 0 if y0 == 0 else overlap // 2
        mx0 = 0 if x0 == 0 else overlap // 2
        my1 = th if (y0 + th >= H) else th - overlap // 2
        mx1 = tw if (x0 + tw >= W) else tw - overlap // 2

        for r, c in res.centers:
            if my0 <= r < my1 and mx0 <= c < mx1:
                all_pts.append((y0 + r, x0 + c))

        core_mask = res.mask[my0:my1, mx0:mx1]
        canopy_px += int(core_mask.sum())
        core_px += int(core_mask.size)

        params_sample = res.params
        grid_sample = res.grid
        for w in res.warnings:
            if w not in warns:
                warns.append(w)
        if progress:
            progress(i + 1, len(coords))

    pts = np.array(all_pts, dtype=float) if all_pts else np.empty((0, 2))

    min_dist = params_sample["pseudostem_min_dist"] if params_sample else 12
    mat_eps = params_sample["mat_eps_px"] if params_sample else 20
    pts = _dedup(pts, 0.6 * min_dist)

    labels, mats = cluster_mats(pts, mat_eps)
    mat_centroids = (
        np.array([m["centroid"] for m in mats]) if mats else np.empty((0, 2))
    )
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
