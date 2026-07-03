"""Soporte geoespacial: lectura de ortomosaicos GeoTIFF y conversion a coordenadas.

Un ortomosaico de dron real es un GeoTIFF georreferenciado (con transformada afin y
CRS) de decenas de miles de pixeles por lado. Este modulo:

  - Lee el raster (por ventanas, sin cargarlo entero en memoria) si esta ``rasterio``.
  - Calcula el GSD (cm/pixel) automaticamente desde la georreferencia.
  - Convierte coordenadas de pixel (fila, columna) a lon/lat (EPSG:4326).

Si ``rasterio`` no esta instalado, cae con elegancia a una imagen normal (imageio):
funciona igual, pero sin georreferencia (hay que pasar --gsd y no habra lon/lat).
"""
from __future__ import annotations

import math

import numpy as np

try:
    import rasterio
    from rasterio.warp import transform as _warp_transform
    from rasterio.windows import Window

    _HAS_RIO = True
except Exception:  # pragma: no cover
    _HAS_RIO = False

import imageio.v2 as imageio


def _gsd_cm_from_dataset(ds):
    """Calcula el GSD en cm/pixel desde la resolucion del dataset y su CRS."""
    xres = abs(ds.transform.a)
    yres = abs(ds.transform.e)
    res = 0.5 * (xres + yres)
    crs = ds.crs
    if crs is None:
        return None
    if crs.is_geographic:
        # grados -> metros (aprox): 1 grado de latitud ~ 111320 m.
        lat = ds.bounds.top - 0.5 * (ds.bounds.top - ds.bounds.bottom)
        m_per_deg = 111320.0 * math.cos(math.radians(lat))
        res_m = res * m_per_deg
    else:
        # CRS proyectado: la resolucion ya esta en las unidades del CRS (normalmente m).
        res_m = res
    return res_m * 100.0


class Raster:
    """Abstraccion sobre un ortomosaico, con o sin georreferencia."""

    def __init__(self, path, gsd_cm=None):
        self.path = path
        self._ds = None
        self._array = None
        self.has_geo = False
        self._crs = None
        self._transform = None

        # solo intentamos la ruta geoespacial en formatos que la soportan
        _geo_ext = (".tif", ".tiff", ".vrt", ".jp2", ".img", ".gpkg")
        opened_geo = False
        if _HAS_RIO and str(path).lower().endswith(_geo_ext):
            try:
                ds = rasterio.open(path)
                if ds.crs is not None and ds.transform is not None:
                    self._ds = ds
                    self._crs = ds.crs
                    self._transform = ds.transform
                    self.has_geo = True
                    self.height = ds.height
                    self.width = ds.width
                    self.n_bands = ds.count
                    auto = _gsd_cm_from_dataset(ds)
                    self.gsd_cm = gsd_cm or auto
                    self.gsd_auto = auto
                    opened_geo = True
                else:
                    ds.close()
            except Exception:
                opened_geo = False

        if not opened_geo:
            arr = imageio.imread(path)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            self._array = arr[..., :3]
            self.height, self.width = self._array.shape[:2]
            self.n_bands = 3
            self.gsd_cm = gsd_cm
            self.gsd_auto = None

    @property
    def shape(self):
        return (self.height, self.width)

    def read_window(self, y0, x0, h, w):
        """Lee una ventana [y0:y0+h, x0:x0+w] como HxWx3 uint8."""
        y1 = min(y0 + h, self.height)
        x1 = min(x0 + w, self.width)
        if self.has_geo:
            win = Window(x0, y0, x1 - x0, y1 - y0)
            n = min(3, self.n_bands)
            data = self._ds.read(list(range(1, n + 1)), window=win)  # (bands, h, w)
            arr = np.moveaxis(data, 0, -1)
            if arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=-1)
            if arr.dtype != np.uint8:
                mx = float(arr.max()) if arr.size else 1.0
                arr = (255.0 * arr / mx).astype(np.uint8) if mx > 0 else arr.astype(np.uint8)
            return arr
        return self._array[y0:y1, x0:x1]

    def read_overview(self, max_side=2000):
        """Lee una version reducida del raster completo (para el mapa del informe)."""
        scale = max(1, int(math.ceil(max(self.height, self.width) / max_side)))
        oh, ow = self.height // scale, self.width // scale
        if self.has_geo:
            data = self._ds.read(
                list(range(1, min(3, self.n_bands) + 1)),
                out_shape=(min(3, self.n_bands), oh, ow),
            )
            arr = np.moveaxis(data, 0, -1)
            if arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=-1)
            if arr.dtype != np.uint8:
                mx = float(arr.max()) if arr.size else 1.0
                arr = (255.0 * arr / mx).astype(np.uint8) if mx > 0 else arr.astype(np.uint8)
        else:
            arr = self._array[::scale, ::scale]
        return arr, scale

    def pixel_to_lonlat(self, rows, cols):
        """Convierte (fila, columna) a (lon, lat) en EPSG:4326. None si no hay geo."""
        if not self.has_geo:
            return None
        rows = np.asarray(rows, dtype=float)
        cols = np.asarray(cols, dtype=float)
        # centro del pixel
        xs, ys = rasterio.transform.xy(self._transform, rows, cols, offset="center")
        xs = np.atleast_1d(np.asarray(xs, dtype=float))
        ys = np.atleast_1d(np.asarray(ys, dtype=float))
        lon, lat = _warp_transform(self._crs, "EPSG:4326", xs, ys)
        return np.asarray(lon), np.asarray(lat)

    def close(self):
        if self._ds is not None:
            self._ds.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
