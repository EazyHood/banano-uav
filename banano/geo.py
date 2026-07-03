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
    xres = abs(ds.transform.a)  # columnas -> x (longitud si es geografico)
    yres = abs(ds.transform.e)  # filas    -> y (latitud si es geografico)
    crs = ds.crs
    if crs is None:
        return None
    if crs.is_geographic:
        # grados -> metros (aprox): 1 grado de LATITUD ~ 111320 m; 1 grado de
        # LONGITUD ~ 111320 * cos(lat). El factor cos(lat) SOLO aplica a la longitud.
        lat = ds.bounds.top - 0.5 * (ds.bounds.top - ds.bounds.bottom)
        xres_m = xres * 111320.0 * math.cos(math.radians(lat))
        yres_m = yres * 111320.0
        res_m = 0.5 * (xres_m + yres_m)
    else:
        # CRS proyectado: la resolucion ya esta en las unidades del CRS (normalmente m).
        res_m = 0.5 * (xres + yres)
    return res_m * 100.0


def _normalize_rgb(arr):
    """Garantiza HxWx3 uint8 de forma DETERMINISTA (independiente del contenido).

    - Canales: 1 -> repite; 2 (gris+alfa u otra) -> usa la 1a banda; >=3 -> toma 3.
    - dtype: uint8 tal cual; uint16 -> /256; otros enteros -> escala por su maximo de
      tipo; flotante -> asume [0,1] si max<=1, si no recorta a [0,255]. NO se normaliza
      por el max del tile (eso haria que cada tile saliera con distinto brillo).
    """
    if arr.ndim == 2:
        arr = arr[..., None]
    c = arr.shape[-1]
    if c == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif c == 2:
        arr = np.repeat(arr[..., :1], 3, axis=-1)
    elif c > 3:
        arr = arr[..., :3]

    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr)
    if arr.dtype == np.uint16:
        return (arr // 256).astype(np.uint8)
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        return (255.0 * (arr.astype(np.float64) - info.min) / (info.max - info.min)).astype(np.uint8)
    # flotante
    arr = arr.astype(np.float32)
    mx = float(arr.max()) if arr.size else 1.0
    if mx <= 1.0:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


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
            return _normalize_rgb(arr)
        return _normalize_rgb(self._array[y0:y1, x0:x1])

    def read_overview(self, max_side=2000):
        """Version reducida del raster completo (para el mapa del informe).

        Devuelve (arr, sy, sx) con los factores de reduccion REALES por eje
        (alto/ancho reales / los del overview), no un entero aproximado: asi los
        marcadores del mapa se alinean con el fondo aunque el factor no sea entero.
        """
        scale = max(1, int(math.ceil(max(self.height, self.width) / max_side)))
        if self.has_geo:
            oh, ow = max(1, self.height // scale), max(1, self.width // scale)
            data = self._ds.read(
                list(range(1, min(3, self.n_bands) + 1)),
                out_shape=(min(3, self.n_bands), oh, ow),
            )
            arr = _normalize_rgb(np.moveaxis(data, 0, -1))
        else:
            arr = _normalize_rgb(self._array[::scale, ::scale])
        sy = self.height / arr.shape[0]
        sx = self.width / arr.shape[1]
        return arr, sy, sx

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
