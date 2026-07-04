# Referencia de API (Python)

`banano-drone` es también una librería. Todo lo público se importa desde `banano`.

## Inicio rápido

```python
import imageio.v2 as imageio
from banano import detect_banana

img = imageio.imread("tile.png")
res = detect_banana(img, gsd_cm=3.0)
print(res.summary())
print(res.n_mats, res.n_pseudostems, res.pseudostem_range)
```

## Ortomosaico completo con georreferencia

```python
from banano import Raster, process_orthomosaic
from banano.report import write_all

raster = Raster("ortomosaico.tif")          # detecta GSD y CRS solo
res = process_orthomosaic(raster, tile=1024, overlap=128)
paths = write_all("resultados", res, raster=raster, input_name="ortomosaico.tif")
raster.close()
print(res.summary())                         # dict con conteos, cobertura, avisos
```

## Configuración reproducible

```python
from banano import PipelineConfig, process_orthomosaic, Raster

cfg = PipelineConfig(gsd_cm=2.5, mode="both", rel_threshold=0.30, tile=1024, overlap=128)
# o desde YAML:  cfg = PipelineConfig.from_yaml("config.yaml")
cfg.validate()                               # lanza ConfigError si algo esta mal

raster = Raster("orto.tif")
res = process_orthomosaic(raster, config=cfg)
```

## Camino de deep learning

```python
from banano import PipelineConfig, Raster, process_orthomosaic

cfg = PipelineConfig(gsd_cm=2.5, model_weights="models/banano_seg_synth_v1.pt", model_conf=0.6)
raster = Raster("orto.tif")
res = process_orthomosaic(raster, config=cfg)   # usa el modelo para detectar macollas
```

## Objetos principales

### `detect_banana(rgb, gsd_cm=None, mode="both", use_mask=True, rel_threshold=0.30, use_grid=True, config=None) -> DetectionResult`
Detección clásica sobre una imagen RGB (un tile).

### `DetectionResult`
- `.n_mats`, `.n_pseudostems` — conteos.
- `.pseudostem_range` — `(min, max)` honesto.
- `.mask`, `.centers`, `.mats`, `.grid`, `.regularity`, `.warnings`, `.params`.
- `.summary()` — dict serializable.

### `Raster(path, gsd_cm=None)`
Lector de ortomosaico (GeoTIFF con `[geo]`, o imagen normal). `.has_geo`, `.gsd_cm`,
`.read_window(...)`, `.pixel_to_lonlat(...)`.

### `process_orthomosaic(raster, config=..., progress=...) -> OrthoResult`
Procesa un ortomosaico completo por tiles con deduplicación. `OrthoResult.summary()`.

### `PipelineConfig`
Todos los parámetros ajustables. `.validate()`, `.from_yaml(path)`, `.from_dict(d)`,
`.to_dict()`, `.to_json()`.

## Excepciones

Todas heredan de `banano.BananoError`: `InputError`, `ConfigError`, `RasterError`,
`DependencyError`, `ModelError`. Captúralas para integraciones robustas.

```python
from banano import BananoError
try:
    res = detect_banana(img, gsd_cm=-1)   # ConfigError
except BananoError as e:
    print("Error controlado:", e)
```
