# Changelog

Todas las novedades notables de este proyecto se documentan aquí.
Formato basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/).

## [2.1.0] — 2026-07-03

Sube la precisión por encima del 98 % de acierto en las tres métricas.

### Cambiado
- **Modelo mayor**: YOLOv8**s**-seg (11.8 M par.) entrenado con dataset ampliado (400+80
  tiles, ~17k instancias, 100 épocas). Sustituye al yolov8n.
- **Benchmark (25 ortomosaicos, tol. estricta 0.5 m)**: **F1 0.993 (99.3 %)**, MAPE **1.23 %**,
  error de conteo total **1.23 %** — las tres cumplen ≥98 % / ≤2 %. (Clásico: F1 0.805, 4.4 %.)
- `model_conf` por defecto calibrado a **0.55**; `overlap` por defecto del benchmark a 128.

### Añadido
- **Test-time augmentation** opcional (`model_augment`, `--augment`): más precisión en
  inferencia a cambio de velocidad.

## [2.0.0] — 2026-07-03

Salto a nivel de producción ("AAA"): robustez, modelo entrenado, benchmark y empaquetado.

### Añadido
- **Configuración centralizada** `PipelineConfig` con validación y carga desde YAML
  (`--config`); reproducibilidad y auditoría de parámetros.
- **Modelo de deep learning** integrado: `banano/model.py` (YOLOv8-seg), pipeline completo
  dataset→entrenamiento→inferencia→integración, y **pesos entrenados incluidos**
  (`models/banano_seg_synth_v1.pt`). El modelo detecta macollas directamente.
- **Benchmark honesto** (`deep/benchmark.py`): MAE/RMSE/MAPE + F1 de localización contra
  verdad de terreno, clásico vs modelo.
- **Robustez de producción**: jerarquía de excepciones (`banano/errors.py`), logging
  (`-v`/`--quiet`), validación de entradas, códigos de salida (0/1/2/130), un tile que
  falla no aborta el lote.
- **Empaquetado**: Dockerfile, `MANIFEST.in`, listo para PyPI (twine OK), `config.example.yaml`,
  `CITATION.cff`, `examples/quickstart.py`.
- **Calidad**: ruff + mypy limpios, cobertura de pruebas ≥78% (~44 pruebas), CI ampliado
  (lint + type-check + coverage).
- **Documentación**: guía de campo (`docs/guia-campo.md`) y referencia de API (`docs/api.md`).

### Corregido
- Etiquetas del dataset: una macolla con hijuelos separados generaba rosetas sin etiquetar
  (falsos positivos al entrenar); ahora se usa el **casco convexo** de todo el cluster.
- Segmentación de vegetación uniforme: umbral local (offset 0) la excluía; ahora se une con
  el piso global de Otsu.
- Entrenamiento en Windows: `workers=0` evita el error CUDA "resource already mapped".

### Corregido (2ª revisión adversarial, 11 bugs confirmados)
- **Config**: valores no numéricos (strings de YAML entre comillas, bool, tipos raros) daban
  `TypeError` crudo en vez de `ConfigError`; ahora se coaccionan/validan tipos. `model_weights`
  y `mode` validan su tipo.
- **CLI/geo**: una imagen corrupta lanzaba `ValueError` crudo (código 2); ahora `Raster` lanza
  `RasterError` controlado (código 1).
- **pipeline**: `detect_banana` no validaba un `config` externo; ahora siempre lo valida.
- **ortho**: reparto complementario del solape (floor/ceil) evita solape/hueco de 1 px con
  `overlap` impar; aviso con `overlap=0`; el fallo de segmentación en el camino de modelo ya
  no infla la cobertura al 100 % (cuenta 0 + aviso).
- **benchmark**: la tolerancia de emparejamiento (1.0 m) inflaba el F1; ahora es fija y
  estricta (0.5 m). Números honestos re-medidos.

## [1.0.0] — 2026-07-03

Primera versión pública, utilizable de punta a punta con un dron (flujo post-vuelo).

### Añadido
- **Pipeline geoespacial completo**: lectura de ortomosaicos **GeoTIFF** con detección
  automática del GSD y la georreferencia (`banano/geo.py`), procesado por **tiles con
  deduplicación en bordes** (`banano/ortho.py`).
- **CLI `banano-detect`**: ortomosaico → conteo + capa GIS + informe, en un comando.
- **Entregables**: informe HTML autocontenido, mapa PNG, **GeoJSON** (capa GIS con lon/lat),
  CSV y resumen JSON (`banano/report.py`).
- **Detección híbrida sin datos etiquetados**: corrección de iluminación, dosel por ExGR
  adaptativo + textura + morfología, **marco de siembra por autocorrelación FFT**, centros
  por **transformada de distancia + watershed** fusionados con **Fast Radial Symmetry
  Transform**, agrupamiento en macollas (DBSCAN).
- **Reporte honesto**: conteo fiable a nivel de macolla + **rango** de pseudotallos +
  cobertura de dosel + avisos (guardarraíl de GSD, dosel cerrado).
- Índice **TGI** (Neupane 2019) y camino de deep learning (YOLOv8-seg / ALSS-YOLO-Seg).
- Generador de plantación sintética + GeoTIFF de ejemplo; pruebas; documentación de estado
  del arte con referencias reales.

### Corregido
- Umbrales de picos ahora son **robustos al tamaño del tile** (percentil/absoluto en vez de
  máximo global), evitando que un blob fuerte suprima plantas débiles en tiles grandes.
- **GSD en CRS geográfico**: `cos(lat)` se aplicaba al promedio de resolución; ahora solo
  escala la longitud (la latitud usa 111320 m/grado). Evita subestimar el GSD ~13% a lat 40°.
- **Rasters de 2 bandas** (gris+alfa) ya no lanzan `IndexError`; normalización de bandas y
  dtype (uint16/float) centralizada y determinista (no por máximo de cada tile).
- **`overlap >= tile`** ya no cuelga el proceso (antes `step` colapsaba a 1 → H·W teselas).
- **Teselado**: se elimina la tesela final redundante que duplicaba la cobertura de dosel.
- **Mapa del informe**: factores de escala reales por eje (sin deriva de marcadores).
- **`pyproject.toml`**: `setuptools>=77.0.3` (requerido por los metadatos PEP 639 de licencia).

Todos verificados por una revisión adversarial multi-agente (7 bugs reales confirmados).

### Licencia
- Publicado bajo **AGPL-3.0** para garantizar que permanezca libre y abierto.
