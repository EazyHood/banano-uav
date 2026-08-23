# Changelog

Todas las novedades notables de este proyecto se documentan aquí.
Formato basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/).

## [Sin publicar] — 2026-08-23

El tema de esta versión es **sacar el entrenamiento del PC del autor** — su equipo falla y
no puede seguir siendo la única máquina capaz de reproducir este proyecto. Al portarlo
aparecieron tres defectos de medición que estaban tapados.

### Añadido
- **Entrenamiento en la nube, desatendido** (`kaggle/`). Un comando sube el notebook a
  Kaggle y arranca la corrida; el PC se puede apagar. 30 h de GPU por semana gratis, sin
  tarjeta, y con 2× Tesla T4 de 16 GB (el doble de VRAM que la RTX 5060, que era lo que
  obligaba a `batch=4`). El notebook aborta si le toca una P100 —la imagen de Kaggle no
  trae kernels de Pascal y el entreno moriría en el primer lote pareciendo que la GPU sí
  estaba— y se presupuesta 11 h de las 12 disponibles, porque al pasarse el guardado de
  ficheros es "best effort" y se pueden perder los pesos.
- **Reconstrucción del dataset sin subir nada** (`cloud/data_manifest.json`,
  `cloud/fetch_data.py`). `realdata/` está en `.gitignore`, así que el repo publicaba
  modelos entrenados sobre datos que nadie más podía rehacer. El manifiesto fija workspace,
  proyecto, **versión** y licencia de las 16 fuentes, y la máquina remota las descarga de
  Roboflow por su cuenta. Reproduce el dataset **exacto**, verificado split a split
  (armah 43/12/7, elliot 322/37/37, count_banana_plants 502/150/50).
- **`deep/leak_audit.py`** — busca imágenes de validación que estén dentro del
  entrenamiento, comparando por MD5 del contenido en vez de por nombre.
- **`deep/scale_audit.py`** — mide de qué tamaño aparece una planta en cada fuente.
- **`cloud/scale_sweep.py`** — barrido de resolución de inferencia, sin reentrenar.
- **Protocolo Leave-One-Farm-Out** (`splits/lofo_*.yaml`, `cloud/make_splits.py`): 7
  particiones con cada finca retenida entera, **6.150 imágenes de validación en vez de 62**.
  0 % de fuga en las siete.
- **`cloud/nuevas_fincas.json`** — 6 fincas de banano nuevas y 2 análogos de preentreno, con
  descarga comprobada y cruzados por hash perceptual contra el corpus para descartar forks.
  Incluye lo descartado *con su prueba*.
- **`docs/escala-y-fugas.md`** — todo lo medido, con cómo reproducirlo.
- 19 pruebas nuevas, validadas por mutación.

### Corregido
- **Fuga train/test en el protocolo que publica 0.746.** `extra/prueba2rgb` y
  `extra/etiquetasnuevas` son el mismo dataset: 505 imágenes idénticas byte a byte, subidas
  a dos workspaces distintos de Roboflow. Como Roboflow renombra cada exportación con un
  hash propio, el dedup "por stem" no podía verlas. Las 25 imágenes de `prueba2rgb/test` que
  `holdout.yaml` usa como validación están **el 100 %** dentro del entrenamiento de v10 y de
  v12: 25 de las 99 del protocolo. Los otros tres protocolos dan 0 % de fuga, así que **el
  0.172 de finca nueva se sostiene entero**.
- **`max_det` de 300 a 1000.** El valor por defecto topaba el recall al 50 % por
  construcción en la imagen del holdout que tiene 600 plantas reales.
- **La receta de control ahora es la que de verdad entrenó a v10** (`degrees: 0`,
  `flipud: 0`, `scale: 0.5`, leído de `runs10/.../args.yaml`). La "augmentación cenital" que
  describe el docstring de `deep/train_v12.py` se aplicó sólo a v12, que no se publicó: el
  modelo que la gente descarga nunca vio una planta rotada ni volteada.
- **Rutas.** Los 13 YAML de datos llevaban `C:/Users/jhona/...` en la primera línea y no
  resolvían en ninguna otra máquina.

### Medido
- **El fallo en finca nueva es, antes que nada, de escala.** A imgsz 768 la planta mediana
  mide 10 px en una finca y 333 px en otra (33×), y el 81 % de las cajas del entrenamiento
  de v10 son plantas de 16-17 px. La finca ciega las tiene de 45 px, donde hay menos del 1 %
  del entrenamiento. Consecuencia práctica sin coste: **subir la inferencia de 768 a 1024
  lleva su mAP50 de 0.172 a 0.285 y su recall de 0.139 a 0.229**, con los mismos pesos.
  En `holdout_newfarms` el efecto es ×23 (0.0053 → 0.1219 a 1280 px).
- **La diversidad real es menor que el recuento de imágenes.** Contando disparos originales
  en vez de ficheros, `plantas_jovenes_80m1` son 407 fotos (no 2.159) y `plantas_platano`
  304 (no 1.600). El 81 % de las cajas sale de 783 fotos de un solo operador. Las fuentes
  realmente independientes del entrenamiento de v10 son **tres**, no cinco.
- **Lo que esto NO arregla, dicho claro:** el conteo en finca nueva sigue sin ser utilizable.
  A 1024 px el modelo predice 295 de 1.378 plantas reales (-78,6 %, frente a -84,5 % a 768).
  Sin etiquetar algo de la finca destino, no hay producto de conteo.

## [2.2.0] — 2026-08-09 / 2026-08-11

El tema de esta versión es **medir de verdad**: el repo ya traía un modelo multi-finca del
que no había ni una cifra guardada, y las que se publicaban venían de protocolos que se
elegían a sí mismos. Ahora hay 4 protocolos, registro en disco de cada corrida y una cifra
de conteo que nadie ha elegido mirando el resultado.

### Añadido
- **Modelo multi-finca publicado**: `models/banana_multifarm_v10.pt` (YOLO11m, 768 px,
  ~5.100 imágenes de 5 fincas independientes). Es **exactamente** el fichero con el que se
  midió (SHA256 `8e4b7d1f7eba3651`), no una copia re-guardada.
  - finca original: mAP50 **0.861**, error del conteo total **1,8 %**;
  - otras 4 fincas del entrenamiento: mAP50 **0.746**, error **8,4 %**;
  - **finca que ningún modelo vio: mAP50 0.172, recall 0.139** — la cifra que de verdad
    predice lo que pasa en una finca nueva, y por eso va en el README.
- **`docs/modelo-multifinca.md`**: los 4 protocolos, las dos tablas, la calibración del
  umbral y por qué el modelo v12 **no** se publica (medido: pierde en las dos únicas
  pruebas honestas; sus victorias están en fincas que lleva en el entrenamiento).
- **`deep/eval_v12_suite.py`**: los 4 protocolos de una tacada, mAP + conteo, con
  comparación contra los baselines ya registrados.
- **`model_imgsz`** (YAML y `--model-imgsz`): fija la resolución de inferencia. Antes el
  pipeline infería a la resolución del tile (1024 por defecto) mientras los modelos
  multi-finca están entrenados a 768, y no había forma de casarlo sin tocar código.
- Prueba que compara `banano.__version__` con la versión de `pyproject.toml`.

### Cambiado
- **Error de conteo honesto**: `deep/eval_count.py` elegía el mejor umbral de confianza
  mirando el mismo conjunto sobre el que después reportaba el error. Ahora calibra en un
  pliegue y mide en el otro (partiendo cada finca por bloques, no alternando tiles
  vecinos). El JSON conserva la cifra in-sample, pero etiquetada como tal.
- Los registros de conteo guardan los **conteos por umbral y por imagen**: recalibrar
  después ya no cuesta otra pasada de GPU.
- `config.example.yaml` documenta el umbral como un **rango medido** (0,10-0,25) en vez de
  una cifra única: con este modelo, 0,10 da 1,8 % de error en una finca y 48 % en otra.

### Corregido
- `config.example.yaml` y el README anunciaban `models/banana_multifarm_v12.pt`, un fichero
  que **no existía en el repo**.
- `banano/__init__.py` se había quedado en 2.1.0 mientras `pyproject.toml` declaraba 2.2.0:
  `banano-detect --version` imprimía una versión distinta a la del paquete.

## [2.1.0] — 2026-07-03 / 2026-07-04

Sube la precisión por encima del 98 % de acierto en las tres métricas (sobre sintético)
y añade el primer modelo entrenado con imágenes reales. (Nota: esta versión se publicó
en dos tandas; antes figuraba como dos secciones [2.1.0] separadas.)

### Cambiado
- **Modelo mayor**: YOLOv8**s**-seg (11.8 M par.) entrenado con dataset ampliado (400+80
  tiles, ~17k instancias, 100 épocas). Sustituye al yolov8n.
- **Benchmark (25 ortomosaicos, tol. estricta 0.5 m)**: **F1 0.993 (99.3 %)**, MAPE **1.23 %**,
  error de conteo total **1.23 %** — las tres cumplen ≥98 % / ≤2 %. (Clásico: F1 0.805, 4.4 %.)
  Estas cifras son del modelo sintético sobre datos sintéticos.
- `model_conf` por defecto calibrado a **0.55** (sintético); `overlap` por defecto del
  benchmark a 128.

### Añadido
- **Test-time augmentation** opcional (`model_augment` en el YAML de configuración): más
  precisión en inferencia a cambio de velocidad. (El flag `--augment` de la CLI llegó
  en 2.2.0.)
- **Modelo entrenado con imágenes UAV REALES de banano** (`models/banano_real_v1.pt`): detector
  YOLOv8 entrenado sobre ~14 000 tiles reales (dataset abierto AI-BananaMapping, Zenodo
  20945958, CC-BY-4.0). Listo para usar sobre ortofotos reales **sin entrenar**.
  - Rendimiento sobre 4 611 tiles de **test real** nunca vistos: **mAP50 0.411**, precisión 0.47,
    recall 0.46 (cifras honestas de campo).
- Utilidades: `deep/prepare_real_dataset.py` (colapsa clases a "banano"), `deep/eval_real.py`
  (evalúa sobre el split de test real + ejemplos visuales).
- Opción `model_augment` (test-time augmentation) en `PipelineConfig`.
- Documentación del modelo real: [`docs/modelo-real.md`](docs/modelo-real.md).

### Corregido
- Entrenamiento en Windows: ejecutar desde un archivo `.py` real (no heredoc por stdin) para
  que el multiprocessing del DataLoader (`workers>0`) no falle con `OSError: '<stdin>'`.

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
