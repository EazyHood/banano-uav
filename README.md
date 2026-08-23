# 🍌 banano-drone

**Identificación y conteo de cultivo de banano en ortomosaicos de dron — libre y abierto.**

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org)

Software libre (AGPL-3.0) para **identificar y contar banano desde un dron**, resolviendo
el problema que hace al banano difícil frente a palma, aguacate o mango. Sin costos, sin
cajas negras que se venden caras: clonas, corres, y tienes tu conteo con capa GIS.

---

## El problema

La palma, el aguacate y el mango **salen como individuos aislados**: cada árbol es una copa
separada, fácil de detectar. El banano **no**: crece en **macollas** (planta madre +
hijuelos), normalmente en **grupos de ~3 pseudotallos** pegados, con las hojas solapadas.
No hay un "individuo" limpio que delinear, y los detectores de copas fallan.

**Solución de este programa:** trabaja en dos niveles y da los dos conteos —
1. **Dosel de banano** (qué píxeles son banano, no suelo ni maleza).
2. **Pseudotallos por simetría radial** (cada planta es una *roseta* vista desde arriba),
   agrupados en **macollas**, la unidad agronómica correcta.

Reporta **nº de macollas** (fiable), un **rango honesto de pseudotallos**, **cobertura de
dosel**, y una **capa GIS georreferenciada** que abres en QGIS/Google Earth.

## El flujo completo con tu dron

```
1. Vuelas el lote          →  dron + app de mapeo (solape 70-80%), a ≤120 m
2. Generas el ortomosaico  →  OpenDroneMap (libre) / Pix4Dfields / Agisoft / DroneDeploy
3. Corres banano-detect    →  este programa: GeoTIFF → conteo + GIS + informe
4. Abres el resultado      →  informe HTML + plantas.geojson en QGIS/Google Earth
```

> **Nota honesta:** "usable en un dron" = flujo **post-vuelo** con el ortomosaico. Es lo que
> hace el 99% del mapeo agrícola. Correr inferencia *a bordo en tiempo real* es otro proyecto
> (mucho mayor) y casi nunca necesario para contar plantas.

## Instalación

```bash
git clone https://github.com/EazyHood/banano-uav
cd banano-uav
python -m venv .venv
# Windows PowerShell:  .venv\Scripts\Activate.ps1   |   Git Bash: source .venv/Scripts/activate
pip install -e .[geo]      # [geo] añade soporte GeoTIFF (rasterio); sin él usa imágenes normales
```

Extras opcionales: `[gis]` (exportar Shapefile con geopandas), `[deep]` (modelo YOLOv8-seg),
`[dev]` (pruebas, lint, type-check).

**Con Docker** (no necesitas instalar nada más):

```bash
docker build -t banano-drone .
docker run -v "$PWD:/data" banano-drone --input /data/ortomosaico.tif --out /data/resultados
```

## Uso

**Sobre un ortomosaico GeoTIFF** (detecta el GSD y la georreferencia solos):

```bash
banano-detect --input mi_ortomosaico.tif --out resultados
```

**Sobre una imagen normal** (pasa la resolución en cm/píxel):

```bash
banano-detect --input foto.jpg --gsd 3.0 --out resultados
```

**Probar sin datos reales** (genera un ortomosaico sintético georreferenciado):

```bash
python scripts/make_example.py --out example/plantacion.tif
banano-detect --input example/plantacion.tif --out resultados
```

### Qué genera en `resultados/`

| Archivo | Contenido |
|---|---|
| `informe.html` | Informe visual autocontenido (resumen + mapa + avisos). Ábrelo en el navegador. |
| `mapa.png` | Mapa del lote con macollas (círculos) y pseudotallos (puntos). |
| `plantas.geojson` | **Capa GIS**: una macolla = un punto, con lon/lat y nº de pseudotallos. Ábrela en QGIS/Google Earth. |
| `plantas.csv` | Tabla de plantas (id, fila/col, lon, lat, pseudotallos). |
| `resumen.json` | Métricas del lote en JSON. |

### Opciones útiles

```
--gsd 3.0            resolución cm/píxel (auto en GeoTIFF; el parámetro de mayor impacto)
--config c.yaml      archivo YAML de configuración (reproducible; ver config.example.yaml)
--tile 1024          tamaño de tile (px) para procesar ortomosaicos grandes
--overlap 128        solape entre tiles (evita cortar plantas en los bordes)
--mode both          centro de simetría: bright / dark / both
--threshold 0.30     umbral de picos (baja si detecta de menos, sube si de más)
--no-mask            no restringir al dosel segmentado (si la segmentación falla)
--weights m.pt       pesos YOLO: activa el camino de deep learning sin editar YAML
--model-conf 0.30    umbral de confianza del modelo (recomendado por modelo, ver YAML)
--augment            test-time augmentation del modelo (más preciso, más lento)
--device cpu         fuerza dispositivo del modelo ('cpu' en máquinas sin GPU)
-v / --quiet         más/menos detalle de log
--version            versión
```

**Configuración reproducible** con un archivo YAML (auditable, versionable):

```bash
cp config.example.yaml mi_config.yaml   # edítalo
banano-detect --input orto.tif --config mi_config.yaml --out resultados
```

## Documentación

- 📘 [**Guía de campo**](docs/guia-campo.md) — cómo volar, generar el ortomosaico e interpretar resultados (para agrónomos y empresas).
- 🐍 [**Referencia de API**](docs/api.md) — usar `banano` como librería Python.
- 📚 [**Estado del arte**](docs/estado-del-arte.md) — fundamento científico y referencias (actualizado 2026-08).
- 🌱 [**Datos reales**](docs/datos-reales.md) — procedencia, licencias, curación anti-fugas y protocolo de evaluación.
- 🤖 [**Modelo real**](docs/modelo-real.md) — alcance y cifras honestas de los modelos entrenados con imágenes reales.

## Cómo funciona (pipeline híbrido, sin datos etiquetados)

Corrección de iluminación → dosel de banano (ExGR adaptativo + textura + morfología) →
**marco de siembra por autocorrelación FFT** (prior geométrico; da escala física aun sin
GSD) → **centros por transformada de distancia + watershed FUSIONADO con Fast Radial
Symmetry Transform** → agrupamiento en macollas (DBSCAN) → **procesado por tiles con
deduplicación** en bordes → georreferenciación a lon/lat.

## Resultados y precisión (benchmark honesto)

Medido sobre plantaciones sintéticas con **verdad de terreno exacta** (`deep/benchmark.py`,
25 ortomosaicos de 1024 px, GSD 3 cm/px, tolerancia estricta 0.5 m). Reproduce con:
`python deep/benchmark.py --n 25 --size 1024 --weights models/banano_seg_synth_v1.pt --model-conf 0.55`

| Método | Acierto de conteo | MAPE por lote | F1 (localización, tol. 0.5 m) |
|---|---|---|---|
| **Modelo YOLOv8s-seg** (incluido) | **98.8 %** (error 1.2 %) ✅ | 1.23 % | **0.993 (99.3 %)** ✅ |
| Clásico (sin datos etiquetados) | 95.6 % (error 4.4 %) | 5.38 % | 0.805 |

- El **modelo entrenado supera el 98 % de acierto** en las tres métricas (conteo, MAPE y
  localización), con error de conteo **1.2 %** sobre datos con verdad de terreno.
- El **clásico** ronda el 95 % pero **no necesita datos etiquetados** — línea base inmediata.
- **Pseudotallos**: se reportan como **rango honesto**; separar plantas *dentro* de una
  macolla es intrínsecamente ambiguo (lo confirma la literatura).

> ⚠️ **Honestidad crítica:** estas cifras (>98 % de acierto) son sobre datos **SINTÉTICOS** con
> verdad de terreno exacta. Demuestran que el sistema, la arquitectura y el modelo son correctos
> y alcanzan el objetivo en un entorno medible. **NO son una promesa de >98 % sobre banano real
> de campo**: la mejor literatura mundial reporta 85-96 % de acierto con deep learning sobre
> banano real. Para tu finca, reentrena con imágenes reales etiquetadas y **mide tu propio
> acierto** ([`docs/guia-campo.md`](docs/guia-campo.md), paso 5).

**Límites de la línea base clásica**: funciona mejor con **GSD ≤ 3 cm/px**; no resuelve por
sí sola malezas de hoja ancha (platanillo/*Heliconia*), dosel adulto totalmente cerrado, ni
plantas subpíxel. Para esos casos, el modelo entrenado. El programa avisa (guardarraíl de GSD,
dosel cerrado) cuando la fiabilidad baja.

## 🎯 Modelo entrenado con imágenes REALES (listo para usar)

Incluye **`models/banana_multifarm_v10.pt`** — un detector YOLO11m entrenado sobre ~5.100
imágenes UAV de **5 fincas independientes** de banano/plátano. Se usa directamente sobre tus
ortofotos, **sin entrenar nada**:

```bash
pip install -e .[deep]
banano-detect --input tu_ortofoto.tif --gsd 3.0 \
    --weights models/banana_multifarm_v10.pt --model-conf 0.10 --model-imgsz 768 \
    --out resultados
```

**Medido en 4 protocolos, del más fácil al más honesto** (mismo modelo, imgsz 768):

| protocolo | mAP50 | error del conteo total | error medio por imagen |
|---|---|---|---|
| finca original (test retenido) | 0.861 | **1,8 %** | 0,16 |
| otras 4 fincas del entrenamiento (test retenido) | 0.746 ⚠️ | 8,4 % | 0,57 |
| **finca nueva que ningún modelo vio** | **0.172** | 77,6 % | 0,63 |

> ⚠️ **La segunda fila está contaminada** (medido el 2026-08-23 con
> [`deep/leak_audit.py`](deep/leak_audit.py)). Ese protocolo valida en parte con
> `extra/prueba2rgb`, que resultó ser **byte a byte el mismo dataset** que
> `extra/etiquetasnuevas`, presente en el entrenamiento: las 505 imágenes coinciden por MD5,
> subidas a dos workspaces distintos de Roboflow con nombres re-hasheados, así que el dedup
> por nombre no podía verlas. Son **25 de las 99 imágenes** del protocolo (25 %) las que el
> modelo ya había visto, así que 0.746 está por encima de lo real. Los otros tres protocolos
> dan **0 % de fuga** y se sostienen enteros. Se vuelve a medir sin la fuente duplicada en la
> próxima tanda de entrenamiento.

> ⚠️ **Lo importante es la última fila.** En las fincas que conoce funciona; en una finca
> nueva de verdad encuentra **~14 de cada 100 plantas**. Si tu finca no se parece a las del
> entrenamiento, etiqueta 100-200 tiles tuyos y afínalo. Y el umbral de confianza **no es
> universal** — hay que calibrarlo con tus imágenes (`deep/eval_count.py` lo hace en una pasada).

> 🔍 **Por qué falla en finca nueva: es, en primer lugar, un problema de escala.**
> [`deep/scale_audit.py`](deep/scale_audit.py) mide que a imgsz 768 la planta mediana ocupa
> **10 px en una finca y 333 px en otra** (33×), y que el **81 % de las cajas** con las que se
> entrenó v10 son plantas de **16-17 px**. La finca nueva las tiene de 45 px, en una banda que
> apenas aparece en el entrenamiento (menos del 1 % de las cajas entre 21 y 46 px). Eso explica
> el recall de 0,139 mejor que "otro suelo, otra luz", y explica que v12 —entrenado con más
> fincas— *empeorase* ahí: añadió masa en los extremos (10 px y 333 px), no donde esa finca vive.
>
> Y tiene una consecuencia práctica que **no cuesta ni una hora de GPU**: subir la resolución de
> inferencia de 768 a **1024** sube el mAP50 de esa finca de 0,172 a **0,285** y el recall de
> 0,139 a **0,229** (+65 % ambos), con los mismos pesos. Medido, reproducible y con su letra
> pequeña —incluido lo que *no* arregla— en
> [`docs/escala-y-fugas.md`](docs/escala-y-fugas.md).

> 🖥️ **Entrenar sin usar tu ordenador.** Todo el camino —descargar los datos, repartirlos por
> fincas, comprobar fugas y entrenar— corre en una GPU gratuita de la nube con un solo comando,
> y el PC se puede apagar: [`kaggle/README.md`](kaggle/README.md).

El error del conteo *total* compensa sobre- y sub-conteos entre imágenes, así que va siempre
acompañado del error **por imagen**. Todo el detalle, la calibración cruzada con la que se
eligieron los umbrales y por qué el modelo v12 se midió pero **no** se publica:
[`docs/modelo-multifinca.md`](docs/modelo-multifinca.md).

También se conserva **`models/banano_real_v1.pt`** (histórico): YOLOv8 sobre ~14 000 tiles del
dataset AI-BananaMapping ([Zenodo](https://zenodo.org/records/20945958), CC-BY-4.0), **mAP50
0.411** en su propio test. Está entrenado contra un objetivo distinto (Sigatoka: planta
enferma, no toda planta), así que sus cifras no son comparables con la tabla de arriba.
Detalle: [`docs/modelo-real.md`](docs/modelo-real.md).

## Entrenar tu propio modelo (opcional)

Incluye también un **modelo YOLOv8-seg sintético** (`models/banano_seg_synth_v1.pt`) y todo el
pipeline reproducible (dataset → entrenamiento → inferencia → integración):

```bash
pip install -e .[deep]

# usar el modelo incluido (detecta macollas directamente):
banano-detect --input orto.tif --config config.example.yaml \
    --out resultados   # pon model_weights: models/banano_seg_synth_v1.pt en el YAML

# o entrenar el tuyo desde cero (sintético o real):
python deep/make_synth_dataset.py --out dataset --train 200 --val 40
python deep/train_yolo.py --data dataset/data.yaml --model yolov8n-seg.pt --epochs 80
python deep/infer_yolo.py --weights runs/segment/banano_seg/weights/best.pt --image tile.png

# medir precisión (benchmark honesto con verdad de terreno):
python deep/benchmark.py --n 20 --size 1024 --weights models/banano_seg_synth_v1.pt
```

> ⚠️ **El modelo incluido está entrenado con datos SINTÉTICOS.** Demuestra el pipeline
> completo y sirve de línea base reproducible, pero para tu finca real debes **reentrenarlo
> con imágenes reales etiquetadas** para el máximo rendimiento. La brecha sintético→real es
> inherente a cualquier modelo; no confíes en cifras de campo sin validar (ver
> [`docs/guia-campo.md`](docs/guia-campo.md), paso 5).

**Recursos reales para arrancar** (ver [`docs/estado-del-arte.md`](docs/estado-del-arte.md)):
- **ALSS-YOLO-Seg** — modelo abierto específico de banano UAV: https://github.com/helloworlder8/computer_vision
- **Dataset de Neupane et al. 2019** (figshare): https://figshare.com/s/62e391492b1be99515b4
- **Roboflow Universe** — busca "banana plantation / banana tree counting".

## Estructura

```
banano/          indices · segment · grid · radial · centers · mats · pipeline
                 geo (GeoTIFF) · ortho (tiles+dedup) · report (CSV/GeoJSON/HTML) · cli
                 config · errors · logconf · model (YOLO det/seg)
models/          pesos entrenados (banana_multifarm_v10.pt · banano_real_v1.pt ·
                 banano_seg_synth_v1.pt)
scripts/         demo.py · run.py · make_example.py
deep/            make_synth_dataset · train_yolo · infer_yolo · benchmark
                 train_v12 · eval_record · eval_count · eval_v12_suite (los 4 protocolos)
real_eval/       registros JSON de cada evaluación (fecha, pesos, SHA, datos, métricas)
docs/            guía de campo · API · estado del arte · datos reales · modelo real
                 modelo multi-finca (los 4 protocolos y su honestidad)
tests/           pruebas (pytest, 72, cobertura 83-87% según extras)
Dockerfile · pyproject.toml · config.example.yaml · .github/workflows/ci.yml
```

## Contribuir

Este proyecto es **libre y comunitario**. Issues y pull requests bienvenidos —
ver [CONTRIBUTING.md](CONTRIBUTING.md). Si mejoras la detección o la validas con imágenes
reales de banano, ¡compártelo!

## Licencia

**AGPL-3.0** — puedes usarlo, modificarlo y redistribuirlo libremente, pero cualquier
versión modificada (incluso ofrecida como servicio web) **debe seguir siendo abierta y
con la misma licencia**. Elegida a propósito para que nadie lo cierre ni lo venda caro.
Ver [LICENSE](LICENSE).
