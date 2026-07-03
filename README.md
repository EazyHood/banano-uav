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
`[dev]` (pruebas).

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
--tile 1024          tamaño de tile (px) para procesar ortomosaicos grandes
--overlap 128        solape entre tiles (evita cortar plantas en los bordes)
--mode both          centro de simetría: bright / dark / both
--threshold 0.30     umbral de picos (baja si detecta de menos, sube si de más)
--no-mask            no restringir al dosel segmentado (si la segmentación falla)
```

## Cómo funciona (pipeline híbrido, sin datos etiquetados)

Corrección de iluminación → dosel de banano (ExGR adaptativo + textura + morfología) →
**marco de siembra por autocorrelación FFT** (prior geométrico; da escala física aun sin
GSD) → **centros por transformada de distancia + watershed FUSIONADO con Fast Radial
Symmetry Transform** → agrupamiento en macollas (DBSCAN) → **procesado por tiles con
deduplicación** en bordes → georreferenciación a lon/lat.

## Resultados y límites (honestos)

En una plantación sintética con verdad de terreno (para validar la lógica):

- **Macollas: 94-96% de acierto** (F1 ~0.92-0.94). Este es el número fiable.
- **Pseudotallos: rango honesto** que contiene la verdad; el punto estimado puede desviarse
  ±30% porque separar plantas *dentro* de una macolla es intrínsecamente ambiguo (lo
  confirma la literatura).

**Lo sintético NO es banano real.** La línea base clásica funciona mejor con **GSD ≤ 3 cm/px**.
No resuelve por sí sola malezas de hoja ancha (platanillo/*Heliconia*), dosel adulto totalmente
cerrado, ni plantas subpíxel. Para esos casos y precisión comercial, usa el modelo entrenado.
Ver el guardarraíl de GSD y los avisos que emite el propio informe.

## Camino de deep learning (precisión comercial)

Cuando tengas tiles etiquetados:

```bash
pip install -e .[deep]
python deep/prepare_dataset.py --image orto.tif --out dataset/images --tile 1024
python deep/train_yolo.py --data deep/banana.yaml --epochs 100
python deep/infer_yolo.py --weights runs/segment/banano_seg/weights/best.pt --image tile.png
```

**Recursos reales para arrancar** (ver [`docs/estado-del-arte.md`](docs/estado-del-arte.md)):
- **ALSS-YOLO-Seg** — modelo abierto específico de banano UAV: https://github.com/helloworlder8/computer_vision
- **Dataset de Neupane et al. 2019** (figshare): https://figshare.com/s/62e391492b1be99515b4
- **Roboflow Universe** — busca "banana plantation / banana tree counting".

## Estructura

```
banano/          indices · segment · grid · radial · centers · mats · pipeline
                 geo (GeoTIFF) · ortho (tiles+dedup) · report (CSV/GeoJSON/HTML) · cli
scripts/         demo.py (demo con métricas) · run.py (una imagen) · make_example.py
deep/            camino YOLOv8-seg / ALSS-YOLO-Seg
docs/            estado del arte y referencias
tests/           pruebas
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
