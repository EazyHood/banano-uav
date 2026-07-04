# Modelo entrenado con imágenes REALES de banano

`banano-drone` incluye **dos** modelos de deep learning listos para usar. Este documento
describe el modelo entrenado con **imágenes UAV reales** (el que una empresa usaría sobre
sus ortofotos, sin tener que entrenar nada).

## Modelos incluidos

| Modelo | Arquitectura | Datos de entrenamiento | Uso |
|---|---|---|---|
| `models/banano_real_v1.pt` | YOLOv8s (detección) | **UAV RGB REAL de banano** (dataset AI-BananaMapping, ~14 000 tiles) | Detectar/contar plantas de banano en ortofotos reales |
| `models/banano_seg_synth_v1.pt` | YOLOv8s-seg (segmentación) | Sintético con verdad de terreno | Segmentación de macollas; benchmark reproducible |

## Origen de los datos reales

- **Dataset:** DS-v1 de *"UAV-based detection of Black Sigatoka in banana crops"* (AI-BananaMapping).
- **Fuente:** Zenodo [record 20945958](https://zenodo.org/records/20945958), **licencia CC-BY-4.0**.
- **Contenido:** imágenes UAV RGB reales (nadir) de cultivos de banano, tileadas a 1024 px,
  con anotaciones YOLO por planta. ~14 180 train / 4 645 val / 4 611 test.
- Las cajas marcan plantas de banano (etiquetadas en el contexto de Sigatoka Negra, una
  enfermedad casi universal en banano comercial); se usan como **una sola clase: banano**.

## Rendimiento sobre imágenes reales (split de test retenido)

Medido sobre **4 611 tiles UAV reales que el modelo nunca vio** (6 017 plantas):

| Métrica | Valor |
|---|---|
| **mAP50** | **0.411** |
| mAP50-95 | 0.164 |
| Precisión | 0.472 |
| Recall | 0.458 |

Reproducible: `python deep/eval_real.py --weights models/banano_real_v1.pt --data realdata/DS-v1/ds-v1/banano_real.yaml`.

**Cómo leer estas cifras (honestidad):** sobre banano REAL, detectar plantas es intrínsecamente
más difícil que sobre sintético. Un mAP50 de ~0.41 significa que el modelo localiza una parte
sustancial de las plantas de banano en imágenes reales, con precisión y recall en torno al
46-47 %. Es un detector real **funcional**, no perfecto — y es la clase de cifra que reporta
la literatura sobre banano real (no los ~99 % que solo se ven en sintético). Para tu finca,
**valida en una parcela** y afina con tus imágenes si necesitas más (el pipeline lo soporta).

**Escala de las imágenes:** el dataset es UAV de **baja altitud (~5-12 m, escala de planta
individual)**. El modelo rinde mejor a esa escala. Para ortomosaicos de gran altura (plantas
pequeñas), valida el GSD y, si hace falta, reentrena/afinar con tiles a tu altura de vuelo.

## Cómo usarlo

```bash
pip install -e .[deep]
banano-detect --input tu_ortofoto.tif --config config.example.yaml --out resultados
# en el YAML: model_weights: models/banano_real_v1.pt
```

O en código:

```python
from banano import PipelineConfig, Raster, process_orthomosaic
cfg = PipelineConfig(gsd_cm=3.0, model_weights="models/banano_real_v1.pt", model_conf=0.35)
raster = Raster("tu_ortofoto.tif")
res = process_orthomosaic(raster, config=cfg)
print(res.summary())
```

## Alcance honesto

- El modelo aprendió la apariencia de plantas de banano en imágenes **UAV reales**, así que
  generaliza a ortofotos reales mucho mejor que el modelo sintético.
- Fue entrenado sobre plantaciones concretas (dominio del dataset). En condiciones muy
  distintas (otra variedad, otra región, GSD muy diferente), **valida en una parcela** y, si
  hace falta, reentrena/afinar con unas pocas imágenes tuyas (`deep/train_yolo.py`).
- Es un detector de plantas de banano; para conteo, el pipeline deduplica en el solape de
  tiles y reporta el total.
