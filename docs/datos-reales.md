# Datos reales: procedencia, licencias y protocolo de evaluación

Registro de TODOS los datasets reales usados para entrenar y evaluar los modelos
de este repo, con su licencia y su papel exacto. Sin esto no hay cifra creíble.

## Fincas de entrenamiento (modelo multi-finca v12)

| Finca (alias) | Fuente | Licencia | Papel | Contenido |
|---|---|---|---|---|
| count_banana_plants | [Roboflow: count-banana-plants](https://universe.roboflow.com/count-banana-plants) | CC BY 4.0 | train + val | 702 imgs aéreas nadir, caja por planta (finca original) |
| plantas_jovenes_50m | Roboflow (comunidad) | CC BY 4.0 | train | vuelo a ~50 m, plantas jóvenes |
| plantas_jovenes_80m1 | Roboflow (comunidad) | CC BY 4.0 | train | vuelo a ~80 m |
| plantas_platano | Roboflow (comunidad) | CC BY 4.0 | train | plantación de plátano |
| prueba2rgb | Roboflow (comunidad) | CC BY 4.0 | train | ortomosaico RGB |
| etiquetasnuevas | correcciones propias (anotador en vivo) | propia | train | tiles corregidos a mano |
| **elliot** (nueva 2026-08) | [Roboflow: banana-tree-detection](https://universe.roboflow.com/elliot-su-mv7fl/banana-tree-detection) | CC BY 4.0 | train | 396 imgs nadir, plantación joven sobre suelo desnudo con acolchado, ~40k cajas |
| **m2** (nueva 2026-08) | [Roboflow: plantain-detection](https://universe.roboflow.com/m2-eewev/plantain-detection) | CC BY 4.0 | train | 197 imgs, plátano disperso en ladera con fondo heterogéneo, ~25k cajas |
| **lasuiza** (nueva 2026-08) | [Roboflow: platano-lasuiza](https://universe.roboflow.com/daniel-f-tovar-r-s-workspace/platano-lasuiza) | CC BY 4.0 | train (solo train: etiquetas flojas) | 153 imgs originales (deduplicadas de 366 con augmentación), plátano adulto, Colombia |

## Fincas de evaluación (nunca entrenadas)

| Finca | Fuente | Licencia | Papel |
|---|---|---|---|
| **armah** (Ghana) | [Roboflow: detecting-plantain-crops](https://universe.roboflow.com/michael-armah/detecting-plantain-crops) | CC BY 4.0 | **Holdout de finca NUNCA vista** (62 imgs, las 3 particiones completas). Ningún modelo entrena con ella. |
| tests de las fincas de train | (las mismas de arriba) | CC BY 4.0 | Holdout de "imágenes no vistas de fincas vistas" (`realdata/holdout.yaml` + `realdata/holdout_newfarms.yaml`) |

## Datasets evaluados y DESCARTADOS (2026-08-09)

- `conteobanano/proyecto_plantas-banano`: dosel lleno de banano con ~5 cajas por
  imagen → **sub-etiquetado severo**; entrenar con él enseña falsos negativos.
- `karachi-university/plant-count`: plántulas en surcos (Sindh, Pakistán); no se
  pudo confirmar visualmente que el cultivo sea banano → fuera por prudencia.
- DS-v1/DS-v2 (Zenodo AI-BananaMapping): etiquetas de *enfermedad* a baja altitud,
  no de conteo por planta (lección de julio-2026: mAP50 0.41 entrenando contra el
  objetivo equivocado). Se conserva la atribución en el modelo v1 histórico.

## Curación anti-fugas (aplicada a las fincas nuevas)

1. **Dedup por stem de Roboflow**: las exportaciones con augmentación reparten
   copias del mismo original entre train/valid/test → se colapsa por el nombre
   base (`*_jpg.rf.<hash>`), conservando una sola copia (prioridad: train).
   Resultado: lasuiza 366→153, armah 148→62, elliot −20.
2. **dhash (hamming ≤5)** de cada imagen nueva contra las ~5 800 existentes
   (train de v10 + todos los holdouts): 2 tiles de lasuiza ya presentes en el
   pool de entrenamiento fueron retirados.
3. Los descartes se mueven a `realdata/newfarms/_dupes/` (no se borran).

## Protocolo de evaluación (tres niveles de honestidad)

1. `realdata/t768.yaml` — test de la finca original (mismo dominio): techo del modelo.
2. `realdata/holdout.yaml` + `holdout_newfarms.yaml` — imágenes no vistas de
   fincas vistas: lo que mejora al añadir datos de una finca al entrenamiento.
3. `realdata/holdout_armah.yaml` — **finca nunca vista**: lo que le pasa a una
   empresa que suelta el modelo en su finca sin reentrenar. La cifra dura.

Cada evaluación se registra con `deep/eval_record.py` (mAP/P/R) y
`deep/eval_count.py` (error de conteo, la métrica de negocio) en `real_eval/*.json`
con fecha, sha de pesos y datos usados.
