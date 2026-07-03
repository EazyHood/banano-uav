# Estado del arte — detección de banano con drones (síntesis)

Resumen de la investigación que fundamenta el diseño de este programa. Compilado a
partir de una búsqueda multi-fuente (jul-2026). Las cifras provienen de los papers
citados; donde el PDF estaba restringido se marca como no verificado.

## Por qué el banano es más difícil que palma/aguacate/mango

- **Crecimiento en macolla (mat + hijuelos/suckers).** Madre + hijuelos crecen
  pegados y comparten base; desde nadir forman una masa foliar continua sin borde
  claro. Distinguir madre vs. hijo es una decisión metodológica, no un hecho visual.
- **Copa irregular, no circular.** Hojas grandes, largas y arqueadas que se solapan
  con las vecinas; no hay la firma geométrica limpia de palma/frutales.
- **Contraste:** palma aceitera y huertos (mango, aguacate) se plantan a marco
  regular, con copas ~circulares, compactas y **separadas** → hasta detectores
  simples o *template matching* alcanzan mAP/F1 del 90-99 %. El banano rompe las tres
  ventajas (ni separación visual, ni copa circular, ni "un árbol = un objeto").
- **Crecimiento asíncrono:** conviven fenologías (tamaños de copa muy distintos) →
  exige multiescala y complica un criterio único de "qué cuenta como planta".
- **Dependencia de la altura de vuelo (GSD):** recall cae de **96 % @1.78 cm/px (40 m)**
  a **76 % @2.54 cm/px (60 m)** incluso con deep learning (Neupane 2019).

## Métodos (de más directo a más sofisticado)

1. **Detección por caja (YOLO).** De facto: trocear ortomosaico → detectar → reproyectar
   a GPS. Fundacional: Neupane 2019 (Faster R-CNN + Inception-v2). Hoy con Ultralytics YOLO.
2. **Pre-realce vegetativo (barato y efectivo).** Linear Contrast Stretch, HSV Synthetic
   Color Transform, y **Triangular Greenness Index `TGI = G − 0.39R − 0.61B`** (implementado
   en `banano/indices.py`). Sube el recall al resaltar hojas jóvenes.
3. **Segmentación de instancias (recomendado para la macolla).** **ALSS-YOLO-Seg** (2024),
   YOLO-seg de una etapa específico para banano en UAV, **mAP50 (máscara) 85.8 %**, ~1.8 M
   parámetros, **código abierto**. Alternativa mantenida: YOLOv8/YOLO11-seg de Ultralytics.
4. **Anotación asistida con SAM2** para reducir el coste de etiquetado de instancias.
5. **Conteo por mapa de densidad** (CSRNet/P2PNet, anotación por puntos) — estándar en
   cultivos densos/solapados; encaja con la macolla, pero **no hay trabajo publicado que
   lo aplique a banano** (sería transferencia de método).
6. **Georreferenciación + deduplicación entre tiles** (patrón de Neupane).

## Cómo lo aborda este programa (sin datos etiquetados)

Pipeline híbrido explicable: corrección de iluminación → dosel (ExGR adaptativo + textura
+ morfología) → **marco de siembra por autocorrelación** (prior geométrico) → **centros por
distancia+watershed fusionados con FRST** → agrupamiento en macollas (DBSCAN). Reporta el
conteo en la **unidad agronómica correcta (macolla)** más un **rango honesto** de pseudotallos
y la **cobertura de dosel**, con guardarraíl de GSD. Es una línea base; para máxima precisión,
el camino de deep learning (`deep/`).

## Cifras de referencia (banano, plantas individuales)

| Trabajo | Método | Precisión reportada |
|---|---|---|
| Neupane et al. 2019 (Tailandia, ~2.695 plantas GT) | Faster R-CNN + Inception-v2 sobre ortomosaico | 40 m: **recall 96.4 % / prec. 99.3 %**; 50 m: 85.1 / 97.9; 60 m: 75.8 / 98.5; 40+50 m combinado: **recall 99 %** |
| ALSS-YOLO-Seg 2024 (3.880 img, DJI Phantom, 5/8/12 m) | YOLO-seg 1 etapa (~1.8 M par.) | **mAP50 máscara 85.8 %** (supera a YOLOv8-seg ~1 %) |
| YOLOv5n-BPCount 2024-25 (conteo de plántulas) | YOLOv5n ligero | cifras no verificadas (PDF restringido) |

Cultivos "fáciles" para dimensionar la brecha: palma **F1 ≈ 92.8 %**; mango **mAP 86.4 %**
(UAV) / prec. 96.1 % en campo; aguacate: detección fiable de copas individuales.

## Datasets y modelos públicos (para el camino deep learning)

- **Neupane 2019 — datos de banano UAV** (figshare): https://figshare.com/s/62e391492b1be99515b4
- **ALSS-YOLO-Seg — código** (banano, segmentación): https://github.com/helloworlder8/computer_vision
- Roboflow Universe — buscar "banana plantation / banana tree counting" (varios datasets anotados).
- Banana Fusarium/Xanthomonas wilt (multiespectral, para enfermedad): SciDB, Nature Sci. Data 2025.

## Fuentes principales

- Neupane et al. 2019, *Deep learning based banana plant detection and counting… UAV RGB*, PLOS ONE — https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0223906
- *Iterative Optimization Annotation Pipeline and ALSS-YOLO-Seg…*, arXiv 2410.07955 (2024) — https://arxiv.org/html/2410.07955v1
- *Detecting Banana Plantations… Aerial Photography and U-Net*, Applied Sciences 2020 — https://www.mdpi.com/2076-3417/10/6/2017
- *Detection of Banana Plants Using Multi-Temporal Multispectral UAV Imagery*, Remote Sensing 2021 — https://www.mdpi.com/2072-4292/13/11/2123
- *Comparative Analysis of Deep Learning Models for Banana Plant Detection* (VARI mejora YOLOv3), CMC 2025 — https://www.techscience.com/cmc/v84n3/63198/html
- Loy & Zelinsky, *A Fast Radial Symmetry Transform for Detecting Points of Interest* — https://link.springer.com/chapter/10.1007/3-540-47969-4_24
- *Verification of color vegetation indices…* (ExG, ExGR, Otsu) — https://www.sciencedirect.com/science/article/abs/pii/S0168169908001063
- *GLCM texture based crop classification using low altitude remote sensing* — https://peerj.com/articles/cs-536/
