# Por qué falla en una finca nueva, medido

*2026-08-23.*

El repo publica que en una finca que ningún modelo ha visto el detector encuentra ~14 de
cada 100 plantas (mAP50 0.172, recall 0.139 sobre `armah`, Ghana). La explicación que se
venía dando era el *domain gap* visual: otro suelo, otra luz, otra variedad, palma aceitera
de fondo. Al medirlo, eso es sólo una parte, y no la primera.

Este documento reúne lo que se midió, con las herramientas que lo reproducen.

---

## 1. El problema es, antes que nada, de ESCALA

`deep/scale_audit.py` mide de qué tamaño aparece una planta en cada fuente, a la resolución
a la que el modelo mira (`imgsz`). A 768 px:

| fuente | planta (px) | cajas/img | papel |
|---|---:|---:|---|
| `newfarms/m2` | 10 | 126 | train desde v12 |
| `extra/plantas_platano` | 16 | 20 | train |
| `extra/plantas_jovenes_80m1` | 17 | 16 | train |
| `extra/plantas_jovenes_50m` | 21 | 8 | train |
| `newfarms/elliot` | 31 | 98 | train desde v12 |
| **`newfarms/armah`** | **45** | 22 | **holdout ciego** |
| `extra/etiquetasnuevas` | 48 | 7 | train |
| `count_banana_plants` | 169 | 12 | train (finca original) |
| `newfarms/lasuiza` | 333 | 5 | train desde v12 |

**33× entre los extremos.** Y lo que decide el aprendizaje no son las imágenes sino las
cajas. Sobre el pool de entrenamiento real de v10:

```
hasta  16 px   39.6 %
hasta  17 px   81.2 %   <- el 81 % del entrenamiento son plantas de 16-17 px
hasta  21 px   84.7 %
hasta  46 px   85.5 %   <- entre 21 y 46 px hay menos del 1 %
hasta 174 px  100.0 %
```

`armah` tiene sus plantas a 45 px: justo en la banda más vacía del entrenamiento. Eso
explica un recall de 0.139 mejor que "el suelo de Ghana es distinto".

Y explica lo que parecía contradictorio: **v12, entrenado con más fincas, empeoró `armah`**
(0.105 frente a 0.172). Las fincas que añadió están en los extremos —`m2` a 10 px y
`lasuiza` a 333 px—, así que diluyeron todavía más la banda intermedia donde `armah` vive.
Más datos, sí, pero de las escalas equivocadas.

### La comprobación que lo decide, y que no cuesta GPU

Si manda la escala, cambiar la resolución de entrada tiene que mover el resultado sin
tocar un solo peso. `cloud/scale_sweep.py` lo mide:

| imgsz | mAP50 | recall |
|---:|---:|---:|
| 320 | 0.0001 | 0.016 |
| 512 | 0.0146 | 0.033 |
| 640 | 0.0848 | 0.092 |
| **768** | **0.1724** | **0.139** ← el que usa el repo |
| 896 | 0.2573 | 0.207 |
| **1024** | **0.2847** | **0.229** ← óptimo |
| 1152 | 0.2708 | 0.220 |
| 1280 | 0.2694 | 0.215 |
| 1536 | 0.2470 | 0.195 |

**+65 % de mAP50 y +65 % de recall, gratis.** El valor a 768 reproduce el fichero
`real_eval/v10_armah_map.json` (0.1724 frente a 0.1723), así que el instrumento está
validado. Máximo interior con campana a los dos lados: no es un artefacto de borde.

En `holdout_newfarms`, donde el número publicado era un 0.0053 que parecía "el modelo no
sirve para nada", pasa lo mismo pero más fuerte: **0.0053 → 0.1219 a 1280 px, ×23** con los
mismos pesos.

> **Una predicción fallada, que se deja escrita.** Antes de medir se predijo que el óptimo
> estaría *abajo* (~300 px), porque a esa resolución las plantas de `armah` se verían del
> tamaño dominante del entrenamiento. Salió al revés: el óptimo está *arriba*. Igualar el
> tamaño relativo no basta — encoger la imagen destruye los píxeles de textura con los que
> se reconoce una roseta. A 320 px la planta tiene el tamaño "correcto" y el mAP es 0.0001.

> **Aviso de método.** Ese 1024 se eligió mirando `armah`, que es el holdout. Afinar un
> hiperparámetro sobre el conjunto con el que luego se presume es el mismo error que este
> repo ya corrigió una vez en `eval_count.py`. El valor que se publique tiene que salir del
> promedio LOFO: `cloud/scale_sweep.py --todas-las-fincas`.

---

## 2. Una de las cifras publicadas está contaminada

`deep/leak_audit.py` compara todas las fuentes por MD5 del contenido, no por nombre.
Resultado:

**`extra/prueba2rgb` y `extra/etiquetasnuevas` son el mismo dataset.** Las 505 imágenes son
idénticas byte a byte. Están subidas a dos workspaces distintos de Roboflow
(`entrenamiento-alterno-dgpgp` y `tesis-hpmog`), y como Roboflow renombra cada exportación
con un hash propio, el dedup "por stem" que se hizo en su día no podía verlas.

Consecuencias medidas:

- v10 y v12 entrenaron ese material **dos veces**, con el doble de peso del previsto.
- `realdata/holdout.yaml` valida en parte con `extra/prueba2rgb/test`. Sus **25 imágenes
  están el 100 % dentro del entrenamiento**, vía `etiquetasnuevas/train+valid`. Son 25 de
  las 99 del protocolo: **una cuarta parte del 0.746 publicado se mide sobre imágenes ya
  vistas**.

Los otros tres protocolos están limpios (0 % de fuga, verificado contra el train de v10 y
el de v12): `t768`, `holdout_newfarms` y `holdout_armah`. **La cifra honesta del README, el
0.172 de finca nueva, se sostiene entera.**

Hay un segundo par que todavía no ha hecho daño pero lo haría: `extra/platano-lasuiza` y
`newfarms/lasuiza` son dos versiones (v3 y v2) del mismo proyecto de Roboflow, con 84
imágenes compartidas. Hoy sólo se usa una; el manifiesto lo avisa por si alguien añade la otra.

---

## 3. La diversidad real es menor de lo que dice el recuento

El README habla de "~5.100 imágenes UAV de 5 fincas independientes". Contando **disparos
originales** en vez de ficheros —Roboflow exporta varias copias augmentadas de cada foto—:

| fuente | imágenes | disparos originales |
|---|---:|---:|
| `extra/plantas_jovenes_80m1` | 2.159 | 407 |
| `extra/plantas_platano` | 1.600 | 304 |
| `extra/plantas_jovenes_50m` | 378 | 72 |
| `count_banana_plants` | 702 | 451 |

Las tres primeras aportan **el 81 % de las cajas** del entrenamiento de v10 y salen de
**783 fotos** de un único operador (workspace `agromatica2025`, tres altitudes de vuelo:
50 m, 80 m y 100 m). Sumado a que `prueba2rgb` es una copia de `etiquetasnuevas`, las
fuentes realmente independientes del entrenamiento de v10 son **tres**, no cinco.

---

## 4. Lo que se corrigió a raíz de esto

- **`splits/lofo_*.yaml`** — protocolo Leave-One-Farm-Out. La cifra honesta salía de una
  sola finca de 62 imágenes, donde un cambio de receta puede subir o bajar por azar del
  reparto. Ahora hay 7 particiones, cada finca retenida entera una vez, y **6.150 imágenes
  de validación en vez de 62**. Verificado con `leak_audit`: 0 % de fuga en las siete.
  Las fincas se agrupan por origen, no por carpeta, para que tres vuelos del mismo operador
  no cuenten como tres fincas.
- **`max_det` de 300 a 1000** — el valor por defecto de ultralytics topaba el recall al 50 %
  por construcción en la imagen del holdout que tiene 600 plantas reales.
- **La receta `escala`** (`cloud/train.py`) usa `scale` como tupla absoluta `(0.25, 2.5)`,
  que cubre 10×; como escalar sólo daba 0.4×-1.6×. No usa `copy_paste`: en ultralytics
  8.4.117 es exclusivo de segmentación y aquí las etiquetas son cajas.
- **La receta de control es la de verdad.** El modelo publicado (v10) se entrenó con
  `degrees: 0`, `flipud: 0`, `scale: 0.5` — los defaults de YOLO. La "augmentación cenital"
  que describe el docstring de `deep/train_v12.py` se aplicó sólo a v12, que no se publicó.
  El modelo que la gente descarga **nunca vio una planta rotada ni volteada**, en una vista
  que es invariante a la rotación. Es margen sin explotar.

---

## 5. Lo que esto NO arregla

Conviene decirlo antes de que suene mejor de lo que es. Subir la resolución mejora el mAP un
65 %, pero el **conteo** en finca nueva sigue sin ser utilizable: sobre las 1.378 plantas
reales de `armah`, a 1024 px y conf 0.10 el modelo predice 295, un **-78,6 %**. A 768 px era
-84,5 %. Es una mejora real y no cambia la conclusión: **hoy el modelo no sirve como contador
en una finca nueva sin etiquetar algo de esa finca.**

La vía con números publicados más sólidos para cerrar ese hueco es el *few-shot*: en el
NTIRE 2025 Cross-Domain Few-Shot Object Detection Challenge, pasar de 1 a 5 imágenes
etiquetadas del dominio destino casi dobla el mAP del baseline (15.3 → 28.4), y de 5 a 10 el
retorno ya decae. Traducido a este proyecto: **5-10 tiles etiquetados de la finca del cliente**
valen más que cualquier cosa que se pueda hacer sin ellos.

Y hay un precedente con el mismo cultivo que apunta en la misma dirección: Neupane et al.
(PLOS ONE, 2019) contaron banano con UAV en la **misma finca** cambiando sólo la altura de
vuelo — 40 m: 96,4 % de recall; 50 m: 85,1 %; 60 m: 75,8 %. Veinte puntos de recall por
cambiar la altura, sin cambiar de finca. Los autores anotan la causa exacta del problema de
la macolla: al subir, *dos plantas juntas se detectaban como una sola*. La dificultad de la
macolla no es una propiedad fija del cultivo: es una función de la escala en píxeles.

---

## Cómo reproducir todo esto

```bash
python deep/scale_audit.py --data realdata/v10_multifarm.yaml --imgsz 768
python deep/leak_audit.py
python deep/leak_audit.py --train realdata/v10_multifarm.yaml --val realdata/holdout.yaml
python cloud/scale_sweep.py --todas-las-fincas
```

El barrido necesita GPU; lo demás corre en CPU en segundos. Para hacerlo sin tocar el PC
propio, está [`kaggle/README.md`](../kaggle/README.md).
