# Modelo multi-finca: qué mide, qué promete y qué no

`models/banana_multifarm_v10.pt` es el modelo recomendado del repo para ortofotos reales.
Este documento existe para que puedas decidir **si te sirve** antes de volar un dron, y para
que puedas **reproducir cada número** de aquí en tu máquina.

- Arquitectura: YOLO11m (detección), entrenado a **768 px**.
- Datos: ~5.100 imágenes UAV de **5 fincas independientes** de banano/plátano (Roboflow,
  CC-BY), curadas con chequeo de solape (`dhash`) para que ningún fork de la misma finca
  contaminara el test.
- SHA256 (16): `8e4b7d1f7eba3651` — es **exactamente** el fichero con el que se midió todo
  lo que sigue. Si tu copia da otro hash, las cifras de abajo no son las tuyas.

## Los 4 protocolos, de más fácil a más honesto

Casi todo el mundo publica el primero. El último es el que de verdad predice lo que te va a
pasar a ti.

| # | protocolo | qué es | fichero |
|---|---|---|---|
| 1 | `samefarm` | test retenido de la finca original | `realdata/t768.yaml` |
| 2 | `seenfarms` | test retenido de las **otras 4 fincas del entrenamiento** | `realdata/holdout.yaml` |
| 3 | `newfarms` | 3 fincas que v10 **no vio jamás** (sí las vio v12) | `realdata/holdout_newfarms.yaml` |
| 4 | `armah` | finca de Ghana, paisaje mixto con palma aceitera, que **ningún modelo vio** | `realdata/holdout_armah.yaml` |

## Detección (mAP), imgsz 768

| protocolo | mAP50 | mAP50-95 | precisión | recall |
|---|---|---|---|---|
| samefarm | 0.861 | 0.469 | 0.858 | 0.772 |
| seenfarms | 0.746 | 0.363 | 0.797 | 0.673 |
| newfarms | 0.005 | 0.004 | 0.012 | 0.028 |
| **armah (finca nueva de verdad)** | **0.172** | 0.081 | 0.551 | **0.139** |

**Léelo así:** dentro de las fincas que el modelo conoce, funciona (mAP50 0.75-0.86). En una
finca nueva encuentra **~14 de cada 100 plantas**. No es un matiz: es la diferencia entre
usarlo tal cual y tener que afinarlo con imágenes tuyas.

El 0.005 de `newfarms` no es un error de medida — es el mismo fenómeno llevado al extremo:
esas 3 fincas se parecen tan poco a las del entrenamiento que el modelo queda ciego. La
generalización entre fincas **no es estable**: 0.005 en unas, 0.172 en otra.

## Conteo (lo que de verdad le importa a una finca)

Dos columnas, y la diferencia entre ellas importa más que los números:

- **error del total**: `|predichas − reales| / reales` sobre todo el conjunto.
- **MAPE**: error medio **por imagen**. Es el que te dice si puedes fiarte de una parcela.

Ambos con el umbral de confianza elegido por **calibración cruzada en 2 pliegues** (se
calibra en una mitad y se mide en la otra, y al revés): nadie eligió el umbral mirando lo
que luego se reporta.

| protocolo | imágenes | reales | contadas | error del total | MAPE por imagen |
|---|---|---|---|---|---|
| samefarm | 50 | 1.166 | 1.145 | **1,80 %** | 0,16 |
| seenfarms | 99 | 1.113 | 1.020 | 8,36 % | 0,57 |
| newfarms | 62 | 5.426 | 650 | 88,02 % | 0,93 |
| armah | 62 | 1.378 | 309 | 77,58 % | 0,63 |

> ⚠️ **El error del total engaña, y mucho.** Suma sobre- y sub-conteos, así que se compensan
> solos. En las mediciones de este repo hay un caso donde el total sale al **0,07 %** mientras
> el error medio por imagen es del **65 %**: el modelo se equivoca en casi todas las parcelas,
> pero los errores se cancelan al sumarlos. **Si vas a decidir por parcela, mira el MAPE.**

## El umbral de confianza hay que calibrarlo en tu finca

No existe un valor universal, y el repo prefiere decírtelo a inventarse uno. Con el mismo
modelo, error del total según el umbral:

| conf | samefarm | seenfarms |
|---|---|---|
| 0,10 | **1,8 %** | 48,5 % |
| 0,15 | 11,4 % | 18,4 % |
| 0,20 | 18,4 % | 5,8 % |
| 0,25 | 23,3 % | **2,2 %** |
| 0,35 (el default histórico del paquete) | 33,0 % | 14,9 % |

Por eso `config.example.yaml` documenta un **rango** (0,10-0,25) y no una cifra mágica.
Calibrar cuesta una sola pasada sobre 30-50 tiles etiquetados tuyos:

```bash
python deep/eval_count.py --weights models/banana_multifarm_v10.pt \
    --data tu_finca.yaml --imgsz 768 --name mi_finca_count
# lee "honesto" en real_eval/mi_finca_count.json: ese es tu umbral y tu error esperado
```

## Por qué NO se publica el modelo v12

Se entrenó un v12 (YOLO11 de 26 M de parámetros) añadiendo 3 fincas al entrenamiento, y se
midió en los mismos 4 protocolos con la misma suite. **No se publica**, y esta es la razón,
con las cifras delante:

| protocolo | v10 (publicado) | v12 | ¿v12 mejora? |
|---|---|---|---|
| samefarm | 0.861 | 0.898 | sí |
| seenfarms | 0.746 | 0.726 | **no** |
| newfarms | 0.005 | 0.787 | sí, pero **v12 entrena con esas fincas** |
| armah (nadie la vio) | **0.172** | 0.105 | **no** |

En la única finca que ninguno de los dos vio, v12 es peor en mAP (0.105 vs 0.172), en
precisión (0.376 vs 0.551) y en recall (0.109 vs 0.139). Sus victorias están todas en fincas
que lleva dentro. Publicarlo sería anunciar como generalización lo que es memoria del
entrenamiento. Además su entrenamiento quedó interrumpido (mejor época 64 de 80 previstas).

Las mediciones de v12 se conservan en `real_eval/v12_*.json` para que la decisión sea
auditable, no una opinión.

## Reproducir todo esto

```bash
# los 4 protocolos, mAP + conteo, del modelo publicado
python deep/eval_v12_suite.py --weights models/banana_multifarm_v10.pt --tag v10

# un protocolo suelto
python deep/eval_v12_suite.py --weights models/banana_multifarm_v10.pt --tag v10 --solo armah
```

Cada corrida deja su registro en `real_eval/<tag>_<protocolo>_{map,count}.json` con fecha,
ruta de los pesos, SHA del modelo, resolución y — en los de conteo — los conteos por umbral
y por imagen, para que recalibrar después no cueste otra pasada de GPU.

## Cómo usarlo

```bash
pip install -e .[deep]
banano-detect --input tu_ortofoto.tif --gsd 3.0 \
    --weights models/banana_multifarm_v10.pt --model-conf 0.10 --model-imgsz 768 \
    --out resultados
```

`--model-imgsz 768` importa: el modelo se entrenó a 768 px y, sin esa opción, el pipeline
infiere a la resolución del tile (1024 por defecto).

## Alcance honesto, en una frase

Si tu finca se parece a las del entrenamiento, cuenta el total con un error del 2-8 % una vez
calibrado el umbral. Si es una finca nueva de verdad, **no lo uses tal cual**: encuentra ~1 de
cada 7 plantas. Etiqueta 100-200 tiles tuyos y afínalo (`deep/train_yolo.py`) — es la
diferencia entre 0.17 y 0.75 de mAP50.
