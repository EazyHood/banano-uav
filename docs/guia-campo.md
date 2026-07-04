# Guía de campo — de tu dron al conteo

Guía práctica para operadores, agrónomos y empresas. Cubre cómo volar, generar el
ortomosaico y correr `banano-drone` para obtener un conteo fiable.

## 1. Planifica el vuelo

| Parámetro | Recomendado | Por qué |
|---|---|---|
| **Altura** | ≤ 120 m (según GSD objetivo) | Menor altura = mejor GSD = mejor detección |
| **GSD objetivo** | **≤ 2 cm/px** (ideal), ≤ 3 cm/px (aceptable) | Por encima de 3 cm/px el conteo individual pierde fiabilidad |
| **Solape frontal** | 75-80 % | Necesario para un buen ortomosaico |
| **Solape lateral** | 65-75 % | Ídem |
| **Hora** | 10:00-14:00, cielo estable | Minimiza sombras largas y cambios de luz |
| **Cámara** | RGB (mínimo); multiespectral (mejor) | RGB basta; NIR/RedEdge ayuda con maleza |

**Cómo calcular el GSD:** `GSD (cm/px) = (altura_m × tamaño_sensor_mm × 100) / (focal_mm × ancho_imagen_px)`.
La mayoría de apps de vuelo (DJI Pilot, Litchi, DroneDeploy) lo muestran al planificar.

## 2. Genera el ortomosaico

Con cualquiera de estos (el primero es libre):

- **OpenDroneMap / WebODM** (libre): `docker run opendronemap/odm` o la UI de WebODM.
- **Pix4Dfields**, **Agisoft Metashape**, **DroneDeploy** (comerciales).

Exporta un **GeoTIFF** (conserva la georreferencia y el GSD). Si solo tienes una imagen
RGB sin georreferencia, puedes usarla igual pasando `--gsd`.

## 3. Corre banano-drone

```bash
# GeoTIFF (detecta GSD y georreferencia solo):
banano-detect --input ortomosaico.tif --out resultados

# Imagen sin georreferencia (pasa el GSD):
banano-detect --input foto.jpg --gsd 2.5 --out resultados

# Con el modelo de deep learning (si tienes pesos entrenados):
banano-detect --input ortomosaico.tif --config config_modelo.yaml --out resultados
```

## 4. Interpreta los resultados

- **`informe.html`** — ábrelo en el navegador: resumen, mapa y avisos.
- **`plantas.geojson`** — cárgalo en **QGIS** o **Google Earth** (arrastra el archivo).
  Cada punto es una macolla, con lon/lat y nº de pseudotallos.
- **`plantas.csv`** — tabla para Excel/análisis.

**Qué número usar:** el conteo de **macollas** es el fiable (la unidad agronómica del
banano). Los pseudotallos se dan como **rango** porque separar plantas dentro de una
macolla es intrínsecamente ambiguo.

## 5. Valida en tu finca (recomendado)

Antes de confiar en el conteo a escala, **valida en 2-3 parcelas pequeñas**:
1. Cuenta a mano las macollas en una zona delimitada (un clic por macolla).
2. Corre `banano-drone` en esa zona.
3. Compara. Ajusta `--threshold` (baja si falta, sube si sobra) o el GSD.

Esto te da tu propio margen de error medido, honesto y defendible ante un cliente.

## 6. Casos difíciles y cómo mitigarlos

| Problema | Mitigación |
|---|---|
| Maleza de hoja ancha (platanillo/*Heliconia*) | Modelo entrenado con tus datos, o vuelo multiespectral |
| Dosel adulto muy cerrado (se solapan) | El conteo de macollas sigue bien; los pseudotallos, no |
| GSD > 3 cm/px | Vuela más bajo; el programa avisa y degrada a cobertura |
| Sombras fuertes | Vuela cerca del mediodía; la corrección de iluminación ayuda |
| Plantas muy jóvenes (subpíxel) | Vuela más bajo o usa el modelo entrenado |

## 7. Precisión: expectativa honesta

- La **línea base clásica** (sin entrenar) funciona sin datos etiquetados y es fiable a
  nivel de macolla con GSD ≤ 3 cm/px.
- El **modelo de deep learning** alcanza precisión muy alta en su distribución de
  entrenamiento; para tu finca real, **reentrénalo con tus imágenes etiquetadas** (ver
  [`../README.md`](../README.md), sección deep learning) para el máximo rendimiento.
- Ningún método garantiza <1 % de error sobre banano real sin validación con datos de
  campo. Usa el paso 5 para medir el tuyo.
