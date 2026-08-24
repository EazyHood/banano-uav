# Entrenar sin usar tu ordenador

Todo el entrenamiento corre en los servidores de Kaggle. Tu PC sólo lanza un comando y se
puede apagar: ni sube imágenes, ni usa la GPU, ni tiene que quedarse encendido.

**Por qué Kaggle y no otra cosa:** es la única opción gratuita *para siempre* que corre
desatendida de verdad — 30 horas de GPU por semana que se renuevan solas los sábados, sin
tarjeta de crédito, y con `Save & Run All` la sesión sigue en el servidor aunque cierres el
navegador. Google Colab gratis **no** vale aquí: prioriza el uso interactivo y la ejecución
en segundo plano es de Colab Pro+ (de pago); si se te apaga el PC, pierdes el entrenamiento.

Y de paso ganas máquina: las **2× Tesla T4 de 16 GB** de Kaggle tienen el doble de memoria
que tu RTX 5060, que es justo lo que te obligaba a entrenar con lotes de 4.

---

## Preparación (una sola vez, ~10 minutos)

**1. Cuenta de Kaggle y verificación por teléfono.**
Crea la cuenta en <https://www.kaggle.com> y verifica el número en
<https://www.kaggle.com/settings> → *Phone verification*. No pide tarjeta, es un SMS. Sin
esta verificación el notebook no puede usar GPU ni salir a Internet, así que no podría ni
descargar los datos.

**2. Token de la API.**
En <https://www.kaggle.com/settings/api> → *Generate New Token*. Hoy ese botón te da un token
suelto que empieza por `KGAT_`, **no** un fichero. Va aquí, en un fichero de texto plano con el
token y nada más:

```
C:\Users\jhona\.kaggle\access_token
```

> ⚠️ **No lo metas en `kaggle.json`.** Ese fichero es del formato antiguo y espera un JSON
> (`{"username": "...", "key": "..."}`); si le pones el token pelado, el CLI no lo puede leer y
> falla sin decir por qué. El botón *Create Legacy API Key* de la misma página sí genera el
> `kaggle.json` de verdad, si prefieres esa vía.

**3. El paquete de línea de comandos** (ya está instalado en el venv del proyecto; si lo
necesitas fuera):

```bash
pip install kaggle
```

---

## Lanzar

```bash
python kaggle/lanzar.py
```

Sube el notebook y arranca la corrida. A partir de ahí puedes apagar el ordenador.

> **La clave de Roboflow ya está resuelta** y no hay que tocar la web. Vive en un **dataset
> privado** de tu cuenta, `jhonatandelriomejia/banano-uav-credenciales`, que el notebook lleva
> adjunto y lee al arrancar. Es el rodeo que documenta el propio Kaggle, y es el único que se
> puede montar entero desde la terminal: el CLI **no tiene ningún comando de secrets**
> (incidencia `Kaggle/kaggle-api#582`, abierta).
>
> Si algún día prefieres el mecanismo oficial, sigue funcionando como alternativa: en el
> notebook, `Add-ons → Secrets → Add a new secret`, etiqueta exacta `ROBOFLOW_API_KEY` y de
> valor tu *Private API Key* de <https://app.roboflow.com/settings/api> (la que **no** empieza
> por `rf_`). El notebook prueba primero el dataset y luego el secreto: basta con uno.
>
> 🔐 Ese dataset es privado, pero guarda una credencial: **no lo hagas público nunca**, y si
> abandonas el proyecto bórralo con
> `kaggle datasets delete jhonatandelriomejia/banano-uav-credenciales`.

## Ver el log de una corrida

```bash
python kaggle/lanzar.py --log
```

Usa `kernels logs`, que trae sólo el log. **No uses `--recoger` para diagnosticar**: eso se
baja la salida entera, que en una corrida fallida puede ser cientos de megas.

## Ver cómo va

```bash
python kaggle/lanzar.py --estado
```

## Recoger los pesos entrenados

```bash
python kaggle/lanzar.py --recoger
```

Los deja en `runs_cloud/kaggle/`.

---

## Lo que hace el notebook, por orden

1. Comprueba la GPU y **se niega a seguir si toca una P100** — la imagen de Kaggle trae
   PyTorch cu128, sin kernels de Pascal, así que `torch.cuda.is_available()` devuelve `True`
   y el entrenamiento muere en el primer lote. Siempre **T4 x2**.
2. Clona este repo desde GitHub.
3. Instala `ultralytics` (no viene en la imagen) y apaga toda la telemetría y los loggers
   externos, que en una sesión desatendida pueden quedarse esperando un login.
4. Lee tu clave de Roboflow del secreto.
5. **Descarga las imágenes de Roboflow directamente en el servidor de Kaggle.** Aquí está la
   clave de que tu PC no participe: `cloud/data_manifest.json` guarda el workspace, el
   proyecto y la **versión exacta** de cada fuente, así que la máquina remota reconstruye el
   mismo dataset que se midió en casa sin que tú subas un byte.
6. Reparte los datos por fincas y **comprueba que no hay fugas** antes de entrenar.
7. Barre la resolución de inferencia sobre todas las fincas retenidas.
8. Entrena.
9. Copia pesos y métricas a la salida y borra el resto.

## Los límites que importan

| | |
|---|---|
| Cuota | 30 h de GPU por semana, se renueva el sábado a medianoche UTC |
| Sesión | **12 h como máximo** |
| Disco de salida | 20 GB en `/kaggle/working`, se guarda entero |
| Máquina | 2× Tesla T4 (16 GB cada una), 4 núcleos, 29 GB de RAM |

⚠️ **Lo más caro que puede pasar:** si una corrida se pasa de las 12 horas, el guardado de
ficheros del final es *best effort* — a veces funciona y a veces pierdes los pesos. Por eso
el notebook se presupuesta 11 h y `cloud/train.py` reanuda desde `last.pt` si lo encuentra:
una corrida cortada se continúa en la siguiente sesión en vez de empezar de cero.

Referencias de tiempo, medidas en las corridas anteriores de este proyecto:
`v10` (60 épocas, 768 px) tardó 4,2 h; `v12` (74 épocas, 768 px) 10,3 h; y `v11` a 1024 px
iba a 637 s por época, o sea unas 17 h para 100 épocas — ése **no** cabe en una sesión y hay
que trocearlo, que es exactamente para lo que sirve la reanudación.
