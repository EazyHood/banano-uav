"""Genera el notebook de Kaggle que entrena el modelo fuera del PC del autor.

El .ipynb es JSON y editarlo a mano es incómodo y propenso a romperse; se genera desde
aquí para que el contenido de las celdas se pueda leer y revisar como código normal.

    python kaggle/build_notebook.py

Escribe kaggle/entrenar/banano-entrenar.ipynb
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DESTINO = Path(__file__).resolve().parent / "entrenar" / "banano-entrenar.ipynb"

REPO = "https://github.com/EazyHood/banano-uav.git"
RAMA = "cloud-training"

CELDAS: list[tuple[str, str]] = [
    (
        "markdown",
        """# banano-uav — entrenamiento en Kaggle

Entrena el detector de plantas de banano **sin usar el PC de casa**. Todo ocurre en los
servidores de Kaggle: el código se clona de GitHub y las imágenes se bajan de Roboflow
directamente aquí dentro.

**Antes de lanzarlo, dos cosas una sola vez:**

1. `Add-ons -> Secrets` -> añade el secreto `ROBOFLOW_API_KEY` con tu Private API Key de
   Roboflow (la de <https://app.roboflow.com/settings/api>, **sin** el prefijo `rf_`).
2. Acelerador: **GPU T4 x2**. No elijas P100: la imagen actual de Kaggle trae PyTorch cu128,
   que no incluye kernels de Pascal, así que `torch.cuda.is_available()` dice `True` y el
   entrenamiento revienta en el primer lote con `cudaErrorNoKernelImageForDevice`.

Después: `Save Version -> Save & Run All (Commit)`. Se puede cerrar el navegador y apagar
el ordenador; la sesión sigue en el servidor. Límite duro: **12 horas**.""",
    ),
    (
        "code",
        """# --- 1. Qué máquina nos ha tocado -----------------------------------------------
import subprocess, sys, time, os, json
T0 = time.time()
LIMITE_H = 11.0   # Kaggle corta a las 12 h y al pasarse el guardado de ficheros es "best effort"

import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {p.name}  {p.total_memory/1024**3:.1f} GB  sm_{p.major}{p.minor}")
    nombre = torch.cuda.get_device_name(0)
    if "P100" in nombre:
        raise RuntimeError(
            "GPU P100 detectada. La imagen de Kaggle no trae kernels de Pascal y el "
            "entrenamiento moriria en el primer lote. Cambia el acelerador a 'GPU T4 x2'."
        )
    # Prueba real: is_available() puede mentir, una operacion CUDA no.
    torch.zeros(8, device="cuda").sum().item()
    print("  operacion CUDA de prueba: OK")

# Internet: sin el, ni se clona el repo ni se bajan las fotos.
import socket
try:
    socket.setdefaulttimeout(10)
    socket.gethostbyname("github.com")
    hay_red = True
except OSError:
    hay_red = False
print("internet:", "OK" if hay_red else "NO")

# Kaggle ACEPTA enable_gpu/enable_internet en el metadata y luego los DENIEGA en
# ejecucion si la cuenta no tiene el telefono verificado. Sin este aviso el sintoma
# que ves es un error de git ("Could not resolve host") que no dice nada del motivo.
if not torch.cuda.is_available() or not hay_red:
    raise RuntimeError(chr(10).join([
        "=" * 70,
        "Este notebook pidio GPU e Internet y Kaggle no los ha concedido.",
        "",
        "Causa casi segura: falta la VERIFICACION POR TELEFONO de la cuenta.",
        "Es lo unico que Kaggle exige para desbloquear las dos cosas a la vez,",
        "y es un SMS: no pide tarjeta.",
        "",
        "    https://www.kaggle.com/settings  ->  Phone verification",
        "",
        "Despues vuelve a lanzarlo. Se aborta aqui a proposito: seguir sin GPU",
        "gastaria las 12 h de sesion para nada.",
        "=" * 70,
    ]))
""",
    ),
    (
        "code",
        f"""# --- 2. Codigo del proyecto ------------------------------------------------------
REPO = {REPO!r}
RAMA = {RAMA!r}
WORK = "/kaggle/working"
SRC  = "/kaggle/temp/banano-uav"   # scratch: NO se autoguarda como salida

if not os.path.isdir(SRC):
    subprocess.run(["git", "clone", "--depth", "1", "-b", RAMA, REPO, SRC], check=True)
else:
    subprocess.run(["git", "-C", SRC, "pull", "--ff-only"], check=False)
sys.path.insert(0, SRC)
print(subprocess.run(["git", "-C", SRC, "log", "--oneline", "-1"],
                     capture_output=True, text=True).stdout.strip())""",
    ),
    (
        "code",
        """# --- 3. Dependencias -------------------------------------------------------------
# ultralytics NO viene en la imagen de Kaggle; hace falta Internet ON en el notebook.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"], check=True)

import ultralytics
from ultralytics import settings
# Telemetria y loggers externos fuera: en una sesion desatendida cualquiera de estos
# puede quedarse esperando un login que nadie va a teclear.
settings.update({k: False for k in
                 ("sync", "wandb", "clearml", "comet", "dvc", "mlflow", "neptune", "raytune")})
print("ultralytics", ultralytics.__version__)""",
    ),
    (
        "code",
        """# --- 4. La clave de Roboflow ------------------------------------------------------
# Dos vias, y basta con una:
#   a) un DATASET PRIVADO adjunto con la clave dentro. Es el rodeo que documenta el propio
#      Kaggle, y es el unico que se puede montar entero desde la terminal: el CLI no tiene
#      ningun comando de secrets (issue Kaggle/kaggle-api#582, abierta).
#   b) Add-ons -> Secrets, que hay que crear a mano desde la web.
import glob

clave = ""
for ruta in glob.glob("/kaggle/input/*/roboflow.json"):
    try:
        clave = (json.load(open(ruta)).get("ROBOFLOW_API_KEY") or "").strip()
        if clave:
            print("clave leida del dataset privado adjunto")
            break
    except Exception:
        continue

if not clave:
    try:
        from kaggle_secrets import UserSecretsClient
        clave = UserSecretsClient().get_secret("ROBOFLOW_API_KEY").strip()
        print("clave leida del secreto de Kaggle")
    except Exception as e:
        # Kaggle responde HTTP 400 cuando el secreto NO EXISTE, y su cliente lo envuelve en
        # un "Connection error trying to communicate with service", que suena a fallo de red
        # y no lo es. Se traduce para no mandar a nadie a mirar la conexion.
        detalle = str(e)
        falta = "400" in detalle or "Connection error" in detalle
        # RuntimeError y no SystemExit: SystemExit revienta el formateador de traceback de
        # IPython (TypeError: object of type NoneType has no len()) y tapa este mensaje.
        raise RuntimeError(chr(10).join([
            "=" * 70,
            ("No hay clave de Roboflow: ni dataset adjunto con roboflow.json, ni secreto."
             if falta else "No se pudo leer la clave de Roboflow: " + detalle),
            "",
            "Cualquiera de las dos vale:",
            "  a) adjunta el dataset privado que lleve un roboflow.json con la clave, o",
            "  b) Add-ons -> Secrets -> Add a new secret",
            "     Label: ROBOFLOW_API_KEY   (exacto, en mayusculas)",
            "     Value: tu Private API Key de https://app.roboflow.com/settings/api",
            "            (la que NO empieza por rf_)",
            "=" * 70,
        ])) from None

os.environ["ROBOFLOW_API_KEY"] = clave
print("clave disponible:", len(clave), "caracteres")""",
    ),
    (
        "code",
        """# --- 5. Los datos, directos de Roboflow a esta maquina ----------------------------
# El PC de casa no sube nada: el manifiesto fija workspace/proyecto/VERSION de cada fuente
# y se descargan aqui. Van al scratch (fuera de /kaggle/working) porque /kaggle/working
# tiene 20 GB y se autoguarda entero al terminar: no queremos 5 GB de imagenes en la salida.
DATOS = "/kaggle/temp/realdata"
os.makedirs(DATOS, exist_ok=True)

INCLUIR_FINCAS_NUEVAS = True   # las 6 fincas de cloud/nuevas_fincas.json

cmd = [sys.executable, f"{SRC}/cloud/fetch_data.py", "--destino", DATOS]
subprocess.run(cmd, check=True, cwd=SRC)
if INCLUIR_FINCAS_NUEVAS:
    subprocess.run(cmd + ["--nuevas"], check=False, cwd=SRC)

total = sum(len(fs) for _, _, fs in os.walk(DATOS))
print(f"\\n{total} ficheros en {DATOS}")""",
    ),
    (
        "code",
        """# --- 6. Reparto por fincas, sin fugas ---------------------------------------------
# Agrupa por finca y deja cada una entera a un lado de la linea. Ojo con dos trampas ya
# medidas: extra/prueba2rgb es byte a byte el mismo dataset que extra/etiquetasnuevas, y
# newfarms/lasuiza y extra/platano-lasuiza son dos versiones del mismo proyecto.
SPLITS = f"{WORK}/splits"
subprocess.run([sys.executable, f"{SRC}/cloud/make_splits.py",
                "--raiz", DATOS, "--salida", SPLITS,
                "--raiz-declarada", DATOS], check=True, cwd=SRC)

# Control: ninguna imagen de validacion puede estar en el entrenamiento.
for y in sorted(os.listdir(SPLITS)):
    if y.startswith("lofo_"):
        r = subprocess.run([sys.executable, f"{SRC}/deep/leak_audit.py", "--data", f"{SPLITS}/{y}"],
                           capture_output=True, text=True, cwd=SRC)
        linea = [ln for ln in r.stdout.splitlines() if "TOTAL" in ln]
        print(f"{y:28s} {linea[0].strip() if linea else r.stdout.strip()[:80]}")""",
    ),
    (
        "code",
        """# --- 7. ¿A que resolucion mira el modelo? -----------------------------------------
# Barato y decisivo: sin reentrenar nada, sólo evaluando el modelo que ya existe a varias
# resoluciones. Sobre la finca ciega, pasar de 768 a 1024 subio el mAP50 un 65% y el
# recall otro 65%. Aqui se repite sobre TODAS las fincas retenidas, porque elegir la
# resolucion mirando una sola es afinar sobre el holdout.
HACER_BARRIDO = True

if HACER_BARRIDO:
    subprocess.run([sys.executable, f"{SRC}/cloud/scale_sweep.py",
                    "--pesos", f"{SRC}/models/banana_multifarm_v10.pt",
                    "--todas-las-fincas",
                    "--salida", f"{WORK}/scale_sweep.json"], check=False, cwd=SRC)
    if os.path.exists(f"{WORK}/scale_sweep.json"):
        d = json.load(open(f"{WORK}/scale_sweep.json"))
        mejores = {}
        for finca, r in d["fincas"].items():
            for fila in r["barrido"]:
                mejores.setdefault(fila["imgsz"], []).append(fila["mAP50"])
        print("\\nmAP50 medio sobre las fincas retenidas, por resolucion:")
        for imgsz, vals in sorted(mejores.items()):
            print(f"  {imgsz:5d}  {sum(vals)/len(vals):.4f}   ({len(vals)} fincas)")""",
    ),
    (
        "code",
        """# --- 8. Entrenar -------------------------------------------------------------------
# La T4 tiene 14,6 GB frente a los 8 GB de casa, asi que cabe mas lote y no hay que forzar
# workers=0 (aquello era un problema de Windows). Se usa UNA sola GPU aunque haya dos: con
# device="0,1" ultralytics lanza DDP y eso muere dentro de un notebook de Kaggle (medido).
IMGSZ  = 1024      # el barrido de arriba manda; si su optimo medio es otro, cambialo
RECETA = "escala"  # v10 | cenital | escala   (ver cloud/train.py)
MODELO = "yolo11m.pt"
DATA   = f"{SPLITS}/todas_las_fincas.yaml"

# NO se fijan epocas: se fija TIEMPO. La primera corrida midio 6,6 s por iteracion con 329
# iteraciones por epoca, o sea 36 min/epoca: 40 epocas habrian sido 24 h y la sesion muere a
# las 12. `--horas` se lo pasa a ultralytics, que lo comprueba al final de cada epoca y para
# solo dejando el best.pt escrito. Asi la corrida SIEMPRE cabe, entrene lo que entrene.
restante_h = LIMITE_H - (time.time() - T0) / 3600
horas_entreno = max(0.5, restante_h - 0.4)   # margen para guardar y recoger
print(f"llevamos {(time.time()-T0)/3600:.2f} h; se entrena un maximo de {horas_entreno:.2f} h")

r = subprocess.run([sys.executable, f"{SRC}/cloud/train.py",
                    "--data", DATA,
                    "--receta", RECETA,
                    "--modelo", MODELO,
                    "--imgsz", str(IMGSZ),
                    "--horas", f"{horas_entreno:.2f}",
                    "--epochs", "300",
                    "--proyecto", f"{WORK}/runs",
                    "--salida", f"{WORK}/cloud_runs.json"], cwd=SRC)

# La primera vez esto era check=False y no se miraba: el entrenamiento murio en la primera
# iteracion, el notebook siguio hasta el final y la sesion se dio por buena. Un fallo aqui
# tiene que verse.
if r.returncode != 0:
    print(f"AVISO: el entrenamiento termino con codigo {r.returncode}. "
          f"Mira {WORK}/cloud_runs.json para el motivo.")

"""
    ),
    (
        "code",
        """# --- 9. Recoger lo que vale la pena guardar ---------------------------------------
# /kaggle/working se guarda entero como salida de la version, pero tiene tope: dejamos los
# pesos y los JSON, no los miles de ficheros intermedios de ultralytics.
import glob, shutil
SALIDA = f"{WORK}/resultados"
os.makedirs(SALIDA, exist_ok=True)

for pt in glob.glob(f"{WORK}/runs/**/weights/best.pt", recursive=True):
    etiqueta = pt.split("/runs/")[1].split("/weights")[0].replace("/", "_")
    shutil.copy2(pt, f"{SALIDA}/{etiqueta}_best.pt")
    print("pesos ->", f"{etiqueta}_best.pt", f"{os.path.getsize(pt)/1e6:.1f} MB")

for j in ("cloud_runs.json", "scale_sweep.json"):
    if os.path.exists(f"{WORK}/{j}"):
        shutil.copy2(f"{WORK}/{j}", f"{SALIDA}/{j}")

for csv in glob.glob(f"{WORK}/runs/**/results.csv", recursive=True):
    etiqueta = csv.split("/runs/")[1].split("/results")[0].replace("/", "_")
    shutil.copy2(csv, f"{SALIDA}/{etiqueta}_results.csv")

# El clon vive en el scratch, asi que no ensucia la salida ni aunque esto no se ejecute.
print(f"\\nlisto en {(time.time()-T0)/3600:.2f} h")
print("\\n".join(sorted(os.listdir(SALIDA))))""",
    ),
]


def main() -> int:
    celdas: list[dict[str, Any]] = []
    for tipo, fuente in CELDAS:
        base: dict[str, Any] = {
            "cell_type": tipo,
            "metadata": {},
            "source": fuente.splitlines(keepends=True),
        }
        if tipo == "code":
            base["execution_count"] = None
            base["outputs"] = []
        celdas.append(base)

    nb = {
        "cells": celdas,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    DESTINO.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(celdas)} celdas -> {DESTINO}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
