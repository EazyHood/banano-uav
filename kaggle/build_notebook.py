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
        raise SystemExit(
            "GPU P100 detectada. La imagen de Kaggle no trae kernels de Pascal y el "
            "entrenamiento moriria en el primer lote. Cambia el acelerador a 'GPU T4 x2'."
        )
    # Prueba real: is_available() puede mentir, una operacion CUDA no.
    torch.zeros(8, device="cuda").sum().item()
    print("  operacion CUDA de prueba: OK")
else:
    print("AVISO: sin GPU. Activa el acelerador o esto tardara dias.")""",
    ),
    (
        "code",
        f"""# --- 2. Codigo del proyecto ------------------------------------------------------
REPO = {REPO!r}
RAMA = {RAMA!r}
WORK = "/kaggle/working"
SRC  = f"{{WORK}}/banano-uav"

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
# Se define en Add-ons -> Secrets (no se puede adjuntar desde la CLI: la issue
# Kaggle/kaggle-api#582 sigue abierta, asi que un kernel recien creado por 'kernels push'
# necesita que abras el editor UNA vez para engancharle el secreto).
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["ROBOFLOW_API_KEY"] = UserSecretsClient().get_secret("ROBOFLOW_API_KEY").strip()
    print("clave leida del secreto de Kaggle")
except Exception as e:
    raise SystemExit(
        f"No se pudo leer el secreto ROBOFLOW_API_KEY ({e}). "
        "Add-ons -> Secrets -> Add a new secret, etiqueta exacta ROBOFLOW_API_KEY."
    )""",
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
# La T4 tiene 16 GB frente a los 8 GB de casa, asi que cabe mas lote y no hay que
# forzar workers=0 (aquello era un problema de Windows).
IMGSZ   = 1024      # el barrido de arriba manda: si su optimo medio es otro, cambialo
EPOCHS  = 100
RECETA  = "escala"  # v10 | cenital | escala   (ver cloud/train.py)
MODELO  = "yolo11m.pt"
DATA    = f"{SPLITS}/todas_las_fincas.yaml"

restante_h = LIMITE_H - (time.time() - T0) / 3600
print(f"quedan {restante_h:.1f} h de las {LIMITE_H} presupuestadas\\n")

subprocess.run([sys.executable, f"{SRC}/cloud/train.py",
                "--data", DATA,
                "--receta", RECETA,
                "--modelo", MODELO,
                "--imgsz", str(IMGSZ),
                "--epochs", str(EPOCHS),
                "--proyecto", f"{WORK}/runs",
                "--salida", f"{WORK}/cloud_runs.json"], check=False, cwd=SRC)""",
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

# El clon del repo y los datos no pintan nada en la salida.
shutil.rmtree(SRC, ignore_errors=True)
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
