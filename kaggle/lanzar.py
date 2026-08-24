"""Lanza el entrenamiento en Kaggle desde la terminal, sin abrir el navegador.

Un comando y el PC se puede apagar: el entrenamiento sigue en los servidores de Kaggle.

    python kaggle/lanzar.py                 # sube el notebook y arranca la corrida
    python kaggle/lanzar.py --estado        # ¿por dónde va?
    python kaggle/lanzar.py --recoger       # baja los pesos entrenados

Rellena solo el campo `id` del kernel-metadata.json con tu usuario de Kaggle, que saca
de las credenciales, para no tener que editar JSON a mano.

REQUISITOS, una sola vez:
  1. Cuenta en kaggle.com y verificación por teléfono (SMS). Sin ella la GPU y el acceso
     a Internet del notebook quedan bloqueados, y el entrenamiento no puede ni descargar
     los datos. No pide tarjeta.
  2. pip install kaggle    (el paquete exige Python >= 3.11)
  3. Un token: https://www.kaggle.com/settings/api -> "Generate New Token". Ese boton da
     hoy un token suelto que empieza por KGAT_, NO un fichero. Guardalo en texto plano en
     ~/.kaggle/access_token (o en la variable KAGGLE_API_TOKEN, o usa `kaggle auth login`).
     OJO: no sirve meterlo en ~/.kaggle/kaggle.json — ese es el formato antiguo y espera un
     JSON con username y key; con el token pelado dentro el CLI falla sin explicar por que.
     El boton "Create Legacy API Key" es el que genera un kaggle.json de verdad.
  4. El secreto ROBOFLOW_API_KEY dentro del notebook (Add-ons -> Secrets). Esto SÍ hay
     que hacerlo por web una vez: la API de Kaggle todavía no permite adjuntar secretos
     al empujar un kernel (issue Kaggle/kaggle-api#582, abierta).

CUOTA: 30 horas de GPU por semana, que se renuevan solas los sábados a medianoche UTC.
Cada sesión dura como mucho 12 horas. Si una corrida se pasa de las 12 h, el guardado de
ficheros del final es "best effort" y puedes perder los pesos: por eso el notebook se
presupuesta 11 h y guarda checkpoints por el camino.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
CARPETA = AQUI / "entrenar"
META = CARPETA / "kernel-metadata.json"
ACELERADOR = "NvidiaTeslaT4"  # NUNCA P100: la imagen de Kaggle no trae kernels de Pascal


def cli() -> str:
    exe = shutil.which("kaggle")
    if exe:
        return exe
    # en Windows el script suele quedar en Scripts/ del venv y no siempre está en PATH
    cand = Path(sys.executable).parent / ("kaggle.exe" if os.name == "nt" else "kaggle")
    if cand.exists():
        return str(cand)
    raise SystemExit(
        "No encuentro el CLI de kaggle. Instálalo con:\n"
        f"    {sys.executable} -m pip install kaggle"
    )


def usuario() -> str:
    """Saca el usuario de Kaggle de donde estén las credenciales."""
    if u := os.environ.get("KAGGLE_USERNAME", "").strip():
        return u
    for p in (Path.home() / ".kaggle" / "kaggle.json", Path.home() / "kaggle.json"):
        if p.exists():
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(d, dict) and d.get("username"):
                    return str(d["username"])
            except json.JSONDecodeError:
                continue
    # Con el token nuevo no hay fichero con el nombre dentro: se lo preguntamos a la API.
    r = subprocess.run([cli(), "config", "view"], capture_output=True, text=True)
    for linea in r.stdout.splitlines():
        if "username" in linea.lower() and ":" in linea:
            return linea.split(":", 1)[1].strip()
    raise SystemExit(
        "No pude averiguar tu usuario de Kaggle. Pásalo a mano:\n"
        "    python kaggle/lanzar.py --usuario TU_USUARIO\n"
        "o define la variable de entorno KAGGLE_USERNAME."
    )


def fija_id(user: str) -> str:
    meta = json.loads(META.read_text(encoding="utf-8"))
    slug = meta["id"].split("/", 1)[1]
    nuevo = f"{user}/{slug}"
    if meta["id"] != nuevo:
        meta["id"] = nuevo
        META.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return nuevo


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--usuario", help="tu usuario de Kaggle, si no se detecta solo")
    ap.add_argument("--estado", action="store_true", help="consulta cómo va la corrida")
    ap.add_argument("--recoger", action="store_true", help="descarga la salida (pesos y métricas)")
    ap.add_argument("--destino", type=Path, default=AQUI.parent / "runs_cloud" / "kaggle")
    args = ap.parse_args()

    k = cli()
    user = args.usuario or usuario()
    kid = fija_id(user)

    if args.estado:
        r = subprocess.run([k, "kernels", "status", kid], capture_output=True, text=True)
        print((r.stdout + r.stderr).strip())
        return r.returncode

    if args.recoger:
        args.destino.mkdir(parents=True, exist_ok=True)
        r = subprocess.run([k, "kernels", "output", kid, "-p", str(args.destino)],
                           capture_output=True, text=True)
        print((r.stdout + r.stderr).strip())
        if r.returncode == 0:
            pesos = sorted(args.destino.rglob("*.pt"))
            print(f"\n{len(pesos)} ficheros de pesos en {args.destino}")
            for p in pesos:
                print(f"  {p.name}  {p.stat().st_size/1e6:.1f} MB")
        return r.returncode

    # Regenerar el notebook para que lo que se sube sea lo que está en el repo
    subprocess.run([sys.executable, str(AQUI / "build_notebook.py")], check=True)

    print(f"Subiendo {kid} con acelerador {ACELERADOR}...")
    r = subprocess.run([k, "kernels", "push", "-p", str(CARPETA), "--accelerator", ACELERADOR],
                       capture_output=True, text=True)
    salida = (r.stdout + r.stderr).strip()
    print(salida)
    if r.returncode != 0:
        if "accelerator" in salida.lower():
            print("\nSi tu CLI no acepta --accelerator, elige 'GPU T4 x2' en el editor web.",
                  file=sys.stderr)
        return r.returncode

    print(
        f"\nLanzado. Ya puedes apagar el PC.\n"
        f"  seguirlo:  https://www.kaggle.com/code/{kid}\n"
        f"  o:         python kaggle/lanzar.py --estado\n"
        f"  al acabar: python kaggle/lanzar.py --recoger\n\n"
        "La PRIMERA vez tienes que abrir el notebook en la web y engancharle el secreto\n"
        "ROBOFLOW_API_KEY (Add-ons -> Secrets): la API de Kaggle todavía no puede hacerlo\n"
        "al empujar el kernel. Sin eso la celda 4 falla y no se descargan los datos."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
