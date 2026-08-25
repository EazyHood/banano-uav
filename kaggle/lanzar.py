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
from typing import Any

AQUI = Path(__file__).resolve().parent
CARPETA = AQUI / "entrenar"
META = CARPETA / "kernel-metadata.json"
ACELERADOR = "NvidiaTeslaT4"  # NUNCA P100: la imagen de Kaggle no trae kernels de Pascal
DATASET_PESOS = "banano-uav-pesos"  # dataset privado que encadena una sesión con la siguiente


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


def encadenar(user: str, destino: Path) -> int:
    """Deja el `last.pt` recogido donde la SIGUIENTE sesión pueda verlo, y lo engancha.

    `/kaggle/working` arranca vacío en cada corrida, así que sin esto cada sesión reentrena
    desde cero y las horas anteriores no suman — que es justo lo que hay que evitar cuando
    la cuota son 30 h por semana. El README lo resolvía con un `+ Add Input` a mano en la
    web; esto hace lo mismo desde la terminal, subiendo los pesos como dataset privado.

    Los ficheros van PLANOS (`last.pt`, `args.yaml`, `origen.json`) a propósito: subir
    carpetas obliga a `--dir-mode zip` y entonces depende de si Kaggle desempaqueta el zip
    o lo deja tal cual. `origen.json` dice a qué tirada pertenecen, que es lo único que la
    estructura de carpetas aportaba.
    """
    ultimos = sorted(destino.glob("runs/*/weights/last.pt"))
    if not ultimos:
        print(f"No hay ningún last.pt en {destino}. Recógelo antes:\n"
              "    python kaggle/lanzar.py --recoger")
        return 1
    if len(ultimos) > 1:
        print(f"Hay {len(ultimos)} tiradas recogidas; se encadena la más reciente.")
        ultimos.sort(key=lambda p: p.stat().st_mtime)
    ultimo = ultimos[-1]
    nombre = ultimo.parent.parent.name

    fase = AQUI / "pesos"
    if fase.exists():
        shutil.rmtree(fase)
    fase.mkdir(parents=True)
    shutil.copy2(ultimo, fase / "last.pt")
    argumentos = ultimo.parent.parent / "args.yaml"
    if argumentos.exists():
        # ultralytics lo necesita para reanudar con la MISMA receta; sin él, resume falla.
        shutil.copy2(argumentos, fase / "args.yaml")
    (fase / "origen.json").write_text(
        json.dumps({"run": nombre, "fichero": "last.pt"}, indent=2) + "\n", encoding="utf-8"
    )
    ref = f"{user}/{DATASET_PESOS}"
    (fase / "dataset-metadata.json").write_text(
        json.dumps({"title": "banano-uav pesos", "id": ref, "licenses": [{"name": "CC0-1.0"}]},
                   indent=2) + "\n",
        encoding="utf-8",
    )

    k = cli()
    mb = ultimo.stat().st_size / 1e6
    print(f"Subiendo {nombre}/last.pt ({mb:.1f} MB) como {ref}...")
    # SIN -u: ese flag es --public. Son pesos entrenados sobre datos con licencia de
    # terceros y una tirada a medias; se sube privado, que es el default.
    r = subprocess.run([k, "datasets", "create", "-p", str(fase)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    salida = (r.stdout + r.stderr).strip()
    if r.returncode != 0 and "already exist" in salida.lower():
        r = subprocess.run([k, "datasets", "version", "-p", str(fase), "-m", f"last.pt de {nombre}"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        salida = (r.stdout + r.stderr).strip()
    print(salida)
    if r.returncode != 0:
        return r.returncode

    meta = json.loads(META.read_text(encoding="utf-8"))
    fuentes = list(meta.get("dataset_sources") or [])
    if ref not in fuentes:
        fuentes.append(ref)
        meta["dataset_sources"] = fuentes
        META.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"{ref} añadido a dataset_sources del notebook.")
    print("La próxima corrida reanudará desde ahí. Lánzala con: python kaggle/lanzar.py")
    return 0


def rutas_probables(nombres: list[str]) -> list[str]:
    """Del inventario (sólo nombres, sin carpeta) a rutas que se puedan pedir.

    El endpoint que lista los ficheros de una versión devuelve `cloud_runs.json` dos
    veces con tamaños distintos: son dos ficheros en dos carpetas, y la carpeta no
    viaja. Se reconstruye a partir de lo que sabe el notebook: la salida vive en
    `resultados/`, los YAML en `splits/`, y cada `<etiqueta>_best.pt` de `resultados/`
    viene de `runs/<etiqueta>/`, que es donde está el `last.pt` con el que se reanuda.
    """
    rutas: list[str] = []
    for n in nombres:
        rutas += [f"resultados/{n}", n, f"splits/{n}"]
        if n.endswith("_best.pt"):
            etiqueta = n[: -len("_best.pt")]
            rutas += [
                f"runs/{etiqueta}/weights/best.pt",
                f"runs/{etiqueta}/weights/last.pt",
                f"runs/{etiqueta}/args.yaml",
                f"runs/{etiqueta}/results.csv",
            ]
    return list(dict.fromkeys(rutas))


def _baja(api: Any, user: str, slug: str, ruta: str) -> bytes | None:
    from kagglesdk.kernels.types.kernels_api_service import ApiDownloadKernelOutputRequest

    with api.build_kaggle_client() as cliente:
        pet = ApiDownloadKernelOutputRequest()
        pet.owner_slug, pet.kernel_slug, pet.file_path = user, slug, ruta
        try:
            res = cliente.kernels.kernels_api_client.download_kernel_output(pet)
        except Exception:
            return None
    return res.content if getattr(res, "status_code", 0) == 200 and res.content else None


def recoger(kid: str, destino: Path) -> int:
    """Baja la salida de la última versión guardada del notebook.

    `kaggle kernels output` NO mira la versión: mira la sesión. Con una pestaña del
    editor abierta —la que abre el propio correo de "tu notebook ha terminado" al pulsar
    'View on Kaggle'— contesta CERO ficheros y sin explicar por qué. Medido el
    2026-08-24: dijo que no había nada justo después de una tirada de 6,3 h que sí había
    guardado 40 MB de pesos. Por eso, si esa vía viene vacía, se baja fichero a fichero
    por su ruta (otro endpoint, y ese sí ve la versión).
    """
    from kaggle.api.kaggle_api_extended import KaggleApi

    destino.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    user, slug = kid.split("/", 1)

    bajados = [Path(f) for f in api.kernels_output(kid, path=str(destino), page_size=200)[0]]

    if not bajados:
        inventario = api.kernels_list_files(kid, page_size=200)
        nombres = [f.name for f in (inventario.files or [])]
        if not nombres:
            print("La versión guardada no tiene ningún fichero: la corrida no llegó a guardar.")
            return 0
        print(f"La sesión abierta tapa la salida; la versión guardada tiene {len(nombres)} "
              f"ficheros. Bajándolos por ruta.")
        # El manifiesto lo escribe la última celda del notebook y trae las rutas exactas.
        # Sólo lo tienen las versiones nuevas: para las viejas se reconstruyen a mano.
        crudo = _baja(api, user, slug, "resultados/MANIFIESTO.json")
        rutas = json.loads(crudo)["ficheros"] if crudo else rutas_probables(nombres)
        for ruta in rutas:
            contenido = _baja(api, user, slug, ruta)
            if contenido is None:
                continue
            fichero = destino.joinpath(*ruta.split("/"))
            fichero.parent.mkdir(parents=True, exist_ok=True)
            fichero.write_bytes(contenido)
            bajados.append(fichero)

    print(f"\n{len(bajados)} ficheros en {destino}")
    for p in sorted(bajados):
        if p.exists():
            print(f"  {p.stat().st_size/1e6:8.3f} MB  {p.relative_to(destino)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--usuario", help="tu usuario de Kaggle, si no se detecta solo")
    ap.add_argument("--estado", action="store_true", help="consulta cómo va la corrida")
    ap.add_argument("--recoger", action="store_true", help="descarga la salida (pesos y métricas)")
    ap.add_argument("--encadenar", action="store_true",
                    help="sube el last.pt recogido y lo engancha, para que la siguiente "
                         "sesión continúe en vez de empezar de cero")
    ap.add_argument("--log", action="store_true", help="enseña el log de la corrida (sin descargar la salida)")
    ap.add_argument("--destino", type=Path, default=AQUI.parent / "runs_cloud" / "kaggle")
    args = ap.parse_args()

    k = cli()
    user = args.usuario or usuario()
    kid = fija_id(user)

    if args.estado:
        r = subprocess.run([k, "kernels", "status", kid], capture_output=True, text=True)
        print((r.stdout + r.stderr).strip())
        return r.returncode

    if args.log:
        # `kernels logs` trae solo el log; `kernels output` se baja TODA la salida, que en
        # una corrida fallida puede ser el repo clonado entero. Diagnosticar no debe costar
        # 161 MB, que es lo que costo la primera vez.
        r = subprocess.run([k, "kernels", "logs", kid], capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        if r.returncode != 0:
            print((r.stdout + r.stderr).strip())
            return r.returncode
        try:
            eventos = json.loads(r.stdout)
        except json.JSONDecodeError:
            print(r.stdout)
            return 0
        for e in eventos:
            texto = (e.get("data") or "").rstrip()
            if texto:
                marca = "!" if e.get("stream_name") == "stderr" else " "
                print(f"{marca} {texto}")
        return 0

    if args.recoger:
        return recoger(kid, args.destino)

    if args.encadenar:
        return encadenar(user, args.destino)

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
