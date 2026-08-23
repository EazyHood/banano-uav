"""Reconstruye realdata/ desde Roboflow, en la máquina que sea.

Está pensado para correr EN LA NUBE (Kaggle, Colab, cualquier VM), no en el PC del
autor: su ordenador falla y no queremos que suba gigabytes ni entrene nada. El
manifiesto `cloud/data_manifest.json` fija workspace, proyecto y VERSIÓN exacta de
cada fuente, así que lo que se descarga aquí es lo mismo que se midió en su día.

Todas las fuentes son CC BY 4.0 de Roboflow Universe. Hace falta una API key propia
(gratuita); se lee, por este orden, de:
    1. la variable de entorno ROBOFLOW_API_KEY
    2. el Kaggle Secret llamado ROBOFLOW_API_KEY
    3. el fichero ~/.roboflow_key

La key es la "Private API Key" de la cuenta (sin el prefijo rf_): las que empiezan
por rf_ son de la API nueva y este endpoint las rechaza.

Uso:
    python cloud/fetch_data.py                      # todo lo que use algún protocolo
    python cloud/fetch_data.py --incluir-sin-usar   # además las 4 descargadas y nunca usadas
    python cloud/fetch_data.py --solo newfarms/armah
    python cloud/fetch_data.py --destino /kaggle/working/realdata
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MANIFIESTO = ROOT / "cloud" / "data_manifest.json"

API = "https://api.roboflow.com/{workspace}/{proyecto}/{version}/yolov8?api_key={key}"
REINTENTOS = 3
ESPERA_S = 5


def lee_key() -> str:
    k = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if k:
        return k
    try:  # Kaggle: el secreto se declara en "Add-ons > Secrets"
        from kaggle_secrets import UserSecretsClient  # type: ignore[import-not-found]

        k = UserSecretsClient().get_secret("ROBOFLOW_API_KEY").strip()
        if k:
            return k
    except Exception:
        pass
    fichero = Path.home() / ".roboflow_key"
    if fichero.exists():
        return fichero.read_text(encoding="utf-8").strip()
    raise SystemExit(
        "No hay API key de Roboflow. Define ROBOFLOW_API_KEY, o crea el secreto del "
        "mismo nombre en Kaggle, o escribe la clave en ~/.roboflow_key"
    )


def _pide_enlace(fuente: dict[str, Any], key: str) -> tuple[str, float]:
    url = API.format(
        workspace=fuente["workspace"],
        proyecto=fuente["proyecto"],
        version=fuente["version"],
        key=key,
    )
    ultimo: Exception | None = None
    for intento in range(REINTENTOS):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                datos = json.load(r)
            export = datos.get("export", {})
            enlace = export.get("link")
            if not enlace:
                # Roboflow genera el zip de forma asíncrona la primera vez
                if datos.get("progress") is not None:
                    time.sleep(ESPERA_S)
                    continue
                raise RuntimeError(f"respuesta sin enlace: {list(datos)}")
            return enlace, float(export.get("size") or 0)
        except urllib.error.HTTPError as e:
            ultimo = e
            if e.code in (401, 403):  # una key mala no mejora reintentando
                raise SystemExit(
                    f"Roboflow rechaza la key ({e.code}) para {fuente['carpeta']}. "
                    "¿Es la Private API Key, sin el prefijo rf_?"
                ) from e
            time.sleep(ESPERA_S * (intento + 1))
        except Exception as e:  # red inestable
            ultimo = e
            time.sleep(ESPERA_S * (intento + 1))
    raise RuntimeError(f"no se pudo obtener el enlace de {fuente['carpeta']}: {ultimo}")


def cuenta_imgs(carpeta: Path) -> int:
    if not carpeta.exists():
        return 0
    vistos = set()
    for p in carpeta.rglob("*"):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            vistos.add(os.path.normcase(str(p)))
    return len(vistos)


def descarga(fuente: dict[str, Any], destino: Path, key: str, forzar: bool) -> str:
    carpeta = destino / fuente["carpeta"]
    esperadas = fuente.get("imgs_exportadas")

    if carpeta.exists() and not forzar:
        hay = cuenta_imgs(carpeta)
        # La exportación con augmentación de Roboflow reparte el mismo original en
        # varias copias, así que el número en disco puede quedar por debajo del
        # anunciado si alguien ya dedupló. Con que haya imágenes, no re-descargamos.
        if hay:
            return f"ya está ({hay} imgs)"

    enlace, tam_mb = _pide_enlace(fuente, key)
    tmp = destino / f".{fuente['carpeta'].replace('/', '_')}.zip"
    tmp.parent.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(enlace, timeout=600) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)

    if carpeta.exists():
        shutil.rmtree(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(tmp) as z:
        z.extractall(carpeta)
    tmp.unlink(missing_ok=True)

    hay = cuenta_imgs(carpeta)
    aviso = ""
    if esperadas and hay != esperadas:
        aviso = f"  ¡OJO! el manifiesto decía {esperadas}"
    return f"{hay} imgs, {tam_mb:.1f} MB{aviso}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--destino", type=Path, default=ROOT / "realdata")
    ap.add_argument("--manifiesto", type=Path, default=MANIFIESTO)
    ap.add_argument("--solo", nargs="*", help="carpetas concretas del manifiesto")
    ap.add_argument("--incluir-sin-usar", action="store_true", help="también las descargadas y nunca usadas")
    ap.add_argument("--incluir-descartados", action="store_true", help="también karachi y conteo")
    ap.add_argument("--forzar", action="store_true", help="re-descargar aunque exista")
    args = ap.parse_args()

    manifiesto = json.loads(args.manifiesto.read_text(encoding="utf-8"))
    fuentes: list[dict[str, Any]] = manifiesto["fuentes"]

    if args.solo:
        pedidas = set(args.solo)
        fuentes = [f for f in fuentes if f["carpeta"] in pedidas]
        faltan = pedidas - {f["carpeta"] for f in fuentes}
        if faltan:
            print(f"No están en el manifiesto: {', '.join(sorted(faltan))}", file=sys.stderr)
            return 1
    else:
        if not args.incluir_descartados:
            fuentes = [f for f in fuentes if not f["rol"].startswith("descartado")]
        if not args.incluir_sin_usar:
            fuentes = [f for f in fuentes if "SIN USAR" not in f["rol"]]

    if not fuentes:
        print("Nada que descargar con esos filtros.", file=sys.stderr)
        return 1

    key = lee_key()
    print(f"{len(fuentes)} fuentes -> {args.destino}\n")
    fallos = 0
    for f in fuentes:
        etiqueta = f"{f['carpeta']} (v{f['version']})"
        print(f"  {etiqueta:52s} ", end="", flush=True)
        try:
            print(descarga(f, args.destino, key, args.forzar))
        except SystemExit:
            raise
        except Exception as e:
            fallos += 1
            print(f"FALLO: {e}")

    print(f"\nListo. {len(fuentes) - fallos}/{len(fuentes)} fuentes disponibles en {args.destino}")
    for aviso in manifiesto.get("avisos", []):
        if not aviso.startswith("sin duplicados"):
            print(f"  aviso: {aviso}")
    return 1 if fallos else 0


if __name__ == "__main__":
    raise SystemExit(main())
