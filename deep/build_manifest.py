"""Reconstruye el manifiesto de procedencia de los datasets a partir del disco.

Por qué existe (2026-08-23). `realdata/` está en .gitignore (son gigabytes), así que
el repo publicaba modelos entrenados sobre datos que NADIE más podía reconstruir: la
única prueba de qué dataset y qué VERSIÓN se usó vivía en unos README.txt sueltos
dentro de carpetas ignoradas por git. Para entrenar en la nube eso es un bloqueo
duro: la máquina remota no tiene el disco de Jhona.

Este script lee los `README.dataset.txt` / `README.roboflow.txt` que Roboflow deja al
exportar y escribe `cloud/data_manifest.json`, que SÍ se versiona. Con ese fichero,
`cloud/fetch_data.py` vuelve a descargar exactamente las mismas versiones en cualquier
máquina, sin subir una sola imagen desde el PC del usuario.

Uso:
    python deep/build_manifest.py                 # regenera cloud/data_manifest.json
    python deep/build_manifest.py --check         # falla si el manifiesto no coincide
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DESTINO = ROOT / "cloud" / "data_manifest.json"

# Roboflow escribe la URL del proyecto en README.dataset.txt y el nombre+versión en la
# segunda línea de README.roboflow.txt ("Count banana plants - v38 2024-10-18 2:31pm").
RE_URL = re.compile(r"https://universe\.roboflow\.com/([\w\-.]+)/([\w\-.]+)")
RE_LIC = re.compile(r"^License:\s*(.+)$", re.MULTILINE)
RE_VER = re.compile(r"\s-\sv(\d+)\b")
RE_IMGS = re.compile(r"includes ([\d,]+) images")

# Rol de cada carpeta en los protocolos del repo. Se mantiene a mano a propósito: es
# una decisión de método (qué finca es holdout ciego), no un dato del disco.
ROLES: dict[str, str] = {
    "count_banana_plants": "train+val (finca original)",
    "extra/plantas_jovenes_50m": "train",
    "extra/plantas_jovenes_80m1": "train",
    "extra/plantas_platano": "train",
    "extra/prueba2rgb": "train",
    "extra/etiquetasnuevas": "train",
    "newfarms/elliot": "train desde v12",
    "newfarms/m2": "train desde v12",
    "newfarms/lasuiza": "train desde v12",
    "newfarms/armah": "HOLDOUT CIEGO — ningún modelo entrena con ella",
    "extra/banana_counting": "descargado, SIN USAR en ningún entrenamiento",
    "extra/ml_agromatica": "descargado, SIN USAR (tiene 2 clases: exige remapeo)",
    "extra/platano-lasuiza": "descargado, SIN USAR (v3 del mismo proyecto que newfarms/lasuiza v2)",
    "extra/conteobanano": "descargado, SIN USAR",
    "newfarms/karachi": "descartado: cultivo sin confirmar",
    "newfarms/conteo": "descartado: sub-etiquetado",
}


def lee_fuente(carpeta: Path, raiz: Path) -> dict[str, Any] | None:
    ds = carpeta / "README.dataset.txt"
    rf = carpeta / "README.roboflow.txt"
    if not ds.exists():
        return None

    texto_ds = ds.read_text(encoding="utf-8", errors="replace")
    m = RE_URL.search(texto_ds)
    if not m:
        return None
    workspace, proyecto = m.group(1), m.group(2)

    lic = RE_LIC.search(texto_ds)
    version: int | str | None = None
    imgs = None
    if rf.exists():
        texto_rf = rf.read_text(encoding="utf-8", errors="replace")
        mv = RE_VER.search(texto_rf)
        if mv:
            version = int(mv.group(1))
        else:
            # "Roboflow Instant" exporta con un id de versión largo, no un entero
            mv2 = re.search(r"\s-\sv(\S+)", texto_rf)
            version = mv2.group(1) if mv2 else None
        mi = RE_IMGS.search(texto_rf)
        if mi:
            imgs = int(mi.group(1).replace(",", ""))

    rel = carpeta.relative_to(raiz).as_posix()
    return {
        "carpeta": rel,
        "workspace": workspace,
        "proyecto": proyecto,
        "version": version,
        "url": f"https://universe.roboflow.com/{workspace}/{proyecto}",
        "licencia": lic.group(1).strip() if lic else "NO DECLARADA",
        "imgs_exportadas": imgs,
        "rol": ROLES.get(rel, "sin clasificar"),
    }


def construye(raiz: Path) -> dict[str, Any]:
    fuentes: list[dict[str, Any]] = []
    for patron in ("*", "extra/*", "newfarms/*"):
        for d in sorted(raiz.glob(patron)):
            if not d.is_dir():
                continue
            f = lee_fuente(d, raiz)
            if f:
                fuentes.append(f)

    # Aviso: el mismo proyecto de Roboflow descargado dos veces en versiones distintas
    # es una fuga train/test esperando a ocurrir.
    por_proyecto: dict[str, list[str]] = {}
    for f in fuentes:
        por_proyecto.setdefault(f"{f['workspace']}/{f['proyecto']}", []).append(f["carpeta"])
    duplicados = {k: v for k, v in por_proyecto.items() if len(v) > 1}

    return {
        "generado_por": "deep/build_manifest.py",
        "fecha": "2026-08-23",
        "nota": (
            "Procedencia exacta de cada carpeta de realdata/, que está en .gitignore. "
            "Con esto cloud/fetch_data.py reconstruye el dataset en cualquier máquina "
            "sin subir imágenes desde el PC del autor. Todas las fuentes son CC BY 4.0 "
            "de Roboflow Universe."
        ),
        "avisos": (
            [
                f"MISMO proyecto en dos carpetas ({k}): {', '.join(v)}. "
                "Si alguna vez se meten las dos en el mismo entrenamiento, hay fuga."
                for k, v in duplicados.items()
            ]
            or ["sin duplicados de proyecto"]
        ),
        "fuentes": fuentes,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raiz", type=Path, default=ROOT / "realdata")
    ap.add_argument("--salida", type=Path, default=DESTINO)
    ap.add_argument("--check", action="store_true", help="no escribe; falla si difiere")
    args = ap.parse_args()

    if not args.raiz.exists():
        print(f"No existe {args.raiz} (normal fuera del PC del autor).", file=sys.stderr)
        return 0 if args.check else 1

    manifiesto = construye(args.raiz)
    texto = json.dumps(manifiesto, indent=2, ensure_ascii=False) + "\n"

    if args.check:
        if not args.salida.exists() or args.salida.read_text(encoding="utf-8") != texto:
            print("El manifiesto no coincide con el disco: regenera con deep/build_manifest.py", file=sys.stderr)
            return 1
        print("Manifiesto al día.")
        return 0

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(texto, encoding="utf-8")
    print(f"{len(manifiesto['fuentes'])} fuentes -> {args.salida}")
    for aviso in manifiesto["avisos"]:
        print(f"  aviso: {aviso}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
