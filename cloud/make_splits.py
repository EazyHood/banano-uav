"""Genera los data.yaml del proyecto: portables y sin fugas.

Dos problemas que arregla (2026-08-23):

1. RUTAS. Los YAML de realdata/ llevan `path: C:/Users/jhona/banano-drone/realdata`
   incrustado. En cualquier máquina que no sea la de Jhona no resuelven, así que el
   entrenamiento no se podía sacar de su PC sin editarlos a mano. Aquí la raíz es un
   parámetro y por defecto se escribe relativa al propio YAML.

2. AGRUPACIÓN POR FINCA. El protocolo "otras 4 fincas" (holdout.yaml) no eran cuatro
   fincas: tres de sus fuentes salen del mismo workspace de Roboflow (agromatica2025)
   y la cuarta, `prueba2rgb`, es byte a byte el mismo dataset que `etiquetasnuevas`,
   que está en el entrenamiento — el 100% de sus 25 imágenes de test estaban vistas
   (ver deep/leak_audit.py). Aquí las fuentes se agrupan en FINCAS y una finca entera
   cae del mismo lado de la línea, nunca partida.

Y añade el protocolo que faltaba: LOFO (Leave-One-Farm-Out). Hoy la única cifra
honesta del repo sale de armah: 62 imágenes, una sola finca, una sola tirada. Con eso
un cambio de receta puede subir o bajar 5 puntos por puro azar del reparto. LOFO
entrena N veces reteniendo una finca distinta cada vez y promedia: misma pregunta
("¿qué pasa en una finca nueva?") con mucho menos ruido, y además dice en CUÁLES
falla, que es lo que orienta el siguiente paso.

Uso:
    python cloud/make_splits.py                          # escribe splits/ en el repo
    python cloud/make_splits.py --raiz /kaggle/working/realdata --salida /kaggle/working/splits
    python cloud/make_splits.py --listar                 # sólo enseña la agrupación
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

# Una FINCA = un origen independiente. Agrupar de más es lo prudente: si dos fuentes
# del mismo workspace fuesen de verdad fincas distintas, tratarlas como una sola sólo
# hace la estimación más pesimista. Al revés —partir una finca entre train y val— es
# lo que infla las cifras, y es justo lo que pasaba.
FINCAS: dict[str, dict[str, Any]] = {
    "original": {
        "carpetas": ["count_banana_plants"],
        "nota": "finca del dataset count-banana-plants. 480x480, planta ~169 px a imgsz 768.",
    },
    "agromatica": {
        "carpetas": [
            "extra/plantas_jovenes_50m",
            "extra/plantas_jovenes_80m1",
            "extra/plantas_platano",
        ],
        "nota": (
            "workspace agromatica2025: tres vuelos (50 m, 80 m, plátano) del mismo operador. "
            "Se agrupan como UNA finca. Aportan el 81% de las cajas del entrenamiento de v10, "
            "todas de 16-21 px: son las que sesgan el modelo hacia objetos diminutos."
        ),
    },
    "tesis": {
        "carpetas": ["extra/etiquetasnuevas"],
        "nota": (
            "IMPORTANTE: extra/prueba2rgb es este MISMO dataset (505 imágenes idénticas por MD5, "
            "subidas a otro workspace). Se usa sólo una de las dos copias; incluir ambas duplicaba "
            "el peso en el entrenamiento y fugaba el test."
        ),
    },
    "elliot": {"carpetas": ["newfarms/elliot"], "nota": "plantación joven sobre suelo desnudo, ~99 cajas/img, planta ~31 px."},
    "m2": {"carpetas": ["newfarms/m2"], "nota": "plátano disperso en ladera. Planta ~10 px a imgsz 768: en el límite de lo detectable."},
    "lasuiza": {
        "carpetas": ["newfarms/lasuiza"],
        "nota": (
            "Colombia, plátano adulto. Planta ~333 px: el extremo opuesto a agromatica. "
            "extra/platano-lasuiza es el mismo proyecto de Roboflow en otra versión (84 imágenes "
            "compartidas): NO usar las dos a la vez."
        ),
    },
    "armah": {
        "carpetas": ["newfarms/armah"],
        "nota": (
            "Ghana, paisaje mixto con palma aceitera. Planta ~45 px. Holdout histórico del repo: "
            "ningún modelo publicado ha entrenado con ella. Se mantiene reservada."
        ),
    },
}

# Descargadas y nunca usadas. Se dejan fuera por defecto pero declaradas, porque son
# ~3.700 imágenes reales disponibles y son la vía más barata de meter fincas nuevas.
FINCAS_SIN_USAR: dict[str, dict[str, Any]] = {
    "banana_counting": {
        "carpetas": ["extra/banana_counting"],
        "nota": "2.344 imgs, 256x256, planta ~184 px. Nunca entró en ningún entrenamiento.",
    },
    "ml_agromatica": {
        "carpetas": ["extra/ml_agromatica"],
        "nota": "1.312 imgs. OJO: tiene 2 clases; exige remapear a clase única antes de usar.",
    },
    "conteobanano": {"carpetas": ["extra/conteobanano"], "nota": "50 imgs, planta ~160 px."},
}

SPLITS = ("train", "valid", "test")


def carpetas_existentes(raiz: Path, finca: dict[str, Any], splits: tuple[str, ...]) -> list[str]:
    """Rutas .../split/images que existen de verdad, relativas a la raíz."""
    out: list[str] = []
    for c in finca["carpetas"]:
        for s in splits:
            p = raiz / c / s / "images"
            if p.is_dir() and any(p.iterdir()):
                out.append(f"{c}/{s}/images")
    return out


def escribe_yaml(destino: Path, raiz_decl: str, train: list[str], val: list[str], comentario: str) -> None:
    lineas = [f"path: {raiz_decl}", "train:"]
    lineas += [f"  - {t}" for t in train]
    lineas.append("val:")
    lineas += [f"  - {v}" for v in val]
    lineas += ["nc: 1", "names: ['banana_plant']"]
    lineas += [f"# {ln}" for ln in comentario.strip().splitlines()]
    destino.write_text("\n".join(lineas) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raiz", type=Path, default=ROOT / "realdata", help="dónde están los datos")
    ap.add_argument("--salida", type=Path, default=ROOT / "splits", help="dónde escribir los YAML")
    ap.add_argument("--raiz-declarada", help="valor literal de 'path:' en el YAML (por defecto, relativo)")
    ap.add_argument("--incluir-sin-usar", action="store_true", help="añade las 3 fincas nunca usadas")
    ap.add_argument("--listar", action="store_true", help="sólo muestra la agrupación")
    args = ap.parse_args()

    fincas = dict(FINCAS)
    if args.incluir_sin_usar:
        fincas.update(FINCAS_SIN_USAR)

    if args.listar:
        for nombre, f in fincas.items():
            print(f"\n{nombre}")
            for c in f["carpetas"]:
                marca = "" if (args.raiz / c).exists() else "   (no está en disco)"
                print(f"    {c}{marca}")
            print(f"    {f['nota']}")
        return 0

    if not args.raiz.exists():
        print(f"No existe {args.raiz}. Descarga primero con cloud/fetch_data.py", file=sys.stderr)
        return 1

    args.salida.mkdir(parents=True, exist_ok=True)
    if args.raiz_declarada:
        raiz_decl = args.raiz_declarada
    else:
        try:
            raiz_decl = str(Path(args.raiz).resolve().relative_to(args.salida.resolve(), walk_up=True)).replace("\\", "/")
        except (ValueError, TypeError):  # Python < 3.12 no tiene walk_up
            raiz_decl = args.raiz.resolve().as_posix()

    disponibles = {n: carpetas_existentes(args.raiz, f, SPLITS) for n, f in fincas.items()}
    vacias = [n for n, v in disponibles.items() if not v]
    for n in vacias:
        print(f"  aviso: la finca '{n}' no tiene datos en disco, se omite", file=sys.stderr)
        disponibles.pop(n)

    if len(disponibles) < 2:
        print("Hacen falta al menos 2 fincas con datos.", file=sys.stderr)
        return 1

    escritos: list[dict[str, Any]] = []

    # 1) LOFO: una finca fuera cada vez.
    for retenida in disponibles:
        train = [p for n, v in disponibles.items() if n != retenida for p in v]
        val = disponibles[retenida]
        destino = args.salida / f"lofo_{retenida}.yaml"
        escribe_yaml(
            destino, raiz_decl, train, val,
            f"LOFO: entrena con {len(disponibles)-1} fincas y valida en '{retenida}', que queda\n"
            f"ENTERA fuera del entrenamiento (sus 3 splits). Generado por cloud/make_splits.py.\n"
            f"{fincas[retenida]['nota']}",
        )
        escritos.append({"yaml": destino.name, "retenida": retenida, "fuentes_train": len(train), "fuentes_val": len(val)})

    # 2) Entrenamiento completo, validando en el split valid de cada finca (sin fugas
    #    porque ninguna finca se parte: valid nunca entra en train).
    train_todo = [p for v in disponibles.values() for p in v if not p.endswith("/valid/images")]
    val_todo = [p for v in disponibles.values() for p in v if p.endswith("/valid/images")]
    if val_todo:
        escribe_yaml(
            args.salida / "todas_las_fincas.yaml", raiz_decl, train_todo, val_todo,
            "Entrenamiento con todas las fincas disponibles; valida en el split 'valid' de cada una.\n"
            "No mide generalización a finca nueva: para eso están los lofo_*.yaml.",
        )
        escritos.append({"yaml": "todas_las_fincas.yaml", "retenida": None, "fuentes_train": len(train_todo), "fuentes_val": len(val_todo)})

    (args.salida / "splits.json").write_text(
        json.dumps({"raiz_declarada": raiz_decl, "fincas": {n: fincas[n] for n in disponibles}, "generados": escritos}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"{len(escritos)} YAML -> {args.salida}  (path: {raiz_decl})")
    for e in escritos:
        etiqueta = f"retiene {e['retenida']}" if e["retenida"] else "todas"
        print(f"  {e['yaml']:32s} {etiqueta:22s} train:{e['fuentes_train']:3d} fuentes  val:{e['fuentes_val']:3d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
