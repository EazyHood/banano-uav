"""Auditoría de escala: ¿de qué TAMAÑO es una planta en cada finca?

Por qué existe (2026-08-23). El modelo multi-finca v10 saca mAP50 0.861 en su finca
y 0.172 en armah (finca nunca vista). Se venía leyendo como "domain gap" visual —
otro suelo, otra luz, otra variedad. Al medir el tamaño de las cajas resultó ser,
en primer lugar, un problema de ESCALA:

    a imgsz 768, la planta mediana mide 10 px en m2 y 333 px en lasuiza (33x).

Y peor: el entrenamiento de v10 está dominado por instancias diminutas. Contando
CAJAS (que es lo que ve la función de pérdida, no las imágenes):

    plantas_jovenes_80m1 + plantas_platano = 65.659 cajas de ~16-17 px
    count_banana_plants                    =  8.601 cajas de ~169 px

El 80% de lo que el modelo aprendió son plantas de 16 píxeles. armah las tiene de
45 px: cae en tierra de nadie entre los dos modos del entrenamiento. Eso explica
un recall de 0.139 mucho mejor que "el suelo de Ghana es distinto", y explica que
v12 empeorase armah (añadió masa en los extremos —m2 10 px, lasuiza 333 px— no en
el centro donde vive armah).

Consecuencia práctica: `scale=0.6` de ultralytics cubre un rango de ~4x (0.4x-1.6x).
Para cubrir 33x hace falta remuestrear los datos, no sólo aumentar.

Uso:
    python deep/scale_audit.py                       # audita todas las fuentes conocidas
    python deep/scale_audit.py --data realdata/v12.yaml
    python deep/scale_audit.py --imgsz 1024 --json out.json

No usa GPU ni carga imágenes completas: lee las cabeceras con PIL y los .txt de
etiquetas. Tarda segundos.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import statistics as st
import sys
from pathlib import Path
from typing import Any

try:
    from PIL import Image
except ImportError:  # pragma: no cover - dependencia opcional
    print("Falta Pillow: pip install pillow", file=sys.stderr)
    raise

ROOT = Path(__file__).resolve().parents[1]

# Muestreo: leer 40 cabeceras de imagen y 200 ficheros de etiquetas basta para la
# mediana; leerlo todo en datasets de 2.000+ imágenes no cambia el diagnóstico.
MUESTRA_IMGS = 40
MUESTRA_LABELS = 200
EXTS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG")

# El stride mayor de YOLO es 32 y el menor 8: por debajo de ~8 px un objeto no tiene
# ni una celda propia en el mapa de características más fino.
PX_INDETECTABLE = 8.0


def _unicos(rutas: list[str]) -> list[str]:
    """Windows no distingue mayúsculas: *.jpg y *.JPG devuelven los MISMOS ficheros.

    Sin esta deduplicación cada imagen se contaba dos veces y los totales salían
    exactamente al doble (702 imágenes se reportaban como 1404).
    """
    vistos: dict[str, str] = {}
    for r in rutas:
        vistos.setdefault(os.path.normcase(os.path.abspath(r)), r)
    return list(vistos.values())


def _imagenes(carpeta: Path) -> list[str]:
    fs: list[str] = []
    for ext in EXTS:
        fs += glob.glob(str(carpeta / "**" / "images" / ext), recursive=True)
    if not fs:  # carpeta que ya es .../images
        for ext in EXTS:
            fs += glob.glob(str(carpeta / "**" / ext), recursive=True)
    return _unicos(fs)


def _etiquetas(carpeta: Path) -> list[str]:
    fs = glob.glob(str(carpeta / "**" / "labels" / "*.txt"), recursive=True)
    return _unicos([f for f in fs if not f.endswith("classes.txt")])


def audita_fuente(carpeta: Path, imgsz: int, rng: random.Random) -> dict[str, Any] | None:
    """Mide tamaño de imagen y de objeto de una carpeta estilo Roboflow/YOLO."""
    imgs = _imagenes(carpeta)
    txts = _etiquetas(carpeta)
    if not imgs or not txts:
        return None

    dims: list[tuple[int, int]] = []
    for p in rng.sample(imgs, min(MUESTRA_IMGS, len(imgs))):
        try:
            with Image.open(p) as im:
                dims.append(im.size)
        except Exception:  # imagen corrupta: no debe tumbar la auditoría
            continue
    if not dims:
        return None
    w_img = st.median([w for w, _ in dims])
    h_img = st.median([h for _, h in dims])

    lados_px: list[float] = []
    clases: set[str] = set()
    cajas_total = 0
    vacias = 0
    muestra_txt = rng.sample(txts, min(MUESTRA_LABELS, len(txts)))
    for t in muestra_txt:
        try:
            lineas = [ln.split() for ln in Path(t).read_text().splitlines() if ln.strip()]
        except Exception:
            continue
        if not lineas:
            vacias += 1
        cajas_total += len(lineas)
        for campos in lineas:
            if len(campos) < 5:
                continue
            clases.add(campos[0])
            try:
                w_rel, h_rel = float(campos[3]), float(campos[4])
            except ValueError:
                continue
            # lado equivalente del cuadrado de la misma área, en píxeles reales
            lados_px.append(((w_rel * w_img) * (h_rel * h_img)) ** 0.5)

    if not lados_px:
        return None

    # ultralytics reescala el lado LARGO a imgsz manteniendo la proporción
    factor = imgsz / max(w_img, h_img)
    lados_768 = [x * factor for x in lados_px]
    lados_768.sort()

    def pct(p: float) -> float:
        i = min(int(p * (len(lados_768) - 1)), len(lados_768) - 1)
        return lados_768[i]

    return {
        "carpeta": str(carpeta.relative_to(ROOT)) if carpeta.is_relative_to(ROOT) else str(carpeta),
        "imgs": len(imgs),
        "ficheros_etiqueta": len(txts),
        "etiquetas_vacias_en_muestra": vacias,
        "cajas_por_img": round(cajas_total / max(len(muestra_txt), 1), 1),
        "cajas_estimadas": int(round(cajas_total / max(len(muestra_txt), 1) * len(txts))),
        "img_w": int(w_img),
        "img_h": int(h_img),
        "clases": sorted(clases),
        "planta_px_real": round(st.median(lados_px), 1),
        "planta_px_imgsz": round(st.median(lados_768), 1),
        "p10_px_imgsz": round(pct(0.10), 1),
        "p90_px_imgsz": round(pct(0.90), 1),
        "frac_indetectable": round(
            sum(1 for x in lados_768 if x < PX_INDETECTABLE) / len(lados_768), 4
        ),
    }


def fuentes_de_yaml(ruta_yaml: Path) -> list[Path]:
    """Saca las carpetas de un data.yaml de ultralytics sin depender de pyyaml."""
    import yaml  # dependencia ya presente en el proyecto

    cfg = yaml.safe_load(ruta_yaml.read_text(encoding="utf-8"))
    base = Path(cfg.get("path", ruta_yaml.parent))
    if not base.is_absolute():
        base = ruta_yaml.parent / base
    fuera: list[Path] = []
    for clave in ("train", "val", "test"):
        v = cfg.get(clave)
        if not v:
            continue
        for item in v if isinstance(v, list) else [v]:
            p = base / item
            # .../split/images -> queremos el nivel del split para hallar labels/
            fuera.append(p.parent if p.name == "images" else p)
    return fuera


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, help="data.yaml de ultralytics a auditar")
    ap.add_argument("--raiz", type=Path, default=ROOT / "realdata", help="raíz de datasets")
    ap.add_argument("--imgsz", type=int, default=768, help="resolución de entrenamiento/inferencia")
    ap.add_argument("--json", type=Path, help="volcar el informe a JSON")
    ap.add_argument("--semilla", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.semilla)

    if args.data:
        candidatas = fuentes_de_yaml(args.data)
    else:
        candidatas = []
        for patron in ("*", "extra/*", "newfarms/*"):
            candidatas += [p for p in sorted(args.raiz.glob(patron)) if p.is_dir()]

    # Una carpeta contenedora (realdata/extra) tiene dentro TODAS las de sus hijas.
    # Auditarla además de sus hijas duplicaba la masa: se contaba cada caja dos veces.
    candidatas = [c for c in candidatas if c.exists()]
    contenedoras = {c for c in candidatas if any(o != c and c in o.parents for o in candidatas)}
    if contenedoras:
        print(
            "Omitidas por contener otras fuentes: "
            + ", ".join(sorted(p.name for p in contenedoras)),
            file=sys.stderr,
        )

    informe: list[dict[str, Any]] = []
    vistas: set[Path] = set()
    for c in candidatas:
        if c in vistas or c in contenedoras:
            continue
        vistas.add(c)
        r = audita_fuente(c, args.imgsz, rng)
        if r:
            informe.append(r)

    if not informe:
        print("No se encontró ninguna fuente con imágenes y etiquetas.", file=sys.stderr)
        return 1

    informe.sort(key=lambda r: r["planta_px_imgsz"])

    cab = (
        f"{'fuente':32s} {'imgs':>6s} {'cajas':>8s} {'c/img':>6s} "
        f"{'img':>10s} {'planta px':>10s} {'p10-p90':>13s} {'<8px':>6s}"
    )
    print(f"\nAuditoría de escala a imgsz={args.imgsz}\n")
    print(cab)
    print("-" * len(cab))
    for r in informe:
        nombre = r["carpeta"].replace("realdata\\", "").replace("realdata/", "")
        print(
            f"{nombre[:32]:32s} {r['imgs']:6d} {r['cajas_estimadas']:8d} {r['cajas_por_img']:6.1f} "
            f"{r['img_w']}x{r['img_h']:<5d} {r['planta_px_imgsz']:8.0f}px "
            f"{r['p10_px_imgsz']:5.0f}-{r['p90_px_imgsz']:<6.0f} {r['frac_indetectable']*100:5.1f}%"
        )

    menor = informe[0]
    mayor = informe[-1]
    razon = mayor["planta_px_imgsz"] / max(menor["planta_px_imgsz"], 1e-6)
    print(
        f"\nRango de escala: {razon:.0f}x  "
        f"({menor['carpeta']} {menor['planta_px_imgsz']:.0f}px  ->  "
        f"{mayor['carpeta']} {mayor['planta_px_imgsz']:.0f}px)"
    )

    # Dónde está la masa: el modelo aprende de CAJAS, no de imágenes.
    total_cajas = sum(r["cajas_estimadas"] for r in informe)
    if total_cajas:
        acum = 0
        print("\nDistribución de la masa de entrenamiento (por caja, acumulada):")
        for r in informe:
            acum += r["cajas_estimadas"]
            print(
                f"  hasta {r['planta_px_imgsz']:6.0f} px  {acum/total_cajas*100:5.1f}%  "
                f"(+{r['cajas_estimadas']:>6d} de {r['carpeta']})"
            )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {"imgsz": args.imgsz, "rango_escala": razon, "fuentes": informe},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"\nInforme -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
