"""Auditoría de fugas: ¿alguna imagen de validación está dentro del entrenamiento?

Por qué existe (2026-08-23). El repo publicaba mAP50 0.746 como resultado en "otras
4 fincas del entrenamiento" (realdata/holdout.yaml). Al comparar las imágenes por
contenido aparecieron dos cosas:

  1. `extra/prueba2rgb` y `extra/etiquetasnuevas` son EL MISMO DATASET: las 505
     imágenes son idénticas byte a byte. Están subidas a dos workspaces distintos de
     Roboflow (entrenamiento-alterno-dgpgp y tesis-hpmog), y como Roboflow renombra
     cada exportación con un hash propio, el dedup "por stem de Roboflow" que se hizo
     en su día no podía verlas. Resultado: v10 y v12 entrenaron ese material dos veces.
  2. Las 25 imágenes de `prueba2rgb/test` que holdout.yaml usa como VALIDACIÓN están
     el 100% dentro del entrenamiento, vía `etiquetasnuevas/train+valid`. Son 25 de
     las 99 imágenes del protocolo: una cuarta parte de esa cifra se mide sobre
     imágenes ya vistas.

El dedup por hash perceptual (dhash) sí las habría cazado, pero sólo se aplicó dentro
de newfarms/. Este script compara TODO contra TODO, por MD5 del contenido, y encima
resuelve el YAML para decir si la fuga afecta a un protocolo real.

Uso:
    python deep/leak_audit.py                          # matriz de solape entre fuentes
    python deep/leak_audit.py --data realdata/holdout.yaml   # ¿este protocolo está limpio?
    python deep/leak_audit.py --train realdata/v12.yaml --val realdata/holdout_armah.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXTS = {".jpg", ".jpeg", ".png"}


def _ficheros(carpeta: Path) -> list[Path]:
    """Imágenes de una carpeta, sin duplicar por mayúsculas (Windows no distingue)."""
    vistos: dict[str, Path] = {}
    for p in carpeta.rglob("*"):
        if p.suffix.lower() in EXTS:
            vistos.setdefault(os.path.normcase(str(p)), p)
    return list(vistos.values())


def huellas(carpeta: Path, cache: dict[str, dict[str, str]]) -> dict[str, str]:
    """MD5 -> nombre de fichero. Cachea por carpeta: hashear 20k imágenes cuesta."""
    clave = str(carpeta)
    if clave in cache:
        return cache[clave]
    out: dict[str, str] = {}
    for p in _ficheros(carpeta):
        try:
            out[hashlib.md5(p.read_bytes()).hexdigest()] = p.name
        except OSError:
            continue
    cache[clave] = out
    return out


def fuentes_de_yaml(ruta: Path, clave: str) -> list[Path]:
    import yaml

    cfg = yaml.safe_load(ruta.read_text(encoding="utf-8"))
    base = Path(cfg.get("path", ruta.parent))
    if not base.is_absolute():
        base = ruta.parent / base
    v = cfg.get(clave)
    if not v:
        return []
    return [base / item for item in (v if isinstance(v, list) else [v])]


def compara(train: list[Path], val: list[Path], cache: dict[str, dict[str, str]]) -> dict[str, Any]:
    h_train: dict[str, str] = {}
    for c in train:
        h_train.update(huellas(c, cache))

    detalle = []
    total_val = 0
    total_fuga = 0
    for c in val:
        h = huellas(c, cache)
        fuga = set(h) & set(h_train)
        total_val += len(h)
        total_fuga += len(fuga)
        detalle.append(
            {
                "fuente": str(c),
                "imgs": len(h),
                "fugadas": len(fuga),
                "pct": round(len(fuga) / max(len(h), 1) * 100, 1),
                "ejemplos": [h[k] for k in list(fuga)[:3]],
            }
        )
    return {
        "imgs_validacion": total_val,
        "imgs_fugadas": total_fuga,
        "pct_fugado": round(total_fuga / max(total_val, 1) * 100, 1),
        "por_fuente": detalle,
    }


def matriz(raiz: Path, cache: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    """Todos los pares de datasets que comparten alguna imagen idéntica."""
    carpetas: list[Path] = []
    for patron in ("*", "extra/*", "newfarms/*"):
        carpetas += [p for p in sorted(raiz.glob(patron)) if p.is_dir()]
    # quitar contenedoras (realdata/extra contiene a todas sus hijas)
    carpetas = [c for c in carpetas if not any(o != c and c in o.parents for o in carpetas)]

    hs: dict[Path, dict[str, str]] = {}
    for c in carpetas:
        h = huellas(c, cache)
        if h:
            hs[c] = h

    pares: list[dict[str, Any]] = []
    nombres = list(hs)
    for i, a in enumerate(nombres):
        for b in nombres[i + 1 :]:
            comun = set(hs[a]) & set(hs[b])
            if comun:
                pares.append(
                    {
                        "a": a.relative_to(raiz).as_posix(),
                        "b": b.relative_to(raiz).as_posix(),
                        "comunes": len(comun),
                        "pct_a": round(len(comun) / len(hs[a]) * 100, 1),
                        "pct_b": round(len(comun) / len(hs[b]) * 100, 1),
                    }
                )
    pares.sort(key=lambda p: -p["comunes"])
    return pares


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raiz", type=Path, default=ROOT / "realdata")
    ap.add_argument("--data", type=Path, help="un data.yaml: compara su train contra su val")
    ap.add_argument("--train", type=Path, help="yaml del que sale el train")
    ap.add_argument("--val", type=Path, help="yaml del que sale la validación")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    cache: dict[str, dict[str, str]] = {}
    salida: dict[str, Any] = {}

    if args.data or (args.train and args.val):
        y_train = args.train or args.data
        y_val = args.val or args.data
        train = fuentes_de_yaml(y_train, "train")
        val = fuentes_de_yaml(y_val, "val")
        print(f"train: {y_train.name} ({len(train)} fuentes)")
        print(f"val  : {y_val.name} ({len(val)} fuentes)\n")
        r = compara(train, val, cache)
        salida["protocolo"] = r
        for d in r["por_fuente"]:
            marca = "  <-- FUGA" if d["fugadas"] else ""
            print(f"  {Path(d['fuente']).parent.name}/{Path(d['fuente']).name:12s} "
                  f"{d['imgs']:5d} imgs  {d['fugadas']:5d} fugadas ({d['pct']:5.1f}%){marca}")
        print(f"\n  TOTAL: {r['imgs_fugadas']}/{r['imgs_validacion']} imágenes de validación "
              f"están en el entrenamiento ({r['pct_fugado']}%)")
        if r["imgs_fugadas"]:
            print("  Este protocolo NO mide generalización sobre esas imágenes.")
    else:
        print("Pares de datasets que comparten imágenes idénticas (MD5 del contenido):\n")
        pares = matriz(args.raiz, cache)
        salida["pares"] = pares
        if not pares:
            print("  ninguno.")
        for p in pares:
            print(f"  {p['a']:34s} <-> {p['b']:34s} {p['comunes']:6d} comunes "
                  f"({p['pct_a']:.0f}% / {p['pct_b']:.0f}%)")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nInforme -> {args.json}")

    fugado = salida.get("protocolo", {}).get("imgs_fugadas", 0) or len(salida.get("pares", []))
    return 2 if fugado else 0


if __name__ == "__main__":
    raise SystemExit(main())
