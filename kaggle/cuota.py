"""Cuánta cuota de GPU queda en Kaggle, y cuántas horas de reloj caben en ella.

Por qué existe. El 24 de agosto de 2026 se lanzó una tirada de 10,6 h de reloj con 14,6 h
de cuota libres, y Kaggle la canceló a mitad: la cuota se agotó antes que el reloj. Se
perdió la corrida entera y la cuota de la semana. Nadie miraba la cuota al lanzar, sólo el
límite de 12 h de la sesión, que es otra cosa.

**La cuota se gasta más rápido que el reloj.** Kaggle asigna DOS Tesla T4 —no hay opción de
una sola cuando pides GPU— y cobra por GPU asignada, no por GPU usada; `cloud/train.py` usa
una sola desde que se arregló lo del DDP. Medido con tres lecturas de `kaggle quota`:

    24-ago 15:28 -> 21:33 :  6,08 h de reloj, 9,88 h de cuota = 1,62x
    24-ago 21:33 -> 25-ago 10:17 : 12,73 h de reloj, 15,02 h de cuota = 1,18x

Se toma 1,7x, el peor caso con margen. Traducido: **las 30 h de cuota semanal son unas
17 h de reloj**, no 30. Una sesión de 11 h se come dos tercios de la semana.

    python kaggle/cuota.py            # cuánto queda y qué cabe
    python kaggle/cuota.py --horas 6  # ¿cabe una tirada de 6 h?
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Peor caso medido, con margen. Si algún día Kaggle deja pedir una sola T4, bajará a ~1.0.
CUOTA_POR_HORA_DE_RELOJ = 1.7
MARGEN_H = 0.3  # para el arranque, la descarga de datos y el guardado final


def cli() -> str:
    cand = Path(sys.executable).parent / ("kaggle.exe" if sys.platform == "win32" else "kaggle")
    return str(cand) if cand.exists() else "kaggle"


def lee() -> tuple[float, float] | None:
    """(horas usadas, horas libres) de la cuota semanal de GPU, o None si no se puede leer."""
    r = subprocess.run([cli(), "quota"], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    for linea in r.stdout.splitlines():
        campos = linea.split()
        if len(campos) >= 3 and campos[0].upper() == "GPU":
            try:
                return float(campos[1].rstrip("h")), float(campos[2].rstrip("h"))
            except ValueError:
                return None
    return None


def caben_horas(libres: float) -> float:
    """Horas de RELOJ que caben en la cuota libre, ya descontado el margen."""
    return max(0.0, libres / CUOTA_POR_HORA_DE_RELOJ - MARGEN_H)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--horas", type=float, default=None,
                    help="comprueba si cabe una tirada de estas horas; sale 2 si no cabe")
    args = ap.parse_args()

    q = lee()
    if not q:
        print("No pude leer la cuota de Kaggle.", file=sys.stderr)
        return 1

    usadas, libres = q
    caben = caben_horas(libres)
    print(f"GPU: {usadas:.2f} h usadas de 30, {libres:.2f} h libres")
    print(f"Caben ~{caben:.1f} h de reloj  (la cuota se gasta ~{CUOTA_POR_HORA_DE_RELOJ}x: "
          "Kaggle asigna dos T4 y cobra por las dos)")

    if libres <= 0.01:
        print("\nCUOTA AGOTADA. Se renueva el sábado a medianoche UTC "
              "(las 19:00 del viernes en Colombia).")
        return 2

    if args.horas is not None:
        necesita = (args.horas + MARGEN_H) * CUOTA_POR_HORA_DE_RELOJ
        if necesita > libres:
            print(f"\nNO CABE una tirada de {args.horas} h: pediría ~{necesita:.1f} h de cuota "
                  f"y sólo hay {libres:.2f} h. Kaggle la cancelaría a mitad.")
            return 2
        print(f"\nCabe: una tirada de {args.horas} h pediría ~{necesita:.1f} h de las "
              f"{libres:.2f} h libres.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
