"""Ejecuta la deteccion de banano sobre una imagen/ortomosaico real de dron.

Uso:
    python scripts/run.py --image ruta/a/tu_imagen.jpg --gsd 3.0 --out outputs

--gsd es la resolucion en cm/pixel (muy recomendable). Se calcula a partir de la
altura de vuelo y la camara, o lo entrega el software de fotogrametria (Pix4D,
Agisoft, OpenDroneMap) al generar el ortomosaico.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import imageio.v2 as imageio  # noqa: E402

from banano.pipeline import detect_banana  # noqa: E402
from banano.visualize import overlay  # noqa: E402


def main():
    ap = argparse.ArgumentParser(
        description="Identificacion de cultivo de banano en imagenes de dron"
    )
    ap.add_argument("--image", required=True, help="Ruta a la imagen RGB")
    ap.add_argument(
        "--gsd", type=float, default=None, help="Resolucion en cm/pixel"
    )
    ap.add_argument("--out", default="outputs", help="Carpeta de salida")
    ap.add_argument(
        "--mode",
        default="both",
        choices=["bright", "dark", "both"],
        help="Tipo de centro de simetria radial",
    )
    ap.add_argument(
        "--no-mask",
        action="store_true",
        help="No restringir al dosel segmentado (util si la segmentacion falla)",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.25,
        help="Umbral relativo de los picos de simetria (0-1)",
    )
    args = ap.parse_args()

    img = imageio.imread(args.image)
    res = detect_banana(
        img,
        gsd_cm=args.gsd,
        mode=args.mode,
        use_mask=not args.no_mask,
        rel_threshold=args.threshold,
    )

    os.makedirs(args.out, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.image))[0]
    overlay(img, res, out_path=os.path.join(args.out, base + "_deteccion.png"))

    report = {"resumen": res.summary(), "params": {k: str(v) for k, v in res.params.items()}}
    with open(
        os.path.join(args.out, base + "_reporte.json"), "w", encoding="utf-8"
    ) as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
