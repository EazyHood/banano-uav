"""Genera un GeoTIFF sintetico georreferenciado de una plantacion de banano.

Sirve para probar el flujo completo (GeoTIFF -> conteo georreferenciado -> informe)
sin necesidad de un vuelo real, y como ejemplo del repositorio.

    python scripts/make_example.py --out example/plantacion_banano.tif

Requiere el extra 'geo':  pip install -e .[geo]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from banano.synth import synth_plantation  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="example/plantacion_banano.tif")
    ap.add_argument("--size", type=int, default=1600, help="Lado en px")
    ap.add_argument("--gsd", type=float, default=3.0, help="cm/px")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    try:
        import rasterio
        from rasterio.transform import from_origin
    except ImportError as e:
        raise SystemExit("Falta rasterio. Instala:  pip install -e .[geo]") from e

    img, gt_ps, gt_mats, meta = synth_plantation(
        H=args.size, W=args.size, gsd_cm=args.gsd, seed=args.seed
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    # Transformada UTM ficticia (zona 18N, Colombia). Origen arbitrario en metros.
    res_m = args.gsd / 100.0
    transform = from_origin(500000.0, 500000.0 + args.size * res_m, res_m, res_m)
    crs = "EPSG:32618"  # WGS84 / UTM zona 18N

    data = np.moveaxis(img, -1, 0)  # (3, H, W)
    with rasterio.open(
        args.out, "w", driver="GTiff",
        height=args.size, width=args.size, count=3, dtype="uint8",
        crs=crs, transform=transform, compress="deflate",
    ) as ds:
        ds.write(data)

    print(f"GeoTIFF escrito: {args.out}")
    print(f"  {args.size}x{args.size} px · GSD {args.gsd} cm/px · CRS {crs}")
    print(f"  verdad de terreno: {len(gt_ps)} pseudotallos, {len(gt_mats)} macollas")


if __name__ == "__main__":
    main()
