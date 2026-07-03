"""CLI de banano-drone: ortomosaico de dron -> conteo + capa GIS + informe.

Punto de entrada instalado como ``banano-detect``. Ejemplos:

    banano-detect --input orto.tif --out resultados
    banano-detect --input foto.jpg --gsd 3.0 --out resultados --tile 1024 --overlap 128

Acepta GeoTIFF (detecta el GSD y la georreferencia solo) o una imagen normal (pasa --gsd).
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from .geo import Raster, _HAS_RIO
from .ortho import process_orthomosaic
from .report import write_all


def _progress(done, total):
    pct = int(100 * done / total) if total else 100
    bar = "#" * (pct // 4) + "-" * (25 - pct // 4)
    sys.stderr.write(f"\r  procesando tiles [{bar}] {done}/{total}")
    sys.stderr.flush()
    if done == total:
        sys.stderr.write("\n")


def build_parser():
    ap = argparse.ArgumentParser(
        prog="banano-detect",
        description="Identifica y cuenta cultivo de banano en ortomosaicos de dron.",
    )
    ap.add_argument("--input", "-i", required=True, help="GeoTIFF u imagen RGB del ortomosaico")
    ap.add_argument("--out", "-o", default="resultados", help="Carpeta de salida")
    ap.add_argument("--gsd", type=float, default=None, help="Resolución cm/píxel (auto si es GeoTIFF)")
    ap.add_argument("--tile", type=int, default=1024, help="Tamaño de tile en píxeles")
    ap.add_argument("--overlap", type=int, default=128, help="Solape entre tiles en píxeles")
    ap.add_argument("--mode", default="both", choices=["bright", "dark", "both"])
    ap.add_argument("--threshold", type=float, default=0.25, help="Umbral relativo de picos (0-1)")
    ap.add_argument("--no-mask", action="store_true", help="No restringir al dosel segmentado")
    ap.add_argument("--quiet", action="store_true", help="Sin barra de progreso")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    t0 = time.time()

    raster = Raster(args.input, gsd_cm=args.gsd)
    print(f"Ortomosaico: {raster.width}x{raster.height} px · georreferencia: "
          f"{'sí' if raster.has_geo else 'no'}"
          + (f" · GSD auto {raster.gsd_auto:.2f} cm/px" if raster.gsd_auto else ""))
    if not raster.has_geo and not args.gsd:
        print("  aviso: sin GeoTIFF ni --gsd; se estimará la escala del marco de siembra "
              "(menos preciso). Recomendado pasar --gsd.", file=sys.stderr)
    if not _HAS_RIO and args.input.lower().endswith((".tif", ".tiff")):
        print("  aviso: 'rasterio' no está instalado; el GeoTIFF se leerá sin georreferencia. "
              "Instala:  pip install banano-drone[geo]", file=sys.stderr)

    result = process_orthomosaic(
        raster,
        gsd_cm=args.gsd,
        tile=args.tile,
        overlap=args.overlap,
        mode=args.mode,
        rel_threshold=args.threshold,
        use_mask=not args.no_mask,
        progress=None if args.quiet else _progress,
    )

    paths = write_all(args.out, result, raster=raster, input_name=args.input)
    raster.close()

    print("\n" + json.dumps(result.summary(), indent=2, ensure_ascii=False))
    print(f"\nEntregables en '{args.out}':")
    for k, v in paths.items():
        print(f"  - {v}")
    print(f"\nListo en {time.time() - t0:.1f}s. Abre el informe: {paths['html']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
