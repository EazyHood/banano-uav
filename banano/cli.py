"""CLI de banano-drone: ortomosaico de dron -> conteo + capa GIS + informe.

Punto de entrada instalado como ``banano-detect``. Ejemplos:

    banano-detect --input orto.tif --out resultados
    banano-detect --input foto.jpg --gsd 3.0 --out resultados --tile 1024 --overlap 128
    banano-detect --input orto.tif --config mi_config.yaml -v

Codigos de salida: 0 = ok, 1 = error de entrada/configuracion, 2 = error inesperado,
130 = interrumpido por el usuario.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

from . import __version__
from .config import PipelineConfig
from .errors import BananoError, InputError
from .geo import _HAS_RIO, Raster
from .logconf import setup_logging
from .ortho import process_orthomosaic
from .report import write_all

logger = logging.getLogger(__name__)

_IMG_EXT = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".jp2", ".bmp", ".vrt", ".img")


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
    ap.add_argument(
        "--gsd", type=float, default=None, help="Resolucion cm/pixel (auto si es GeoTIFF)"
    )
    ap.add_argument("--config", "-c", default=None, help="Archivo YAML de configuracion (opcional)")
    ap.add_argument("--tile", type=int, default=None, help="Tamano de tile en pixeles")
    ap.add_argument("--overlap", type=int, default=None, help="Solape entre tiles en pixeles")
    ap.add_argument("--mode", default=None, choices=["bright", "dark", "both"])
    ap.add_argument("--threshold", type=float, default=None, help="Umbral relativo de picos (0-1)")
    ap.add_argument("--no-mask", action="store_true", help="No restringir al dosel segmentado")
    ap.add_argument("-v", "--verbose", action="count", default=0, help="Mas detalle (-v, -vv)")
    ap.add_argument("--quiet", action="store_true", help="Solo avisos y errores")
    ap.add_argument("--version", action="version", version=f"banano-drone {__version__}")
    return ap


def _load_config(args) -> PipelineConfig:
    """Construye la configuracion: archivo YAML como base + overrides de la CLI."""
    if args.config:
        if not os.path.isfile(args.config):
            raise InputError(f"El archivo de configuracion no existe: {args.config}")
        cfg = PipelineConfig.from_yaml(args.config)
    else:
        cfg = PipelineConfig()
    # los flags de la CLI (si se pasaron) tienen prioridad sobre el YAML
    if args.gsd is not None:
        cfg.gsd_cm = args.gsd
    if args.tile is not None:
        cfg.tile = args.tile
    if args.overlap is not None:
        cfg.overlap = args.overlap
    if args.mode is not None:
        cfg.mode = args.mode
    if args.threshold is not None:
        cfg.rel_threshold = args.threshold
    if args.no_mask:
        cfg.use_mask = False
    return cfg.validate()


def run(args) -> int:
    cfg = _load_config(args)

    if not os.path.isfile(args.input):
        raise InputError(f"El archivo de entrada no existe: {args.input}")
    if not args.input.lower().endswith(_IMG_EXT):
        logger.warning("Extension no habitual '%s'; se intentara abrir de todos modos.", args.input)

    raster = Raster(args.input, gsd_cm=cfg.gsd_cm)
    logger.info(
        "Ortomosaico %dx%d px | georreferencia: %s%s",
        raster.width,
        raster.height,
        "si" if raster.has_geo else "no",
        f" | GSD auto {raster.gsd_auto:.2f} cm/px" if raster.gsd_auto else "",
    )

    if not raster.has_geo and not cfg.gsd_cm:
        logger.warning(
            "Sin GeoTIFF ni --gsd; se estimara la escala del marco de siembra "
            "(menos preciso). Recomendado pasar --gsd."
        )
    if not _HAS_RIO and args.input.lower().endswith((".tif", ".tiff")):
        logger.warning(
            "'rasterio' no esta instalado; el GeoTIFF se leera sin georreferencia. "
            "Instala:  pip install banano-drone[geo]"
        )

    try:
        result = process_orthomosaic(raster, config=cfg, progress=None if args.quiet else _progress)
        paths = write_all(args.out, result, raster=raster, input_name=args.input)
    finally:
        raster.close()

    print("\n" + json.dumps(result.summary(), indent=2, ensure_ascii=False))
    print(f"\nEntregables en '{args.out}':")
    for v in paths.values():
        print(f"  - {v}")
    print(f"\nAbre el informe: {paths['html']}")
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(verbosity=args.verbose, quiet=args.quiet)
    t0 = time.time()
    try:
        code = run(args)
        logger.info("Terminado en %.1fs", time.time() - t0)
        return code
    except InputError as e:
        logger.error("Entrada invalida: %s", e)
        return 1
    except BananoError as e:
        logger.error("%s", e)
        return 1
    except KeyboardInterrupt:  # pragma: no cover
        logger.error("Interrumpido por el usuario.")
        return 130
    except Exception as e:  # noqa: BLE001 — ultimo recurso, no debe filtrar traceback crudo
        logger.exception("Error inesperado: %s", e)
        logger.error("Por favor reporta esto en https://github.com/EazyHood/banano-uav/issues")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
