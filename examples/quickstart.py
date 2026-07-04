"""Ejemplo mínimo ejecutable: de una imagen sintética al conteo + entregables.

    python examples/quickstart.py

No requiere datos reales: genera una plantación sintética, corre el pipeline y
escribe los entregables en ./quickstart_out/.
"""
from __future__ import annotations

import os
import sys
import tempfile

import imageio.v2 as imageio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from banano import PipelineConfig, Raster, process_orthomosaic
from banano.report import write_all
from banano.synth import synth_plantation


def main():
    # 1. una imagen de ejemplo (en la práctica: tu ortomosaico GeoTIFF)
    img, _gt_ps, gt_mats, _ = synth_plantation(H=1200, W=1200, gsd_cm=3.0, seed=42)
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "ejemplo.png")
    imageio.imwrite(path, img)

    # 2. configuración reproducible
    cfg = PipelineConfig(gsd_cm=3.0, tile=640, overlap=96).validate()

    # 3. procesar
    raster = Raster(path, gsd_cm=3.0)
    res = process_orthomosaic(raster, config=cfg)

    # 4. entregables + resumen
    out = os.path.join(os.getcwd(), "quickstart_out")
    paths = write_all(out, res, raster=raster, input_name=path)
    raster.close()

    print("Resumen:", res.summary())
    print(f"Verdad de terreno (sintética): {len(gt_mats)} macollas")
    print(f"Detectado: {res.n_mats} macollas")
    print("Entregables:")
    for v in paths.values():
        print("  -", v)


if __name__ == "__main__":
    main()
