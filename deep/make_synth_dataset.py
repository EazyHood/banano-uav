"""Genera un dataset sintetico de banano en formato YOLOv8-seg.

Cada macolla se convierte en un poligono de instancia (clase 0 = macolla). Sirve
para entrenar el modelo de deep learning de punta a punta y para el benchmark.

    python deep/make_synth_dataset.py --out dataset --train 160 --val 40 --size 640

NOTA: es un dataset SINTETICO. El modelo entrenado sobre el demuestra el pipeline
completo (datos -> entrenamiento -> inferencia) y sirve de linea base reproducible,
pero para campo real hay que reetiquetar con imagenes reales (ver README / docs).
"""
from __future__ import annotations

import argparse
import os
import sys

import imageio.v2 as imageio
import numpy as np
from skimage import measure, morphology

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from banano.synth import synth_plantation_labeled  # noqa: E402


def mask_to_polygon(mask, tol=1.5):
    """Convierte la mascara de instancia de UNA macolla en un poligono (fila, col).

    Una macolla puede tener varios pseudotallos (hijuelos) cuyas rosetas quedan
    espacialmente separadas -> la mascara puede ser DESCONECTADA. Se toma el CASCO
    CONVEXO de toda la instancia para que el poligono cubra el cluster completo
    (definicion agronomica de macolla) y no queden rosetas sin etiquetar (lo que
    generaria falsos positivos al entrenar).
    """
    if mask.sum() < 25:  # demasiado pequena
        return None
    hull = morphology.convex_hull_image(mask)
    padded = np.pad(hull.astype(float), 1)
    contours = measure.find_contours(padded, 0.5)
    if not contours:
        return None
    c = max(contours, key=len) - 1.0  # deshace el pad
    c = measure.approximate_polygon(c, tolerance=tol)
    if len(c) < 3:
        return None
    return c


def write_tile(img, inst, img_path, lbl_path):
    H, W = inst.shape
    lines = []
    for mat_id in range(1, int(inst.max()) + 1):
        poly = mask_to_polygon(inst == mat_id)
        if poly is None:
            continue
        coords = []
        for r, c in poly:
            x = min(0.999999, max(0.0, c / W))
            y = min(0.999999, max(0.0, r / H))
            coords.append(f"{x:.6f} {y:.6f}")
        if len(coords) >= 3:
            lines.append("0 " + " ".join(coords))
    imageio.imwrite(img_path, img)
    with open(lbl_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


def build(out, n_train, n_val, size, gsd):
    dirs = {}
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            d = os.path.join(out, kind, split)
            os.makedirs(d, exist_ok=True)
            dirs[(kind, split)] = d

    rng = np.random.default_rng(12345)
    total_inst = 0
    plan = [("train", n_train, 0), ("val", n_val, 100000)]
    for split, n, base in plan:
        for i in range(n):
            seed = base + i
            spacing = float(rng.uniform(2.2, 3.0))
            img, inst, _, _, _ = synth_plantation_labeled(
                H=size, W=size, gsd_cm=gsd, spacing_m=spacing, seed=seed
            )
            name = f"{split}_{i:05d}"
            k = write_tile(
                img, inst,
                os.path.join(dirs[("images", split)], name + ".png"),
                os.path.join(dirs[("labels", split)], name + ".txt"),
            )
            total_inst += k
        print(f"  {split}: {n} tiles")

    yaml_path = os.path.join(out, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as fh:
        fh.write(
            f"path: {os.path.abspath(out)}\n"
            "train: images/train\n"
            "val: images/val\n\n"
            "names:\n  0: macolla\n"
        )
    print(f"Dataset listo en '{out}': {n_train}+{n_val} tiles, ~{total_inst} instancias")
    print(f"data.yaml: {yaml_path}")
    return yaml_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--train", type=int, default=160)
    ap.add_argument("--val", type=int, default=40)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--gsd", type=float, default=3.0)
    args = ap.parse_args()
    build(args.out, args.train, args.val, args.size, args.gsd)


if __name__ == "__main__":
    main()
