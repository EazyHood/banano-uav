"""Trocea un ortomosaico grande en tiles para anotar/entrenar.

Los ortomosaicos de dron suelen tener decenas de miles de pixeles por lado y no
caben en una red. Se cortan en tiles con solape. Uso:

    python deep/prepare_dataset.py --image orto.tif --out dataset/images --tile 1024 --overlap 128
"""
from __future__ import annotations

import argparse
import os

import imageio.v2 as imageio


def tile_image(image_path, out_dir, tile=1024, overlap=128):
    os.makedirs(out_dir, exist_ok=True)
    img = imageio.imread(image_path)
    H, W = img.shape[:2]
    step = tile - overlap
    base = os.path.splitext(os.path.basename(image_path))[0]
    n = 0
    for y in range(0, max(1, H - overlap), step):
        for x in range(0, max(1, W - overlap), step):
            y1, x1 = min(y + tile, H), min(x + tile, W)
            crop = img[y:y1, x:x1]
            if crop.shape[0] < tile // 2 or crop.shape[1] < tile // 2:
                continue
            imageio.imwrite(
                os.path.join(out_dir, f"{base}_y{y}_x{x}.png"), crop
            )
            n += 1
    print(f"{n} tiles escritos en {out_dir}")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="dataset/images")
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--overlap", type=int, default=128)
    args = ap.parse_args()
    tile_image(args.image, args.out, args.tile, args.overlap)


if __name__ == "__main__":
    main()
