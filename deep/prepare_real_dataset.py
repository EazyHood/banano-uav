"""Prepara un dataset YOLO real de banano para ENTRENAR UN DETECTOR DE PLANTAS.

Muchos datasets reales de banano aereo estan etiquetados por ENFERMEDAD (p.ej.
sano / Moko / Sigatoka). Para un contador de plantas, colapsamos todas las clases
a una sola: "banano". Este script:

  1. Localiza las particiones (train/val/test) y sus imagenes + etiquetas YOLO.
  2. Reescribe cada etiqueta poniendo el indice de clase = 0 (una sola clase).
  3. Escribe un data.yaml nuevo con names: {0: banano}.

No copia imagenes (usa las existentes); solo crea etiquetas colapsadas y el yaml.

    python deep/prepare_real_dataset.py --root realdata/DS-v1 --out realdata/banano_yolo
"""
from __future__ import annotations

import argparse
import os
import shutil

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def find_splits(root):
    """Devuelve dict split -> (images_dir, labels_dir) detectando la estructura YOLO."""
    splits = {}
    # patron 1: root/images/train, root/labels/train
    for split in ("train", "val", "valid", "test"):
        img = os.path.join(root, "images", split)
        lbl = os.path.join(root, "labels", split)
        if os.path.isdir(img) and os.path.isdir(lbl):
            splits[split] = (img, lbl)
    if splits:
        return splits
    # patron 2: root/train/images, root/train/labels
    for split in ("train", "val", "valid", "test"):
        img = os.path.join(root, split, "images")
        lbl = os.path.join(root, split, "labels")
        if os.path.isdir(img) and os.path.isdir(lbl):
            splits[split] = (img, lbl)
    return splits


def collapse_labels(src_lbl_dir, dst_lbl_dir):
    os.makedirs(dst_lbl_dir, exist_ok=True)
    n_files = n_boxes = 0
    for name in os.listdir(src_lbl_dir):
        if not name.endswith(".txt"):
            continue
        out_lines = []
        with open(os.path.join(src_lbl_dir, name), encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 5:
                    continue
                parts[0] = "0"  # colapsa la clase a 0 = banano
                out_lines.append(" ".join(parts))
                n_boxes += 1
        with open(os.path.join(dst_lbl_dir, name), "w", encoding="utf-8") as fh:
            fh.write("\n".join(out_lines) + ("\n" if out_lines else ""))
        n_files += 1
    return n_files, n_boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Raiz del dataset YOLO extraido")
    ap.add_argument("--out", required=True, help="Carpeta de salida (etiquetas colapsadas + yaml)")
    args = ap.parse_args()

    splits = find_splits(args.root)
    if not splits:
        raise SystemExit(
            f"No se encontro estructura YOLO (images/labels) bajo {args.root}. "
            "Revisa la estructura extraida."
        )

    os.makedirs(args.out, exist_ok=True)
    yaml_splits = {}
    for split, (img_dir, lbl_dir) in splits.items():
        dst_lbl = os.path.join(args.out, "labels", split)
        nf, nb = collapse_labels(lbl_dir, dst_lbl)
        # enlazamos las imagenes originales via ruta absoluta en el yaml
        yaml_splits[split] = (os.path.abspath(img_dir), os.path.abspath(dst_lbl))
        print(f"  {split}: {nf} etiquetas, {nb} cajas -> clase unica 'banano'")

    # ultralytics espera que las etiquetas esten junto a las imagenes o via 'path'.
    # Creamos una estructura estandar images/labels por split con symlink/copia de imagenes.
    data_root = os.path.join(args.out, "data")
    for split, (img_dir, dst_lbl) in yaml_splits.items():
        di = os.path.join(data_root, "images", split)
        dl = os.path.join(data_root, "labels", split)
        os.makedirs(di, exist_ok=True)
        os.makedirs(dl, exist_ok=True)
        # copia etiquetas
        for n in os.listdir(dst_lbl):
            shutil.copy2(os.path.join(dst_lbl, n), os.path.join(dl, n))
        # copia imagenes (mismo basename)
        for n in os.listdir(img_dir):
            if n.lower().endswith(IMG_EXT):
                shutil.copy2(os.path.join(img_dir, n), os.path.join(di, n))

    val_key = "val" if "val" in splits else ("valid" if "valid" in splits else "test")
    train_key = "train" if "train" in splits else list(splits)[0]
    yaml_path = os.path.join(args.out, "banano_real.yaml")
    with open(yaml_path, "w", encoding="utf-8") as fh:
        fh.write(
            f"path: {os.path.abspath(data_root)}\n"
            f"train: images/{train_key}\n"
            f"val: images/{val_key}\n"
        )
        if "test" in splits:
            fh.write("test: images/test\n")
        fh.write("\nnames:\n  0: banano\n")
    print(f"\ndata.yaml: {yaml_path}")


if __name__ == "__main__":
    main()
