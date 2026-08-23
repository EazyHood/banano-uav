"""Pruebas de las herramientas de auditoria y de preparacion para la nube.

Cubren los tres defectos reales que estas herramientas encontraron el 2026-08-23:
  - dos datasets que son el mismo material y fugaban el 100% de un split de test,
  - 35x de diferencia de escala entre fincas, invisible hasta que se midio,
  - splits atados a rutas C:/Users/... que no resuelven fuera de este PC.

Todo corre sobre datasets sinteticos minusculos: sin GPU, sin red y en segundos.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

from deep.leak_audit import compara, huellas, matriz  # noqa: E402
from deep.scale_audit import audita_fuente  # noqa: E402


def _dataset(base, nombre, split, n_imgs, lado_caja, semilla=0, tam=(200, 200)):
    """Crea .../nombre/split/{images,labels} con cajas de tamano relativo conocido."""
    rng = np.random.default_rng(semilla)
    d_img = base / nombre / split / "images"
    d_lab = base / nombre / split / "labels"
    d_img.mkdir(parents=True, exist_ok=True)
    d_lab.mkdir(parents=True, exist_ok=True)
    # `nombre` puede venir con barra ("newfarms/elliot"): el fichero lleva solo la hoja
    hoja = nombre.replace("\\", "/").split("/")[-1]
    for i in range(n_imgs):
        arr = rng.integers(0, 255, (tam[1], tam[0], 3), dtype=np.uint8)
        Image.fromarray(arr).save(d_img / f"{hoja}_{i}.jpg", quality=95)
        (d_lab / f"{hoja}_{i}.txt").write_text(f"0 0.5 0.5 {lado_caja} {lado_caja}\n")
    return base / nombre


def _copia_exacta(origen_dir, destino_dir):
    """Misma imagen, OTRO nombre: es lo que hace Roboflow al reexportar."""
    destino_dir.joinpath("images").mkdir(parents=True, exist_ok=True)
    destino_dir.joinpath("labels").mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(sorted((origen_dir / "images").glob("*.jpg"))):
        (destino_dir / "images" / f"otro_nombre_{i}.jpg").write_bytes(p.read_bytes())
        (destino_dir / "labels" / f"otro_nombre_{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n")


# --------------------------------------------------------------------------- fugas


def test_detecta_dataset_duplicado_aunque_cambie_el_nombre(tmp_path):
    # El caso real: prueba2rgb y etiquetasnuevas, 505 imagenes identicas con nombres
    # distintos. El dedup por nombre no las veia; por contenido si.
    a = _dataset(tmp_path, "original", "train", 5, 0.2)
    _copia_exacta(a / "train", tmp_path / "renombrado" / "train")

    pares = matriz(tmp_path, {})
    assert len(pares) == 1, pares
    assert pares[0]["comunes"] == 5
    assert pares[0]["pct_a"] == 100.0 and pares[0]["pct_b"] == 100.0


def test_fuga_train_test_se_mide_y_se_reporta(tmp_path):
    a = _dataset(tmp_path, "finca_a", "train", 6, 0.2)
    _copia_exacta(a / "train", tmp_path / "finca_b" / "test")
    _dataset(tmp_path, "finca_c", "test", 4, 0.2, semilla=9)

    r = compara(
        [tmp_path / "finca_a" / "train"],
        [tmp_path / "finca_b" / "test", tmp_path / "finca_c" / "test"],
        {},
    )
    assert r["imgs_validacion"] == 10
    assert r["imgs_fugadas"] == 6
    assert r["pct_fugado"] == 60.0
    por_fuente = {os.path.basename(os.path.dirname(d["fuente"])): d for d in r["por_fuente"]}
    assert por_fuente["finca_b"]["pct"] == 100.0
    assert por_fuente["finca_c"]["pct"] == 0.0


def test_sin_solape_no_inventa_fuga(tmp_path):
    _dataset(tmp_path, "finca_a", "train", 4, 0.2, semilla=1)
    _dataset(tmp_path, "finca_b", "test", 4, 0.2, semilla=2)
    r = compara([tmp_path / "finca_a" / "train"], [tmp_path / "finca_b" / "test"], {})
    assert r["imgs_fugadas"] == 0
    assert matriz(tmp_path, {}) == []


def test_huellas_no_cuenta_dos_veces_la_misma_imagen(tmp_path):
    # En Windows *.jpg y *.JPG devuelven los MISMOS ficheros: sin deduplicar por ruta
    # normalizada, todos los totales salian exactamente al doble.
    d = _dataset(tmp_path, "finca", "train", 3, 0.2)
    assert len(huellas(d, {})) == 3


# -------------------------------------------------------------------------- escala


def test_mide_el_tamano_del_objeto_en_pixeles(tmp_path):
    # Caja del 20% sobre imagen de 200 px = 40 px reales; a imgsz 400 son 80 px.
    d = _dataset(tmp_path, "finca", "train", 4, 0.2, tam=(200, 200))
    import random

    r = audita_fuente(d, 400, random.Random(0))
    assert r is not None
    assert abs(r["planta_px_real"] - 40) < 1
    assert abs(r["planta_px_imgsz"] - 80) < 1
    assert r["imgs"] == 4 and r["clases"] == ["0"]


def test_separa_fincas_de_escalas_distintas(tmp_path):
    # Es el hallazgo que explica el fallo en finca nueva: la misma imagen con plantas
    # 10 veces mas pequenas es, para el detector, otro problema.
    grande = _dataset(tmp_path, "grande", "train", 3, 0.4, tam=(200, 200))
    pequena = _dataset(tmp_path, "pequena", "train", 3, 0.04, tam=(200, 200), semilla=3)
    import random

    rg = audita_fuente(grande, 768, random.Random(0))
    rp = audita_fuente(pequena, 768, random.Random(0))
    assert rg is not None and rp is not None
    assert rg["planta_px_imgsz"] / rp["planta_px_imgsz"] > 9


def test_marca_lo_que_es_demasiado_pequeno_para_detectarse(tmp_path):
    # Por debajo de ~8 px un objeto no tiene ni una celda propia en el mapa mas fino.
    d = _dataset(tmp_path, "diminuta", "train", 3, 0.01, tam=(200, 200))
    import random

    r = audita_fuente(d, 320, random.Random(0))
    assert r is not None
    assert r["frac_indetectable"] == 1.0


# -------------------------------------------------------------------------- splits


def _make_splits(tmp_path, *args):
    return subprocess.run(
        [sys.executable, os.path.join(RAIZ, "cloud", "make_splits.py"), *args],
        capture_output=True,
        text=True,
        cwd=RAIZ,
    )


def test_splits_no_llevan_rutas_absolutas_de_windows(tmp_path):
    raiz = tmp_path / "realdata"
    for finca, carpeta in [("original", "count_banana_plants"), ("elliot", "newfarms/elliot")]:
        for split in ("train", "valid", "test"):
            _dataset(raiz, carpeta, split, 2, 0.2, semilla=hash(finca + split) % 1000)

    salida = tmp_path / "splits"
    p = _make_splits(tmp_path, "--raiz", str(raiz), "--salida", str(salida))
    assert p.returncode == 0, p.stderr

    for y in salida.glob("*.yaml"):
        texto = y.read_text(encoding="utf-8")
        assert "C:/Users" not in texto and "C:\\Users" not in texto, y.name


def test_lofo_deja_la_finca_retenida_entera_fuera_del_train(tmp_path):
    raiz = tmp_path / "realdata"
    for carpeta in ("count_banana_plants", "newfarms/elliot", "newfarms/armah"):
        for split in ("train", "valid", "test"):
            _dataset(raiz, carpeta, split, 2, 0.2, semilla=abs(hash(carpeta + split)) % 1000)

    salida = tmp_path / "splits"
    p = _make_splits(tmp_path, "--raiz", str(raiz), "--salida", str(salida))
    assert p.returncode == 0, p.stderr

    import yaml

    cfg = yaml.safe_load((salida / "lofo_armah.yaml").read_text(encoding="utf-8"))
    assert all("armah" not in t for t in cfg["train"]), cfg["train"]
    # y la finca retenida entra con sus TRES splits, no solo con test
    assert len(cfg["val"]) == 3 and all("armah" in v for v in cfg["val"])

    datos = json.loads((salida / "splits.json").read_text(encoding="utf-8"))
    assert {g["retenida"] for g in datos["generados"] if g["retenida"]} == {"original", "elliot", "armah"}


def test_prueba2rgb_no_se_incluye_junto_a_etiquetasnuevas():
    # Regresion del defecto real: son el mismo dataset. Si alguien vuelve a meter las
    # dos, el entrenamiento pesa doble ese material y el test queda fugado.
    from cloud.make_splits import FINCAS

    todas = [c for f in FINCAS.values() for c in f["carpetas"]]
    assert "extra/etiquetasnuevas" in todas
    assert "extra/prueba2rgb" not in todas
    # y lo mismo con las dos versiones del proyecto platano-lasuiza
    assert "newfarms/lasuiza" in todas
    assert "extra/platano-lasuiza" not in todas
