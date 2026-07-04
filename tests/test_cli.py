"""Pruebas de la CLI: codigos de salida, validacion y ejecucion completa."""
from __future__ import annotations

import os
import sys

import imageio.v2 as imageio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from banano.cli import main
from banano.synth import synth_plantation


def _make_png(tmp_path, size=500, seed=4):
    img, _, _, _ = synth_plantation(H=size, W=size, gsd_cm=3.0, seed=seed)
    p = os.path.join(str(tmp_path), "orto.png")
    imageio.imwrite(p, img)
    return p


def test_cli_missing_file_exit1(tmp_path):
    code = main(["--input", os.path.join(str(tmp_path), "no.tif"), "--out", str(tmp_path)])
    assert code == 1


def test_cli_invalid_threshold_exit1(tmp_path):
    p = _make_png(tmp_path)
    code = main(["--input", p, "--threshold", "5", "--out", str(tmp_path)])
    assert code == 1


def test_cli_full_run_exit0(tmp_path):
    p = _make_png(tmp_path)
    out = os.path.join(str(tmp_path), "res")
    code = main(["--input", p, "--gsd", "3.0", "--out", out, "--quiet"])
    assert code == 0
    for f in ("resumen.json", "plantas.csv", "plantas.geojson", "informe.html"):
        assert os.path.exists(os.path.join(out, f)), f


def test_cli_config_file(tmp_path):
    p = _make_png(tmp_path)
    cfg = os.path.join(str(tmp_path), "c.yaml")
    with open(cfg, "w", encoding="utf-8") as fh:
        fh.write("gsd_cm: 3.0\nmode: both\ntile: 400\noverlap: 80\n")
    out = os.path.join(str(tmp_path), "res2")
    code = main(["--input", p, "--config", cfg, "--out", out, "--quiet"])
    assert code == 0


def test_cli_bad_config_exit1(tmp_path):
    p = _make_png(tmp_path)
    cfg = os.path.join(str(tmp_path), "bad.yaml")
    with open(cfg, "w", encoding="utf-8") as fh:
        fh.write("mode: morado\n")
    code = main(["--input", p, "--config", cfg, "--out", str(tmp_path)])
    assert code == 1
