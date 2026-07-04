"""Pruebas de PipelineConfig: validacion, carga y coherencia."""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from banano.config import PipelineConfig
from banano.errors import ConfigError


def test_defaults_valid():
    cfg = PipelineConfig().validate()
    assert cfg.mode == "both"
    assert cfg.tile >= 32


def test_invalid_mode():
    with pytest.raises(ConfigError):
        PipelineConfig(mode="verde").validate()


@pytest.mark.parametrize("field,value", [
    ("rel_threshold", 5.0),
    ("rel_threshold", -0.1),
    ("gsd_cm", 0.0),
    ("gsd_cm", -2.0),
    ("tile", 10),
    ("assumed_spacing_m", 0.0),
    ("model_conf", 2.0),
])
def test_invalid_ranges(field, value):
    with pytest.raises(ConfigError):
        PipelineConfig(**{field: value}).validate()


def test_overlap_ge_tile_sanitized():
    cfg = PipelineConfig(tile=100, overlap=200).validate()
    assert cfg.overlap < cfg.tile


def test_from_dict_unknown_key():
    with pytest.raises(ConfigError):
        PipelineConfig.from_dict({"foo": 1})


def test_from_dict_roundtrip():
    cfg = PipelineConfig(gsd_cm=3.0, mode="bright", tile=800)
    d = cfg.to_dict()
    cfg2 = PipelineConfig.from_dict(d)
    assert cfg2.gsd_cm == 3.0 and cfg2.mode == "bright" and cfg2.tile == 800


def test_from_yaml(tmp_path):
    p = os.path.join(str(tmp_path), "c.yaml")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("mode: dark\nrel_threshold: 0.2\ntile: 512\n")
    cfg = PipelineConfig.from_yaml(p)
    assert cfg.mode == "dark" and cfg.rel_threshold == 0.2 and cfg.tile == 512


def test_from_yaml_not_mapping(tmp_path):
    p = os.path.join(str(tmp_path), "bad.yaml")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("- 1\n- 2\n")
    with pytest.raises(ConfigError):
        PipelineConfig.from_yaml(p)


def test_string_numbers_coerced():
    # YAML puede traer numeros entre comillas -> deben coaccionarse, no crashear.
    cfg = PipelineConfig.from_dict({"tile": "1024", "overlap": "128", "gsd_cm": "3.0"})
    assert cfg.tile == 1024 and isinstance(cfg.tile, int)
    assert cfg.gsd_cm == 3.0 and isinstance(cfg.gsd_cm, float)


@pytest.mark.parametrize("field,value", [
    ("gsd_cm", True),          # bool no debe colarse como numero
    ("tile", True),
    ("rel_threshold", "abc"),  # string no numerico
    ("tile", "muchos"),
    ("model_weights", 123),    # debe ser str o None
    ("mode", 5),               # debe ser str
])
def test_bad_types_raise_configerror(field, value):
    with pytest.raises(ConfigError):
        PipelineConfig.from_dict({field: value})
