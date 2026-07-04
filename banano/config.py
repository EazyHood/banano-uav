"""Configuracion centralizada del pipeline de deteccion.

Un unico objeto ``PipelineConfig`` reune TODOS los parametros ajustables, con
valores por defecto sensatos, validacion y carga desde YAML/dict. Esto permite a
una empresa versionar y auditar exactamente con que parametros se hizo cada
conteo (reproducibilidad), y ajustar sin tocar el codigo.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields

from .errors import ConfigError

# Supuestos agronomicos por defecto (banano comercial adulto).
DEFAULT_ASSUMED_SPACING_M = 2.5  # separacion tipica entre plantas
DEFAULT_LEAF_LEN_M = 0.9  # largo de hoja
DEFAULT_PSEUDOSTEM_SEP_M = 0.5  # separacion entre pseudotallos dentro de una macolla
DEFAULT_GRID_MIN_STRENGTH = 0.05  # por debajo, el marco estimado no es fiable


@dataclass
class PipelineConfig:
    """Parametros del pipeline de deteccion clasico y del procesado de ortomosaico."""

    # --- escala fisica ---
    gsd_cm: float | None = None  # resolucion cm/pixel (auto en GeoTIFF)
    assumed_spacing_m: float = DEFAULT_ASSUMED_SPACING_M
    leaf_len_m: float = DEFAULT_LEAF_LEN_M
    pseudostem_sep_m: float = DEFAULT_PSEUDOSTEM_SEP_M

    # --- deteccion de centros ---
    mode: str = "both"  # 'bright' | 'dark' | 'both'
    rel_threshold: float = 0.30  # umbral relativo de picos (0-1)
    use_mask: bool = True  # restringir al dosel segmentado
    use_grid: bool = True  # usar el marco de siembra estimado
    grid_min_strength: float = DEFAULT_GRID_MIN_STRENGTH

    # --- procesado de ortomosaico ---
    tile: int = 1024  # tamano de tile (px)
    overlap: int = 128  # solape entre tiles (px)

    # --- modelo de deep learning (opcional) ---
    model_weights: str | None = None  # ruta a pesos YOLOv8-seg; None = clasico
    model_conf: float = 0.65  # umbral de confianza del modelo (calibrado para conteo)

    VALID_MODES = ("bright", "dark", "both")

    def _coerce_numeric(self, errs):
        """Coacciona campos numericos (YAML entre comillas -> str) y rechaza tipos raros.

        Rechaza bool (subclase de int) para que True/False no se cuelen como 1/0.
        """

        def cast(name, caster, allow_none=False):
            v = getattr(self, name)
            if v is None:
                if not allow_none:
                    errs.append(f"{name} no puede ser None")
                return
            if isinstance(v, bool) or not isinstance(v, (int, float, str)):
                errs.append(f"{name} debe ser numerico, no {type(v).__name__} ({v!r})")
                return
            try:
                setattr(self, name, caster(v))
            except (TypeError, ValueError):
                errs.append(f"{name} debe ser {caster.__name__}, no {v!r}")

        cast("gsd_cm", float, allow_none=True)
        for name in (
            "assumed_spacing_m",
            "leaf_len_m",
            "pseudostem_sep_m",
            "rel_threshold",
            "grid_min_strength",
            "model_conf",
        ):
            cast(name, float)
        for name in ("tile", "overlap"):
            cast(name, int)

    def validate(self) -> PipelineConfig:
        """Valida tipos, rangos y coherencia. Lanza ConfigError si algo esta mal."""
        errs: list[str] = []

        # 1. tipos: coacciona numericos y valida strings/None antes de comparar
        self._coerce_numeric(errs)
        if self.model_weights is not None and not isinstance(self.model_weights, str):
            errs.append(
                f"model_weights debe ser una ruta (str) o None, no {type(self.model_weights).__name__}"
            )
        if not isinstance(self.mode, str):
            errs.append(f"mode debe ser str, no {type(self.mode).__name__}")
        if errs:  # cortar antes de comparar rangos sobre valores de tipo invalido
            raise ConfigError("Configuracion invalida:\n  - " + "\n  - ".join(errs))

        # 2. rangos y coherencia
        if self.gsd_cm is not None and self.gsd_cm <= 0:
            errs.append(f"gsd_cm debe ser > 0 (o None), no {self.gsd_cm}")
        if self.mode not in self.VALID_MODES:
            errs.append(f"mode debe ser uno de {self.VALID_MODES}, no {self.mode!r}")
        if not (0.0 <= self.rel_threshold <= 1.0):
            errs.append(f"rel_threshold debe estar en [0,1], no {self.rel_threshold}")
        if not (0.0 <= self.grid_min_strength <= 1.0):
            errs.append(f"grid_min_strength debe estar en [0,1], no {self.grid_min_strength}")
        if self.tile < 32:
            errs.append(f"tile debe ser >= 32, no {self.tile}")
        if self.overlap < 0:
            errs.append(f"overlap debe ser >= 0, no {self.overlap}")
        for name in ("assumed_spacing_m", "leaf_len_m", "pseudostem_sep_m"):
            v = getattr(self, name)
            if v <= 0:
                errs.append(f"{name} debe ser > 0, no {v}")
        if not (0.0 <= self.model_conf <= 1.0):
            errs.append(f"model_conf debe estar en [0,1], no {self.model_conf}")
        if errs:
            raise ConfigError("Configuracion invalida:\n  - " + "\n  - ".join(errs))
        # saneo suave: overlap < tile
        if self.overlap >= self.tile:
            self.overlap = self.tile // 2
        return self

    @classmethod
    def from_dict(cls, data: dict) -> PipelineConfig:
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ConfigError(f"Claves de configuracion desconocidas: {sorted(unknown)}")
        return cls(**{k: v for k, v in data.items() if k in known}).validate()

    @classmethod
    def from_yaml(cls, path: str) -> PipelineConfig:
        try:
            import yaml
        except ImportError as e:  # pragma: no cover
            raise ConfigError(
                "Para leer configuracion YAML instala pyyaml: pip install pyyaml"
            ) from e
        try:
            with open(path, encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        except OSError as e:
            raise ConfigError(f"No se pudo leer la configuracion {path!r}: {e}") from e
        if not isinstance(data, dict):
            raise ConfigError(f"El YAML de configuracion debe ser un mapeo, no {type(data)}")
        return cls.from_dict(data)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
