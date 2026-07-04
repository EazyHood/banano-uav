"""Jerarquia de excepciones de banano-drone.

Todas las excepciones esperables heredan de ``BananoError`` para que las
integraciones (CLI, API, servicios) puedan capturarlas de forma uniforme y
distinguirlas de bugs internos inesperados.
"""

from __future__ import annotations


class BananoError(Exception):
    """Base de todos los errores controlados de banano-drone."""


class InputError(BananoError):
    """Entrada invalida del usuario (archivo faltante, formato no soportado, etc.)."""


class RasterError(BananoError):
    """Fallo al abrir o leer el raster/ortomosaico."""


class ConfigError(BananoError):
    """Configuracion invalida (parametro fuera de rango o incoherente)."""


class DependencyError(BananoError):
    """Falta una dependencia opcional necesaria para la operacion solicitada."""


class ModelError(BananoError):
    """Fallo relacionado con el modelo de deep learning (pesos, inferencia)."""
