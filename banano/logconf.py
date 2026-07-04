"""Configuracion de logging para la CLI y las integraciones."""

from __future__ import annotations

import logging
import sys


def setup_logging(verbosity: int = 0, quiet: bool = False):
    """Configura el logging raiz del paquete.

    verbosity : 0 = INFO, 1 = DEBUG (-v), >=2 = DEBUG con libs.
    quiet : solo WARNING y superiores.
    """
    if quiet:
        level = logging.WARNING
    elif verbosity >= 1:
        level = logging.DEBUG
    else:
        level = logging.INFO

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))

    root = logging.getLogger("banano")
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False
    return root
