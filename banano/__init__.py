"""banano — identificacion de cultivo de banano en imagenes de dron.

El banano crece en macollas (planta madre + hijuelos, tipicamente en grupos de
~3 pseudotallos), a diferencia de palma/aguacate/mango que salen como individuos
aislados. Eso rompe los detectores de copas clasicos. Este paquete resuelve el
problema en dos niveles:

  1. Identifica el DOSEL de banano (que pixeles son banano y no suelo/maleza).
  2. Detecta CENTROS de pseudotallo por simetria radial (roseta de hojas) y los
     agrupa en MACOLLAS, la unidad agronomica correcta.

Uso rapido:
    from banano import detect_banana
    res = detect_banana(rgb, gsd_cm=3.0)
    print(res.n_mats, res.n_pseudostems)
"""

from .config import PipelineConfig
from .errors import (
    BananoError,
    ConfigError,
    DependencyError,
    InputError,
    ModelError,
    RasterError,
)
from .geo import Raster
from .ortho import OrthoResult, process_orthomosaic
from .pipeline import DetectionResult, detect_banana

__all__ = [
    "detect_banana",
    "DetectionResult",
    "process_orthomosaic",
    "OrthoResult",
    "Raster",
    "PipelineConfig",
    "BananoError",
    "InputError",
    "ConfigError",
    "RasterError",
    "DependencyError",
    "ModelError",
]
__version__ = "2.1.0"
