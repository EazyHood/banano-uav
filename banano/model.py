"""Integracion del modelo de deep learning (YOLOv8-seg) para deteccion de macollas.

El modelo detecta MACOLLAS directamente (segmentacion de instancias). Es el camino
de maxima precision cuando hay pesos entrenados. Se carga de forma perezosa y
requiere el extra ``[deep]`` (ultralytics). Si no esta instalado, se lanza
``DependencyError`` con instrucciones claras.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

from .errors import DependencyError, ModelError

logger = logging.getLogger(__name__)


def _require_ultralytics():
    try:
        from ultralytics import YOLO  # noqa: F401
    except ImportError as e:
        raise DependencyError(
            "El modelo de deep learning requiere 'ultralytics'. "
            "Instala:  pip install banano-drone[deep]"
        ) from e
    return YOLO


@lru_cache(maxsize=4)
def load_model(weights: str):
    """Carga (y cachea) un modelo YOLOv8-seg desde un archivo de pesos."""
    YOLO = _require_ultralytics()
    try:
        model = YOLO(weights)
    except Exception as e:  # pesos corruptos/incompatibles
        raise ModelError(f"No se pudo cargar el modelo {weights!r}: {e}") from e
    logger.info("Modelo cargado: %s (%s clases)", weights, len(getattr(model, "names", {})))
    return model


class BananaModel:
    """Envoltura fina sobre un modelo YOLOv8-seg para detectar macollas."""

    def __init__(self, weights: str, conf: float = 0.25, imgsz: int = 640, augment: bool = False):
        self.weights = weights
        self.conf = conf
        self.imgsz = imgsz
        self.augment = augment  # test-time augmentation (mas preciso, mas lento)
        self.model = load_model(weights)

    def predict_mats(self, image: np.ndarray, conf: float | None = None):
        """Detecta macollas en una imagen RGB (HxWx3 uint8).

        Devuelve dict con:
          centroids : (N, 2) en (fila, columna)
          confidences : (N,)
          n : numero de macollas detectadas
        """
        c = self.conf if conf is None else conf
        try:
            results = self.model.predict(
                image, conf=c, imgsz=self.imgsz, augment=self.augment, verbose=False
            )
        except Exception as e:
            raise ModelError(f"Fallo la inferencia del modelo: {e}") from e

        centroids = []
        confs = []
        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is None or boxes.xywh is None:
                continue
            xywh = boxes.xywh.cpu().numpy()  # (n, 4): cx, cy, w, h en px
            cfs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xywh))
            for (cx, cy, _w, _h), cf in zip(xywh, cfs):
                centroids.append((float(cy), float(cx)))  # (fila, columna)
                confs.append(float(cf))

        arr = np.array(centroids, dtype=float) if centroids else np.empty((0, 2))
        return {
            "centroids": arr,
            "confidences": np.array(confs, dtype=float),
            "n": int(arr.shape[0]),
        }
