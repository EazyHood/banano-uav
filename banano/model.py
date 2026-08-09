"""Integracion del modelo de deep learning (YOLO det/seg) para deteccion de macollas.

El modelo detecta MACOLLAS directamente (deteccion o segmentacion de instancias:
cualquier familia YOLO de ultralytics — v8, 11, 26). Es el camino de maxima
precision cuando hay pesos entrenados. Se carga de forma perezosa y requiere el
extra ``[deep]`` (ultralytics). Si no esta instalado, se lanza ``DependencyError``
con instrucciones claras.
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
    """Carga (y cachea) un modelo YOLO (det/seg) desde un archivo de pesos."""
    YOLO = _require_ultralytics()
    try:
        model = YOLO(weights)
    except Exception as e:  # pesos corruptos/incompatibles
        raise ModelError(f"No se pudo cargar el modelo {weights!r}: {e}") from e
    logger.info("Modelo cargado: %s (%s clases)", weights, len(getattr(model, "names", {})))
    return model


class BananaModel:
    """Envoltura fina sobre un modelo YOLO (det/seg) para detectar macollas."""

    def __init__(
        self,
        weights: str,
        conf: float = 0.25,
        imgsz: int = 640,
        augment: bool = False,
        device: str | None = None,
    ):
        self.weights = weights
        self.conf = conf
        self.imgsz = imgsz
        self.augment = augment  # test-time augmentation (mas preciso, mas lento)
        self.device = device  # None = auto; 'cpu' fuerza CPU (maquinas sin GPU)
        self.model = load_model(weights)

    def predict_mats(self, image: np.ndarray, conf: float | None = None):
        """Detecta macollas en una imagen RGB (HxWx3 uint8).

        Devuelve dict con:
          centroids : (N, 2) en (fila, columna)
          confidences : (N,)
          boxes_xyxy : (N, 4) en px (x0, y0, x1, y1) — para fusion/filtrado por area
          n : numero de macollas detectadas
        """
        c = self.conf if conf is None else conf
        kwargs = {} if self.device is None else {"device": self.device}
        try:
            results = self.model.predict(
                image, conf=c, imgsz=self.imgsz, augment=self.augment, verbose=False, **kwargs
            )
        except Exception as e:
            raise ModelError(f"Fallo la inferencia del modelo: {e}") from e

        centroids = []
        confs = []
        xyxys = []
        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is None or boxes.xywh is None:
                continue
            xywh = boxes.xywh.cpu().numpy()  # (n, 4): cx, cy, w, h en px
            xyxy = boxes.xyxy.cpu().numpy()  # (n, 4): x0, y0, x1, y1 en px
            cfs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xywh))
            for (cx, cy, _w, _h), bb, cf in zip(xywh, xyxy, cfs):
                centroids.append((float(cy), float(cx)))  # (fila, columna)
                xyxys.append([float(v) for v in bb])
                confs.append(float(cf))

        arr = np.array(centroids, dtype=float) if centroids else np.empty((0, 2))
        return {
            "centroids": arr,
            "confidences": np.array(confs, dtype=float),
            "boxes_xyxy": np.array(xyxys, dtype=float) if xyxys else np.empty((0, 4)),
            "n": int(arr.shape[0]),
        }
