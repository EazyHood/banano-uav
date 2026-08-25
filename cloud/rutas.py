"""Hace que un data.yaml con `path:` relativo lo entienda ultralytics.

El problema, medido el 2026-08-24. `cloud/make_splits.py` escribe los YAML con una raíz
RELATIVA a propósito, para que se puedan versionar y funcionen en cualquier máquina:

    path: ../realdata

Pero ultralytics no resuelve ese `path` contra el fichero YAML, sino contra su propio
`datasets_dir` (el de `~/AppData/Roaming/Ultralytics/settings.json`). Resultado real al
intentar medir desde la raíz del repo:

    Dataset 'splits/lofo_armah.yaml' images not found,
    missing path 'C:\\Users\\jhona\\realdata\\newfarms\\armah\\train\\images'

...que es la raíz del repo subida un nivel de más. En Kaggle no se notaba porque el
notebook llama a make_splits con `--raiz-declarada` absoluta, así que el fallo sólo salía
al usar los YAML versionados, que es justo lo que hace cualquiera que clone el repo.

Aquí se resuelve el `path` contra el propio YAML y se escribe una copia temporal con la
ruta absoluta. No se toca el fichero versionado: sigue siendo portable.
"""

from __future__ import annotations

import tempfile
from pathlib import Path


def resuelve(data: Path) -> Path:
    """Devuelve un data.yaml que ultralytics pueda abrir desde donde sea.

    Si el `path` ya es absoluto y existe, devuelve el mismo fichero. Si es relativo, lo
    resuelve contra el directorio del YAML y escribe una copia temporal.
    """
    import yaml

    data = Path(data)
    cfg = yaml.safe_load(data.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or "path" not in cfg:
        return data

    raiz = Path(str(cfg["path"]))
    if raiz.is_absolute():
        return data

    absoluta = (data.parent / raiz).resolve()
    cfg["path"] = absoluta.as_posix()

    tmp = Path(tempfile.gettempdir()) / f"banano_{data.stem}_resuelto.yaml"
    tmp.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return tmp
