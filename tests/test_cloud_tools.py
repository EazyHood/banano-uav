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
import pathlib
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


# ---------------------------------------------------------------------- dedup Roboflow


def _roboflow(base, dataset, split, original, hashes):
    """Simula la exportacion de Roboflow: <original>_jpg.rf.<hash>.jpg, N copias."""
    d_img = base / dataset / split / "images"
    d_lab = base / dataset / split / "labels"
    d_img.mkdir(parents=True, exist_ok=True)
    d_lab.mkdir(parents=True, exist_ok=True)
    for h in hashes:
        n = f"{original}_jpg.rf.{h}.jpg"
        Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(d_img / n)
        (d_lab / f"{original}_jpg.rf.{h}.txt").write_text("0 0.5 0.5 0.2 0.2\n")


def test_dedup_deja_una_copia_por_disparo_original(tmp_path):
    # Es lo que hace que el holdout de la nube sea el mismo que se midio en casa: armah
    # se descarga con 148 imagenes y tiene que quedarse en 62.
    from cloud.fetch_data import deduplica

    _roboflow(tmp_path, "finca", "train", "foto_a", ["aa11", "bb22", "cc33"])
    _roboflow(tmp_path, "finca", "train", "foto_b", ["dd44"])
    augmentadas, cruzadas = deduplica(tmp_path / "finca")

    assert (augmentadas, cruzadas) == (2, 0)
    quedan = sorted(p.name for p in (tmp_path / "finca" / "train" / "images").iterdir())
    assert quedan == ["foto_a_jpg.rf.aa11.jpg", "foto_b_jpg.rf.dd44.jpg"]
    # la etiqueta se va con su imagen, no se queda huerfana
    assert sorted(p.stem for p in (tmp_path / "finca" / "train" / "labels").iterdir()) == [
        "foto_a_jpg.rf.aa11",
        "foto_b_jpg.rf.dd44",
    ]


def test_dedup_borra_del_test_lo_que_esta_en_train(tmp_path):
    # Roboflow reparte a veces el MISMO disparo entre train y test: eso es una fuga dentro
    # del propio dataset. Gana train, y el test se queda limpio (7 casos reales en elliot).
    from cloud.fetch_data import deduplica

    _roboflow(tmp_path, "finca", "train", "foto_x", ["ff99"])
    _roboflow(tmp_path, "finca", "test", "foto_x", ["aa00"])
    _roboflow(tmp_path, "finca", "valid", "foto_x", ["bb11"])
    augmentadas, cruzadas = deduplica(tmp_path / "finca")

    assert (augmentadas, cruzadas) == (0, 2)
    assert [p.name for p in (tmp_path / "finca" / "train" / "images").iterdir()] == [
        "foto_x_jpg.rf.ff99.jpg"
    ]
    assert list((tmp_path / "finca" / "test" / "images").iterdir()) == []
    assert list((tmp_path / "finca" / "valid" / "images").iterdir()) == []


def test_el_manifiesto_dice_que_fuentes_hay_que_deduplicar():
    # No es uniforme y no puede serlo: count_banana_plants conserva a proposito las dos
    # copias augmentadas de cada disparo (502 = 251x2 en su train), mientras que las de
    # newfarms/ se limpiaron. Si se aplica el mismo criterio a todas, el dataset de la
    # nube deja de ser el que se midio.
    import json

    m = json.loads((pathlib.Path(RAIZ) / "cloud" / "data_manifest.json").read_text(encoding="utf-8"))
    por_carpeta = {f["carpeta"]: f for f in m["fuentes"]}

    assert por_carpeta["newfarms/armah"]["dedup_aplicado"] is True
    assert por_carpeta["newfarms/elliot"]["dedup_aplicado"] is True
    assert por_carpeta["count_banana_plants"]["dedup_aplicado"] is False

    # Y la diversidad real: mas imagenes no son mas fotos.
    p80 = por_carpeta["extra/plantas_jovenes_80m1"]
    assert p80["imgs_en_disco"] > 5 * p80["disparos_originales"] * 0.9


def test_solo_se_deduplican_las_fuentes_que_lo_estaban(tmp_path):
    # count_banana_plants conserva sus copias augmentadas a proposito; armah no. Si se
    # aplica el mismo criterio a las dos, el dataset reconstruido en la nube deja de
    # coincidir con el que produjo las cifras publicadas.
    from cloud.fetch_data import debe_deduplicar

    limpia = {"carpeta": "newfarms/armah", "dedup_aplicado": True}
    augmentada = {"carpeta": "count_banana_plants", "dedup_aplicado": False}

    assert debe_deduplicar(limpia, True) is True
    assert debe_deduplicar(augmentada, True) is False
    # --sin-dedup lo apaga todo
    assert debe_deduplicar(limpia, False) is False
    # una fuente sin el campo (manifiesto viejo) se deduplica: es lo conservador
    assert debe_deduplicar({"carpeta": "x"}, True) is True


def test_la_verdad_de_terreno_acepta_cajas_y_poligonos(tmp_path):
    # realdata/newfarms/lasuiza trae POLIGONOS, no cajas: 61 de 61 lineas de su test.
    # Leerlos con p[1:5] tomaba los dos primeros vertices como centro y tamano, sin dar
    # error: cajas disparatadas en silencio.
    from deep.eval_ensemble import _gt_cajas

    d = tmp_path / "split"
    (d / "images").mkdir(parents=True)
    (d / "labels").mkdir(parents=True)
    img = d / "images" / "a.jpg"
    img.touch()

    # caja: centro (0.5, 0.5), lado 0.2  ->  xyxy 0.4-0.6 sobre 100 px = 40..60
    (d / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    caja = _gt_cajas(img, 100, 100)
    assert caja.shape == (1, 4)
    assert list(caja[0]) == [40.0, 40.0, 60.0, 60.0]

    # el MISMO cuadrado escrito como poligono tiene que dar la MISMA caja
    (d / "labels" / "a.txt").write_text("0 0.4 0.4 0.6 0.4 0.6 0.6 0.4 0.6\n")
    poli = _gt_cajas(img, 100, 100)
    assert poli.shape == (1, 4)
    assert np.allclose(poli[0], caja[0])

    # y leerlo como si fuera una caja habria dado algo muy distinto: con p[1:5] el
    # centro seria (0.4, 0.4) y el tamano (0.6, 0.4) -> x de 10 a 70, no de 40 a 60
    assert not np.allclose(poli[0], [10.0, 20.0, 70.0, 60.0])


# ------------------------------------------------------------------ notebook Kaggle


def _notebook():
    import ast
    import json as _json

    nb = _json.loads(
        (pathlib.Path(RAIZ) / "kaggle" / "entrenar" / "banano-entrenar.ipynb").read_text(
            encoding="utf-8"
        )
    )
    celdas = []
    for c in nb["cells"]:
        src = "".join(c["source"])
        if c["cell_type"] == "code":
            ast.parse(src)  # que sea Python valido no es negociable: corre desatendido
        celdas.append(src)
    return celdas


def test_el_notebook_de_kaggle_es_python_valido_y_lleva_sus_guardas():
    # Corre desatendido en una maquina que no vemos: un fallo tonto ahi cuesta una
    # sesion entera. Las tres guardas salen de fallos reales, no de precaucion abstracta.
    celdas = _notebook()
    todo = "\n".join(celdas)

    # 1. P100: la imagen de Kaggle no trae kernels de Pascal, is_available() dice True
    #    y el entreno muere en el primer lote.
    assert "P100" in todo

    # 2. Kaggle ACEPTA enable_gpu/enable_internet y luego los DENIEGA si la cuenta no
    #    tiene el telefono verificado. Pasado por alto, el sintoma es un error de git
    #    que no menciona el motivo. Medido el 2026-08-24 en una corrida real.
    assert "Phone verification" in todo
    assert "gethostbyname" in todo

    # 3. Sin GPU hay que abortar, no seguir: 12 h de sesion tiradas.
    assert "raise RuntimeError" in celdas[1]

    # 4. NUNCA SystemExit en una celda: revienta el formateador de traceback de IPython
    #    ("TypeError: object of type 'NoneType' has no len()") y tapa el mensaje que
    #    explica que hacer. Medido en la corrida del 2026-08-24.
    assert "raise SystemExit" not in todo

    # 5. Un HTTP 400 al pedir el secreto significa "no existe", no un fallo de red: el
    #    cliente de Kaggle lo envuelve en "Connection error" y despista.
    assert "No hay clave de Roboflow" in todo

    # 6. La clave tiene DOS vias y basta con una: el dataset privado adjunto (la unica
    #    montable desde la terminal, porque el CLI no tiene comando de secrets) y el
    #    Secret de siempre. Si alguien quita la primera, esto vuelve a exigir un clic.
    assert "/kaggle/input/*/roboflow.json" in todo
    assert "UserSecretsClient" in todo

    # y el limite de tiempo se presupuesta por debajo de las 12 h de Kaggle
    assert "LIMITE_H = 11.0" in todo


def test_el_notebook_no_deja_los_datos_en_la_salida_autoguardada():
    # /kaggle/working se guarda entero al terminar y tiene 20 GB: las imagenes van al
    # scratch, no ahi. Si alguien cambia esto, la corrida muere al subir la salida.
    celdas = _notebook()
    todo = "\n".join(celdas)
    assert 'DATOS = "/kaggle/temp/realdata"' in todo


def test_el_manifiesto_de_fincas_nuevas_esta_medido_no_copiado():
    # La primera version de este fichero venia de un informe y traia tres errores que el
    # control --comprobar encontro: un dataset de "preentreno de palma" que resultaron ser
    # 1.021 imagenes con clases half/raw en vez de 24.985 con coronas de palma, una fuente
    # cuya unica version no se puede exportar, y los recuentos de imagenes tomados del
    # proyecto en vez de la version augmentada.
    import json as _json

    m = _json.loads(
        (pathlib.Path(RAIZ) / "cloud" / "nuevas_fincas.json").read_text(encoding="utf-8")
    )
    todas = m["fuentes"] + m["preentreno_analogo"]

    # lo que se elimino no puede volver por la puerta de atras
    carpetas = {f["carpeta"] for f in todas}
    assert "preentreno/palma" not in carpetas
    assert "nuevas/qpl6j" not in carpetas
    assert m["correcciones_del_control"], "las correcciones se documentan, no se borran"

    # y cada fuente que quede tiene que traer lo que hace falta para usarla sin sorpresas
    for f in todas:
        assert f["licencia"] == "CC BY 4.0", f["carpeta"]
        assert f["descarga_verificada_mb"] > 0, f["carpeta"]
        assert f["clases"], f["carpeta"]
        # si tiene mas de una clase, la nota tiene que avisar: hay que remapear
        if len(f["clases"]) > 1:
            assert "remapear" in f["notas"], f["carpeta"]


# --------------------------------------------------------------------- portabilidad


def test_ningun_script_codifica_el_interprete_de_windows():
    # deep/eval_v12_suite.py hacia `PY = ROOT / ".venv" / "Scripts" / "python.exe"`, que en
    # Linux no existe: la suite entera moria en la primera linea al correrla en la nube.
    # Lo correcto es sys.executable, que ademas usa el mismo entorno que lanzo el proceso.
    #
    # Se mira el AST y no el texto: en los docstrings de "Uso:" si vale poner el comando de
    # Windows como ejemplo para quien trabaje aqui. Lo que no puede haber es un literal asi
    # dentro del CODIGO, que es lo que rompe fuera de Windows.
    import ast

    raiz = pathlib.Path(RAIZ)
    malos = []
    for py in sorted(
        list(raiz.glob("deep/*.py")) + list(raiz.glob("cloud/*.py")) + list(raiz.glob("kaggle/*.py"))
    ):
        arbol = ast.parse(py.read_text(encoding="utf-8", errors="replace"))
        docs = set()
        for nodo in ast.walk(arbol):
            es_contenedor = isinstance(
                nodo, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            )
            if es_contenedor and ast.get_docstring(nodo, clean=False) is not None:
                docs.add(id(nodo.body[0].value))
        for nodo in ast.walk(arbol):
            if not isinstance(nodo, ast.Constant) or not isinstance(nodo.value, str):
                continue
            if id(nodo) in docs:
                continue
            v = nodo.value
            if "python.exe" in v or v == "Scripts":
                malos.append(f"{py.name}:{nodo.lineno}  ->  {v!r}")
    assert not malos, "codifican el interprete de Windows en CODIGO: " + "; ".join(malos)


def test_los_splits_que_se_versionan_no_llevan_rutas_de_una_maquina():
    # realdata/*.yaml lleva rutas absolutas de Windows, pero esa carpeta esta en .gitignore
    # y no viaja. Los que SI se publican son splits/, y esos tienen que resolver en cualquier
    # maquina o el entrenamiento en la nube no arranca.
    raiz = pathlib.Path(RAIZ) / "splits"
    if not raiz.exists():
        return
    for y in raiz.glob("*.yaml"):
        texto = y.read_text(encoding="utf-8")
        assert "C:/Users" not in texto and ("C:" + chr(92) + "Users") not in texto, y.name
        assert "/home/" not in texto, y.name


def test_el_resumen_del_barrido_destapa_la_media_arrastrada_por_una_finca():
    # Elegir la resolucion mirando una sola finca es afinar sobre el holdout — el error que
    # este repo ya corrigio una vez con el umbral de confianza. Pero un promedio a secas
    # tampoco basta: una finca disparada puede ganar la media siendo la peor opcion para
    # todas las demas. Por eso se reporta EN CUANTAS FINCAS gana cada resolucion.
    from cloud.scale_sweep import resumen_lofo

    fincas = {
        "lofo_a": {"barrido": [{"imgsz": 768, "mAP50": 0.20, "recall": 0.15},
                               {"imgsz": 1024, "mAP50": 0.30, "recall": 0.25}]},
        "lofo_b": {"barrido": [{"imgsz": 768, "mAP50": 0.22, "recall": 0.17},
                               {"imgsz": 1024, "mAP50": 0.31, "recall": 0.26}]},
        "lofo_c": {"barrido": [{"imgsz": 768, "mAP50": 0.95, "recall": 0.90},
                               {"imgsz": 1024, "mAP50": 0.10, "recall": 0.08}]},
    }
    r = {x["imgsz"]: x for x in resumen_lofo(fincas)}

    # 768 gana la media...
    assert r[768]["mAP50_medio"] > r[1024]["mAP50_medio"]
    # ...pero solo porque UNA finca esta disparada: 1024 es mejor en dos de las tres
    assert r[768]["gana_en_n_fincas"] == 1
    assert r[1024]["gana_en_n_fincas"] == 2
    # y el rango lo deja ver a simple vista
    assert r[768]["mAP50_min"] == 0.20 and r[768]["mAP50_max"] == 0.95
    assert r[1024]["mAP50_max"] - r[1024]["mAP50_min"] < 0.25


def test_multi_scale_es_una_fraccion_y_no_un_interruptor():
    # En ultralytics >= 8.4, multi_scale es una FRACCION de imgsz (cfg/default.yaml:40), no un
    # booleano. Con True se interpreta como 1.0 y detect/train.py:120-129 sortea el tamano del
    # lote en randrange(32, 2*imgsz+32): a imgsz 1024 salen lotes de hasta 2048 px, o sea 4x
    # las activaciones, y el CUDA out of memory llega en una epoca cualquiera. Y ahi no hay
    # red: la reduccion automatica de batch solo actua en la primera epoca y en una sola GPU
    # (trainer.py:522). Encontrado por revision adversarial el 2026-08-24, ya estaba lanzado.
    import argparse

    from cloud.train import RECETAS, entrena

    for nombre, receta in RECETAS.items():
        ms = receta.get("multi_scale")
        assert not isinstance(ms, bool), f"receta '{nombre}': multi_scale={ms!r} es booleano"
        if ms is not None:
            assert 0.0 < ms <= 0.5, f"receta '{nombre}': multi_scale={ms} fuera de rango sensato"

    # y la guarda salta si alguien lo vuelve a poner
    args = argparse.Namespace(
        batch=None, workers=None, receta="x", modelo="m.pt", imgsz=640, proyecto="/tmp/p",
        desde_cero=True, horas=None, epochs=1, patience=1, semilla=0, nombre="n", max_det=100,
    )
    try:
        entrena(args, pathlib.Path("no_existe.yaml"), {"multi_scale": True}, {"cuda": False, "so": "Linux"})
    except ValueError as e:
        assert "fraccion" in str(e)
    else:
        raise AssertionError("la guarda de multi_scale no salto")


def test_los_lotes_no_pueden_pasar_de_1_5x_las_activaciones_nominales():
    # Traduccion del limite a lo que de verdad importa: cuanta memoria puede pedir el peor
    # lote frente al nominal. 4x no cabe en una T4; 1.56x si.
    from cloud.train import RECETAS

    ms = RECETAS["escala"]["multi_scale"]
    imgsz, stride = 1024, 32
    mayor = (int(imgsz * (1.0 + ms) + stride) - 1) // stride * stride
    assert (mayor / imgsz) ** 2 <= 1.6, f"el peor lote pide {(mayor/imgsz)**2:.2f}x"


# ------------------------------------------------------------------- credenciales


def test_la_clave_de_roboflow_no_puede_acabar_en_un_log():
    # La clave viaja en la URL (?api_key=...), que es como la espera Roboflow. Si la peticion
    # falla, urllib mete la URL ENTERA en el mensaje de la excepcion, y ese mensaje se imprime
    # a stdout y de ahi al log de Kaggle, que se descarga con `kernels logs`. Medido el
    # 2026-08-24: con una clave que lleve un espacio, urllib lanza
    #   "URL can't contain control characters. '/w/p/1/yolov8?api_key=<LA CLAVE>'"
    from cloud.fetch_data import sin_clave

    msg = "URL can't contain control characters. '/w/p/1/yolov8?api_key=SECRETO123'"

    # con la clave conocida, desaparece
    limpio = sin_clave(msg, "SECRETO123")
    assert "SECRETO123" not in limpio
    assert "<CLAVE>" in limpio

    # y aunque no sepamos cual era, el patron api_key= se tapa igual
    ciego = sin_clave(msg, "una-clave-distinta")
    assert "SECRETO123" not in ciego


def test_una_clave_con_caracteres_raros_se_rechaza_antes_de_usarla():
    # Es la que provoca que urllib exponga la URL. Mejor pararla al leerla, y con un mensaje
    # que diga la longitud pero NO la clave.
    from cloud.fetch_data import _valida

    assert _valida("VRcC0d7LUJUuNGXOk9wW") == "VRcC0d7LUJUuNGXOk9wW"
    assert _valida("con-guiones_y_bajos") == "con-guiones_y_bajos"

    for mala in ("clave con espacio", "clave\ncon salto", "clave\tcon tab", ""):
        try:
            _valida(mala)
        except SystemExit as e:
            # el mensaje dice la longitud, pero nunca la clave. Ojo: para la cadena
            # vacia, `"" in cualquier_cosa` es siempre cierto y el assert no medía nada.
            if mala.strip():
                assert mala.strip() not in str(e)
            assert "Longitud leida" in str(e)
        else:
            raise AssertionError(f"deberia haber rechazado {mala!r}")


def test_el_entrenamiento_valida_con_el_mismo_max_det_que_la_evaluacion():
    # ultralytics valida al final de cada epoca y de ahi sale el best.pt (detect/val.py:125
    # usa args.max_det). Si no se le pasa, usa el default 300, y con fincas de 328 cajas por
    # imagen el recall de esa validacion esta topado: el "mejor" modelo se elegia con una
    # metrica recortada, distinta de la que luego se publica.
    import inspect

    from cloud import train

    fuente = inspect.getsource(train.entrena)
    assert "max_det=args.max_det" in fuente, "train() no pasa max_det a modelo.train()"
    assert train.MAX_DET >= 1000


def test_con_tope_de_horas_la_parada_temprana_no_manda():
    # Medido en Kaggle el 2026-08-24: presupuesto 10,53 h, la tirada murio a las 6,3 h en
    # la epoca 28 porque la paciencia por defecto (20) conto desde la epoca 8. Con un tope
    # de horas la parada temprana no ahorra NADA —la sesion de nube se paga entera— y solo
    # puede tirar tiempo ya comprado: 4,2 h en esa corrida.
    from cloud.train import paciencia_efectiva

    assert paciencia_efectiva(None, 10.53) == 0     # 0 = float("inf") en torch_utils.py:1003
    assert paciencia_efectiva(None, None) == 20     # sin reloj, la parada temprana si sirve
    assert paciencia_efectiva(5, 10.53) == 5        # pedirla a mano sigue mandando
    assert paciencia_efectiva(0, None) == 0
    assert paciencia_efectiva(-3, None) == 0        # nada de paciencias negativas


def test_el_mejor_no_se_elige_por_mAP50_sino_por_una_fitness_que_es_90_por_ciento_mAP50_95():
    # Por que la epoca 8 gano a la 28 aun teniendo MENOS mAP50: ultralytics puntua con
    # 0.1*mAP50 + 0.9*mAP50-95 (utils/metrics.py, DetMetrics.fitness). Con los numeros
    # reales de results.csv de la corrida del 2026-08-24, la 8 gana por un pico de
    # mAP50-95 (0,307 entre vecinas de 0,18-0,26) y ese pico ademas arranco el contador
    # de la paciencia. Es la razon de que un tope de horas y una paciencia corta se lleven
    # mal: la metrica que decide es la mas ruidosa de las dos.
    def fitness(mAP50, mAP50_95):
        return 0.1 * mAP50 + 0.9 * mAP50_95

    ep8 = fitness(0.71289, 0.30719)
    ep28 = fitness(0.75263, 0.24753)
    assert ep8 > ep28                    # gana la 8, aunque...
    assert 0.75263 > 0.71289             # ...la 28 detecta mas
    assert 28 - 8 >= 20                  # y por eso salto la paciencia de 20


def test_recoger_reconstruye_la_carpeta_que_la_api_de_kaggle_no_devuelve():
    # El endpoint que lista los ficheros de una VERSION da solo el nombre: `cloud_runs.json`
    # aparece dos veces, con tamanos distintos, porque son dos ficheros en dos carpetas. Y
    # el endpoint que si da rutas mira la SESION, asi que con una pestana del editor abierta
    # contesta cero. Sin esta reconstruccion, `--recoger` decia "0 ficheros de pesos" con
    # 40 MB guardados. Medido el 2026-08-24.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "lanzar_kaggle", pathlib.Path(RAIZ) / "kaggle" / "lanzar.py"
    )
    lanzar = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lanzar)

    rutas = lanzar.rutas_probables(
        ["cloud_runs.json", "cloud_runs.json", "todas_las_fincas_escala_yolo11m_1024_best.pt"]
    )

    # el last.pt es el que permite REANUDAR: si no se baja, la sesion siguiente reempieza
    assert "runs/todas_las_fincas_escala_yolo11m_1024/weights/last.pt" in rutas
    assert "runs/todas_las_fincas_escala_yolo11m_1024/args.yaml" in rutas  # va con el last.pt
    assert "resultados/todas_las_fincas_escala_yolo11m_1024_best.pt" in rutas
    # el nombre repetido se prueba en las dos carpetas, pero no dos veces
    assert rutas.count("resultados/cloud_runs.json") == 1
    assert "cloud_runs.json" in rutas
    assert len(rutas) == len(set(rutas))


def test_el_notebook_deja_escritas_las_rutas_de_lo_que_guarda():
    # La API no las da (ver la prueba de arriba), asi que las escribe el propio notebook.
    celdas = _notebook()
    todo = "\n".join(celdas)
    assert "MANIFIESTO.json" in todo
    assert "os.walk(WORK)" in todo


def test_el_notebook_sabe_reanudar_por_las_dos_vias():
    # /kaggle/working arranca vacio en cada corrida: sin reanudacion, cada sesion reentrena
    # desde cero y las 6,3 h de la anterior no suman. Con 30 h de cuota semanal eso es la
    # diferencia entre encadenar cuatro sesiones o repetir la primera cuatro veces.
    celdas = _notebook()
    todo = "\n".join(celdas)

    # a) la salida de la version anterior, adjuntada como Notebook Output desde la web
    assert "/kaggle/input/**/runs/**/weights/last.pt" in todo
    # b) el dataset plano que sube `lanzar.py --encadenar`, sin tocar el navegador
    assert "/kaggle/input/*/origen.json" in todo
    # y en las dos, el args.yaml viaja con el last.pt: ultralytics lo necesita para
    # reanudar con la misma receta
    assert todo.count("args.yaml") >= 3


def test_el_dataset_de_pesos_no_se_sube_publico():
    # `kaggle datasets create -u` significa --public, no --update. Son pesos entrenados
    # sobre datos con licencia de terceros y de una tirada a medias: van privados, que es
    # el default. Un flag de una letra mal leido los publicaria sin avisar.
    import importlib.util
    import inspect

    spec = importlib.util.spec_from_file_location(
        "lanzar_kaggle_pub", pathlib.Path(RAIZ) / "kaggle" / "lanzar.py"
    )
    lanzar = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lanzar)

    fuente = inspect.getsource(lanzar.encadenar)
    assert '"-u"' not in fuente and "'-u'" not in fuente
    assert '"--public"' not in fuente


def test_el_notebook_no_entrena_con_el_holdout_ciego_dentro():
    # El 2026-08-24 la primera tirada en la nube uso todas_las_fincas.yaml, que mete
    # newfarms/armah (su train Y su test) en el entrenamiento. armah es el holdout ciego del
    # repo: la unica cifra que dice que pasa en una finca nueva. Entrenar con ella dentro no
    # rompe nada, pero deja el modelo IMPOSIBLE de comparar con v10 en lo que importa, y esas
    # 6,3 h de GPU no respondieron a la pregunta que las pagaba.
    #
    # El orden correcto es al reves: medir con una finca fuera (lofo_*), y entrenar el modelo
    # final con todos los datos DESPUES, cuando ya se sabe si la receta gana.
    celdas = _notebook()
    todo = "\n".join(celdas)
    assert "lofo_armah.yaml" in todo
    assert "todas_las_fincas.yaml" not in todo, (
        "el notebook volveria a entrenar con el holdout ciego dentro"
    )


def test_los_yaml_lofo_no_dejan_entrar_su_finca_en_el_train():
    # La garantia del protocolo. Si un lofo_X mete X en el train, su cifra no vale nada y
    # nadie se daria cuenta mirando el numero.
    import yaml as _yaml

    for y in sorted((pathlib.Path(RAIZ) / "splits").glob("lofo_*.yaml")):
        finca = y.stem.replace("lofo_", "")
        cfg = _yaml.safe_load(y.read_text(encoding="utf-8"))
        # 'original' vive en count_banana_plants y 'agromatica'/'tesis' agrupan varias
        # carpetas: se comprueba contra las rutas que el propio yaml pone en val
        carpetas_val = {v.rsplit("/", 2)[0] for v in cfg["val"]}
        for t in cfg["train"]:
            assert t.rsplit("/", 2)[0] not in carpetas_val, f"{y.name}: {t} esta en train Y en val"
