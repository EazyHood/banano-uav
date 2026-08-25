"""Barrido de resolución: ¿el fallo en finca nueva es de ESCALA o de dominio visual?

El experimento decisivo, y es barato: no entrena nada, sólo evalúa el modelo que ya
existe a varias resoluciones de entrada. Minutos de GPU en vez de horas.

EL RAZONAMIENTO
---------------
deep/scale_audit.py midió que, a imgsz 768, la planta mediana mide 10 px en m2 y
333 px en lasuiza (35x de diferencia), y que el 81% de las cajas con las que se
entrenó v10 son plantas de 16-21 px. armah, la finca nunca vista donde el modelo
saca mAP50 0.172, tiene plantas de 45 px: cae en una banda que apenas aparece en el
entrenamiento (menos del 1% de las cajas están entre 21 y 46 px).

Si el problema es de ESCALA, cambiar imgsz mueve el tamaño aparente del objeto y el
rendimiento tiene que subir de forma clara. Si el problema es de DOMINIO visual
(otro suelo, otra luz, otra variedad, palma aceitera de fondo), el barrido saldrá
plano y habrá que atacarlo por otra vía.

LA PREDICCIÓN Y EL RESULTADO (2026-08-23) — se deja escrita porque falló a medias
---------------------------------------------------------------------------------
Antes de medir se predijo: "si manda la escala, el mejor imgsz para armah cae lejos
de 768, hacia 300-450, porque el 81% del entrenamiento son plantas de 16-21 px y a
imgsz ~292 las de armah se verían de ese tamaño".

Medido sobre armah con los pesos v10 sin tocar (el valor a 768 reprodujo el fichero
real_eval, 0.1724 frente a 0.1723, así que el instrumento está validado):

    imgsz  320 -> mAP50 0.0001   R 0.0160
    imgsz  512 -> mAP50 0.0146   R 0.0334
    imgsz  640 -> mAP50 0.0848   R 0.0922
    imgsz  768 -> mAP50 0.1724   R 0.1386   <- el que usa el repo hoy
    imgsz  896 -> mAP50 0.2573   R 0.2068
    imgsz 1024 -> mAP50 0.2847   R 0.2290   <- óptimo, +65% de mAP y +65% de recall
    imgsz 1152 -> mAP50 0.2708   R 0.2195
    imgsz 1280 -> mAP50 0.2694   R 0.2152
    imgsz 1536 -> mAP50 0.2470   R 0.1949

La mitad acertada: la curva NO es plana, la escala manda, y el punto que usa el repo
no es el bueno. Máximo interior con campana a los dos lados, no un artefacto de borde.
La mitad equivocada, y conviene entender por qué: el óptimo está ARRIBA (1024), no
abajo. Igualar el tamaño *relativo* al del entrenamiento no basta, porque encoger la
imagen destruye los píxeles de textura con los que se reconoce una roseta. A 320 px
la planta tiene el tamaño "correcto" y aun así el mAP es 0.0001. El detector necesita
resolución absoluta, no sólo proporción.

Sube el recall a la vez que el mAP (0.1386 -> 0.2290), así que no es la mejora
tramposa de "detectar menos y fallar menos".

AVISO DE MÉTODO: ese 1024 se eligió mirando armah, que es el holdout. Afinar un
hiperparámetro sobre el conjunto con el que luego presumes es exactamente el error
que este repo ya corrigió una vez en eval_count.py. Por eso existe --todas-las-fincas:
el imgsz que se publique debe salir del promedio LOFO, no de armah sola.

Uso:
    python cloud/scale_sweep.py --pesos models/banana_multifarm_v10.pt --data splits/lofo_armah.yaml
    python cloud/scale_sweep.py --todas-las-fincas          # barre cada lofo_*.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

# Por debajo de 512 el mAP se hunde a ~0 en todas las fincas medidas: no aporta puntos
# a la curva y cuesta una pasada de validacion cada uno. Se barre de 640 en adelante.
IMGSZ_POR_DEFECTO = [640, 768, 896, 1024, 1152, 1280, 1536, 1920]


def dispositivo() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "0"
    except ImportError:
        pass
    return "cpu"


def barre(pesos: Path, data: Path, tamanos: list[int], device: str, conf: float, max_det: int) -> list[dict[str, Any]]:
    from ultralytics import YOLO

    if str(ROOT) not in sys.path:  # ejecutado como script, la raíz del repo no está
        sys.path.insert(0, str(ROOT))
    from cloud.rutas import resuelve

    # ultralytics no resuelve el `path:` relativo contra el YAML sino contra su propio
    # datasets_dir, asi que los splits versionados no abren sin esto (medido 2026-08-24).
    data = resuelve(data)

    filas: list[dict[str, Any]] = []
    for imgsz in tamanos:
        # ultralytics exige múltiplos del stride máximo (32)
        imgsz = int(round(imgsz / 32) * 32)
        t0 = time.time()
        modelo = YOLO(str(pesos))  # recargar evita que val() arrastre estado del anterior
        try:
            m = modelo.val(
                data=str(data),
                imgsz=imgsz,
                device=device,
                conf=conf,
                max_det=max_det,
                verbose=False,
                plots=False,
                save_json=False,
            )
        except Exception as e:
            print(f"  imgsz {imgsz:5d}  FALLO: {e}", file=sys.stderr)
            continue
        fila = {
            "imgsz": imgsz,
            "mAP50": round(float(m.box.map50), 4),
            "mAP50_95": round(float(m.box.map), 4),
            "precision": round(float(m.box.mp), 4),
            "recall": round(float(m.box.mr), 4),
            "segundos": round(time.time() - t0, 1),
        }
        filas.append(fila)
        print(
            f"  imgsz {imgsz:5d}   mAP50 {fila['mAP50']:.4f}   R {fila['recall']:.4f}   "
            f"P {fila['precision']:.4f}   ({fila['segundos']}s)"
        )
    return filas


def resumen_lofo(fincas: dict[str, Any]) -> list[dict[str, Any]]:
    """Agrega el barrido de TODAS las fincas retenidas, por resolucion.

    Elegir el imgsz mirando una sola finca es afinar sobre el holdout, y este repo ya
    cometio ese error una vez con el umbral de confianza (lo corrigio eval_count.py con
    calibracion cruzada). Aqui la decision sale del promedio, y ademas se reporta en
    CUANTAS fincas gana cada resolucion: un promedio alto que viene de una sola finca
    disparada no es una recomendacion, es un accidente.
    """
    por_imgsz: dict[int, list[tuple[str, float, float]]] = {}
    for nombre, datos in fincas.items():
        for fila in datos.get("barrido", []):
            por_imgsz.setdefault(fila["imgsz"], []).append(
                (nombre, fila["mAP50"], fila["recall"])
            )

    # en cuantas fincas es ESTA resolucion la mejor
    mejor_de: dict[int, int] = {}
    for datos in fincas.values():
        filas = datos.get("barrido", [])
        if filas:
            ganadora = max(filas, key=lambda f: f["mAP50"])["imgsz"]
            mejor_de[ganadora] = mejor_de.get(ganadora, 0) + 1

    out: list[dict[str, Any]] = []
    for imgsz, vals in sorted(por_imgsz.items()):
        maps = [v[1] for v in vals]
        recs = [v[2] for v in vals]
        out.append({
            "imgsz": imgsz,
            "fincas": len(vals),
            "mAP50_medio": round(sum(maps) / len(maps), 4),
            "mAP50_min": round(min(maps), 4),
            "mAP50_max": round(max(maps), 4),
            "recall_medio": round(sum(recs) / len(recs), 4),
            "gana_en_n_fincas": mejor_de.get(imgsz, 0),
            "peor_finca": min(vals, key=lambda v: v[1])[0],
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pesos", type=Path, default=ROOT / "models" / "banana_multifarm_v10.pt")
    ap.add_argument("--data", type=Path, default=ROOT / "splits" / "lofo_armah.yaml")
    ap.add_argument("--todas-las-fincas", action="store_true", help="barre cada splits/lofo_*.yaml")
    ap.add_argument("--imgsz", type=int, nargs="*", default=IMGSZ_POR_DEFECTO)
    ap.add_argument("--conf", type=float, default=0.001, help="umbral bajo: para mAP se quiere la curva entera")
    ap.add_argument("--max-det", type=int, default=1000,
                    help="el default de ultralytics es 300 y hay imagenes con 600 plantas")
    ap.add_argument("--device", default=None)
    ap.add_argument("--salida", type=Path, default=ROOT / "real_eval" / "scale_sweep.json")
    args = ap.parse_args()

    if not args.pesos.exists():
        print(f"No están los pesos: {args.pesos}", file=sys.stderr)
        return 1

    device = args.device or dispositivo()
    if device == "cpu":
        print("AVISO: sin GPU. El barrido funcionará pero irá lento.", file=sys.stderr)

    datas = sorted((ROOT / "splits").glob("lofo_*.yaml")) if args.todas_las_fincas else [args.data]
    datas = [d for d in datas if d.exists()]
    if not datas:
        print("No hay YAML que evaluar. Genera con cloud/make_splits.py", file=sys.stderr)
        return 1

    informe: dict[str, Any] = {
        "pesos": str(args.pesos.name),
        "device": device,
        "conf": args.conf,
        "max_det": args.max_det,
        "nota_metodo": (
            "El imgsz que se publique debe salir del promedio LOFO, no de una sola finca: "
            "elegirlo mirando armah es afinar sobre el holdout."
        ),
        "fincas": {},
    }

    for d in datas:
        print(f"\n=== {d.stem} ===")
        filas = barre(args.pesos, d, args.imgsz, device, args.conf, args.max_det)
        if not filas:
            continue
        mejor = max(filas, key=lambda f: f["mAP50"])
        base = next((f for f in filas if f["imgsz"] == 768), None)
        veredicto = "sin base a 768"
        if base and base["mAP50"] > 0:
            ganancia = mejor["mAP50"] / base["mAP50"] - 1
            gan_recall = (mejor["recall"] / base["recall"] - 1) if base["recall"] > 0 else float("inf")
            if mejor["imgsz"] == 768 or ganancia < 0.20:
                veredicto = f"ESCALA REFUTADA (mejor {mejor['imgsz']}, +{ganancia*100:.0f}% sobre 768)"
            elif gan_recall <= 0:
                veredicto = f"DUDOSO: sube mAP +{ganancia*100:.0f}% pero NO sube el recall"
            else:
                veredicto = (
                    f"ESCALA CONFIRMADA: mejor imgsz {mejor['imgsz']} da mAP50 {mejor['mAP50']:.4f} "
                    f"(+{ganancia*100:.0f}%) y recall {mejor['recall']:.4f} (+{gan_recall*100:.0f}%)"
                )
        print(f"  -> {veredicto}")
        informe["fincas"][d.stem] = {"barrido": filas, "mejor": mejor, "base_768": base, "veredicto": veredicto}

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(json.dumps(informe, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nInforme -> {args.salida}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("YOLO_VERBOSE", "false")
    raise SystemExit(main())
