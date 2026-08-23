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

PREDICCIÓN, ESCRITA ANTES DE MIRAR EL RESULTADO (2026-08-23)
------------------------------------------------------------
armah tiene imágenes de 1049x626 y plantas de ~61 px reales. A imgsz X la planta
aparente mide 61 * X/1049 px. El entrenamiento de v10 tiene dos modos:

    modo dominante  ~17 px (81% de las cajas)  ->  X = 17 * 1049/61 =~ 292
    modo grande    ~169 px (6% de las cajas)   ->  X = 169 * 1049/61 =~ 2906

Predigo que, SI manda la escala, la curva NO será plana y tendrá su mejor punto
lejos de 768 — con un máximo hacia la izquierda (~300-450), porque ahí es donde
está la masa del entrenamiento. Si en cambio el mejor punto es 768 o la curva se
mueve menos de un 20% relativo, la hipótesis de escala queda refutada y el problema
es de dominio.

Anoto también el riesgo: a imgsz muy pequeño el mAP puede subir por una razón
tramposa — menos detecciones, menos falsos positivos. Por eso se registra el RECALL
junto al mAP: el síntoma real de armah es recall 0.139, así que una mejora que no
suba el recall no vale.

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

# Cubre desde "la planta se ve como en agromatica" hasta "como en count_banana_plants".
IMGSZ_POR_DEFECTO = [320, 448, 576, 768, 960, 1280, 1600, 1920]


def dispositivo() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "0"
    except ImportError:
        pass
    return "cpu"


def barre(pesos: Path, data: Path, tamanos: list[int], device: str, conf: float) -> list[dict[str, Any]]:
    from ultralytics import YOLO

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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pesos", type=Path, default=ROOT / "models" / "banana_multifarm_v10.pt")
    ap.add_argument("--data", type=Path, default=ROOT / "splits" / "lofo_armah.yaml")
    ap.add_argument("--todas-las-fincas", action="store_true", help="barre cada splits/lofo_*.yaml")
    ap.add_argument("--imgsz", type=int, nargs="*", default=IMGSZ_POR_DEFECTO)
    ap.add_argument("--conf", type=float, default=0.001, help="umbral bajo: para mAP se quiere la curva entera")
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
        "prediccion_previa": (
            "Si manda la escala, el mejor imgsz para armah cae lejos de 768, hacia 300-450, "
            "porque el 81% del entrenamiento son plantas de 16-21 px. Si la curva es plana "
            "(menos del 20% de variación relativa) o el máximo es 768, la hipótesis se refuta."
        ),
        "fincas": {},
    }

    for d in datas:
        print(f"\n=== {d.stem} ===")
        filas = barre(args.pesos, d, args.imgsz, device, args.conf)
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
