"""Entrenamiento que corre en cualquier máquina, no sólo en el PC del autor.

Sustituye a deep/train_v12.py, que estaba atado a este equipo: `device=0` fijo,
`workers=0` porque en Windows fallaba el DataLoader, `batch=4` porque la RTX 5060
tiene 8 GiB, y rutas `C:/Users/jhona/...` dentro de los YAML de datos. Nada de eso
vale en una GPU prestada.

Aquí el dispositivo, los workers y el batch se deciden en tiempo de ejecución, y los
datos salen de splits/ (generados por cloud/make_splits.py, con rutas relativas).

RECETAS
-------
`base`   reproduce la receta de v10/v12 tal cual, para tener control con el que comparar.
`escala` ataca lo que midió deep/scale_audit.py: entre las fincas hay 35x de diferencia
         de tamaño de planta y el 81% del entrenamiento son plantas de 16-21 px, así que
         una finca nueva con plantas de otro tamaño cae fuera de lo aprendido.
         `scale=0.6` sólo cubre 0.4x-1.6x (4x). Esta receta sube `scale` y activa
         `multi_scale`, que varía la resolución de entrada entre lotes: juntos cubren un
         rango mucho más ancho. Además sube el jitter de color, porque el cambio de finca
         trae también cambio de suelo y de luz.

Ninguna de las dos es "la buena" por decreto: se entrenan y se comparan con el mismo
protocolo LOFO. Si `escala` no gana en las fincas retenidas, no se publica.

CORTES DE SESIÓN
----------------
Las GPU gratuitas se cortan. El script reanuda solo: si encuentra `last.pt` de una
tirada con el mismo nombre, continúa desde ahí en vez de empezar de cero. Por eso
importa que el proyecto apunte a almacenamiento que sobreviva a la sesión.

Uso:
    python cloud/train.py --data splits/todas_las_fincas.yaml --receta escala
    python cloud/train.py --data splits/lofo_armah.yaml --receta base --epochs 60
    python cloud/train.py --lofo --receta escala      # entrena una vez por finca retenida
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("WANDB_DISABLED", "true")  # si no, pide login y cuelga un run desatendido

ROOT = Path(__file__).resolve().parents[1]

RECETAS: dict[str, dict[str, Any]] = {
    "base": {
        "_nota": "receta de v10/v12 tal cual: el control con el que comparar.",
        "degrees": 180.0,
        "flipud": 0.5,
        "scale": 0.6,
    },
    "escala": {
        "_nota": (
            "contra los 35x de diferencia de tamaño entre fincas. scale 0.9 cubre 0.1x-1.9x "
            "y multi_scale mueve además la resolución de entrada entre lotes; el jitter de "
            "color acompaña porque cambiar de finca cambia suelo y luz."
        ),
        "degrees": 180.0,
        "flipud": 0.5,
        "scale": 0.9,
        "multi_scale": True,
        "hsv_h": 0.02,
        "hsv_s": 0.8,
        "hsv_v": 0.5,
        "mosaic": 1.0,
        "close_mosaic": 10,
    },
}


def entorno() -> dict[str, Any]:
    info: dict[str, Any] = {"so": platform.system(), "python": platform.python_version()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpus"] = torch.cuda.device_count()
            info["gpu_nombre"] = torch.cuda.get_device_name(0)
            info["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
    except ImportError:
        info["torch"] = None
    return info


def decide(info: dict[str, Any], batch: int | None, workers: int | None) -> tuple[str, int, int]:
    """device, batch y workers según la máquina donde toque correr."""
    if not info.get("cuda"):
        return "cpu", batch or 4, workers or 2

    device = ",".join(str(i) for i in range(info["gpus"])) if info.get("gpus", 1) > 1 else "0"

    if batch is None:
        vram = info.get("vram_gb", 8)
        # medido en la RTX 5060 (8 GiB): con batch 8 a 768px CUDA paginaba a memoria
        # compartida y el paso iba 9 veces más lento. Se deja margen.
        batch = 4 if vram < 10 else (8 if vram < 16 else 16)
        if info.get("gpus", 1) > 1:
            batch *= info["gpus"]

    if workers is None:
        # En Windows el DataLoader con workers>0 murió repetidamente (v2 y seg2), y tras
        # matar un entrenamiento quedaban segmentos de memoria compartida que reventaban
        # el arranque siguiente. En Linux (que es lo que hay en la nube) no pasa.
        workers = 0 if info["so"] == "Windows" else 8

    return device, batch, workers


def entrena(args: argparse.Namespace, data: Path, receta: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
    from ultralytics import YOLO

    device, batch, workers = decide(info, args.batch, args.workers)
    nombre = args.nombre or f"{data.stem}_{args.receta}_{args.modelo.replace('.pt','')}_{args.imgsz}"
    proyecto = args.proyecto

    ultimo = Path(proyecto) / nombre / "weights" / "last.pt"
    reanudar = ultimo.exists() and not args.desde_cero
    punto = str(ultimo) if reanudar else args.modelo
    if reanudar:
        print(f"  reanudando desde {ultimo}")

    hiper = {k: v for k, v in receta.items() if not k.startswith("_")}
    t0 = time.time()
    modelo = YOLO(punto)
    modelo.train(
        data=str(data),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=batch,
        workers=workers,
        device=device,
        project=proyecto,
        name=nombre,
        exist_ok=True,
        resume=reanudar,
        seed=args.semilla,
        deterministic=False,
        val=True,
        plots=False,
        **hiper,
    )

    metricas = modelo.val(data=str(data), imgsz=args.imgsz, device=device, verbose=False, plots=False)
    resultado = {
        "data": data.name,
        "receta": args.receta,
        "receta_hiper": hiper,
        "modelo_base": args.modelo,
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": batch,
        "workers": workers,
        "device": device,
        "entorno": info,
        "minutos": round((time.time() - t0) / 60, 1),
        "pesos": str(Path(proyecto) / nombre / "weights" / "best.pt"),
        "metricas": {
            "mAP50": round(float(metricas.box.map50), 4),
            "mAP50_95": round(float(metricas.box.map), 4),
            "precision": round(float(metricas.box.mp), 4),
            "recall": round(float(metricas.box.mr), 4),
        },
    }
    print(
        f"  -> mAP50 {resultado['metricas']['mAP50']:.4f}  "
        f"recall {resultado['metricas']['recall']:.4f}  ({resultado['minutos']} min)"
    )
    return resultado


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "splits" / "todas_las_fincas.yaml")
    ap.add_argument("--lofo", action="store_true", help="una tirada por cada splits/lofo_*.yaml")
    ap.add_argument("--receta", choices=sorted(RECETAS), default="escala")
    ap.add_argument("--modelo", default="yolo11m.pt")
    ap.add_argument("--imgsz", type=int, default=768)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch", type=int, default=None, help="por defecto, según la VRAM")
    ap.add_argument("--workers", type=int, default=None, help="por defecto, 0 en Windows y 8 fuera")
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--nombre", default=None)
    ap.add_argument("--proyecto", default=str(ROOT / "runs_cloud"))
    ap.add_argument("--desde-cero", action="store_true", help="ignora last.pt y reempieza")
    ap.add_argument("--salida", type=Path, default=ROOT / "real_eval" / "cloud_runs.json")
    args = ap.parse_args()

    info = entorno()
    print(f"Entorno: {json.dumps(info, ensure_ascii=False)}")
    if not info.get("cuda"):
        print("AVISO: no hay GPU. Entrenar aquí tardará días; esto está pensado para la nube.", file=sys.stderr)

    receta = RECETAS[args.receta]
    print(f"Receta '{args.receta}': {receta['_nota']}\n")

    datas = sorted((ROOT / "splits").glob("lofo_*.yaml")) if args.lofo else [args.data]
    datas = [d for d in datas if d.exists()]
    if not datas:
        print("No hay YAML. Genera con cloud/make_splits.py", file=sys.stderr)
        return 1

    # Se va escribiendo tirada a tirada: si la sesión se corta, no se pierde lo ya pagado.
    hechas: list[dict[str, Any]] = []
    if args.salida.exists():
        try:
            hechas = json.loads(args.salida.read_text(encoding="utf-8")).get("tiradas", [])
        except json.JSONDecodeError:
            hechas = []

    for d in datas:
        print(f"=== {d.stem} ({args.receta}) ===")
        try:
            hechas.append(entrena(args, d, receta, info))
        except KeyboardInterrupt:
            print("interrumpido por el usuario", file=sys.stderr)
            break
        except Exception as e:
            print(f"  FALLO en {d.stem}: {e}", file=sys.stderr)
            hechas.append({"data": d.name, "receta": args.receta, "error": str(e)})
        args.salida.parent.mkdir(parents=True, exist_ok=True)
        args.salida.write_text(json.dumps({"tiradas": hechas}, indent=2, ensure_ascii=False), encoding="utf-8")

    validas = [t for t in hechas if "metricas" in t and t.get("receta") == args.receta]
    lofo = [t for t in validas if t["data"].startswith("lofo_")]
    if len(lofo) > 1:
        media = sum(t["metricas"]["mAP50"] for t in lofo) / len(lofo)
        media_r = sum(t["metricas"]["recall"] for t in lofo) / len(lofo)
        print(f"\nLOFO ({len(lofo)} fincas) receta '{args.receta}': mAP50 medio {media:.4f}, recall medio {media_r:.4f}")
        for t in sorted(lofo, key=lambda x: x["metricas"]["mAP50"]):
            print(f"  {t['data']:28s} mAP50 {t['metricas']['mAP50']:.4f}  recall {t['metricas']['recall']:.4f}")

    print(f"\nRegistro -> {args.salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
