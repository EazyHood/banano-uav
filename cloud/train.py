"""Entrenamiento que corre en cualquier máquina, no sólo en el PC del autor.

Sustituye a deep/train_v12.py, que estaba atado a este equipo: `device=0` fijo,
`workers=0` porque en Windows fallaba el DataLoader, `batch=4` porque la RTX 5060
tiene 8 GiB, y rutas `C:/Users/jhona/...` dentro de los YAML de datos. Nada de eso
vale en una GPU prestada.

Aquí el dispositivo, los workers y el batch se deciden en tiempo de ejecución, y los
datos salen de splits/ (generados por cloud/make_splits.py, con rutas relativas).

RECETAS
-------
`v10`     lo que de verdad entrenó al modelo publicado (leído de runs10/.../args.yaml):
          los defaults de YOLO, sin rotación ni volteo. Es el control honesto.
`cenital` la de v12: rotación libre y volteo vertical, que la vista nadir permite.
`escala`  cenital + el arreglo de lo que se midió: entre fincas hay 11x de diferencia en
          el tamaño de la planta (mediana de 15,6 px a 175,3 px a imgsz 768), y ningún
          `scale` escalar cubre eso. Usa `scale` como tupla absoluta (0.25, 2.5) y
          `multi_scale`.

Ninguna es "la buena" por decreto: se entrenan y se comparan con el mismo protocolo
LOFO. Si `escala` no gana en las fincas retenidas, no se publica.

CORTES DE SESIÓN
----------------
Las GPU gratuitas se cortan. El script reanuda solo: si encuentra `last.pt` de una
tirada con el mismo nombre, continúa desde ahí en vez de empezar de cero. En Kaggle eso
significa apuntar --proyecto a /kaggle/working, que es lo que se autoguarda.

Uso:
    python cloud/train.py --data splits/todas_las_fincas.yaml --receta escala
    python cloud/train.py --data splits/lofo_armah.yaml --receta v10 --epochs 60
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
    "v10": {
        "_nota": (
            "lo que de verdad entrenó al modelo publicado, leído de runs10/.../args.yaml: "
            "degrees 0, flipud 0, scale 0.5. Es decir, los defaults de YOLO. La "
            "'augmentation cenital' que describe el docstring de deep/train_v12.py se aplicó "
            "sólo a v12, que se decidió no publicar. El modelo que la gente descarga nunca "
            "vio una planta rotada ni volteada. Este es el control honesto."
        ),
        "degrees": 0.0,
        "flipud": 0.0,
        "scale": 0.5,
    },
    "cenital": {
        "_nota": (
            "la receta de v12: aprovecha que la vista nadir es invariante a rotación y a "
            "volteo vertical, cosa que los defaults de YOLO no explotan."
        ),
        "degrees": 180.0,
        "flipud": 0.5,
        "scale": 0.6,
    },
    "escala": {
        "_nota": (
            "cenital + el arreglo del problema medido: entre fincas hay 11x de diferencia en "
            "el tamaño de la planta (mediana de 15,6 px a 175,3 px a imgsz 768). `scale` "
            "admite TUPLA (min,max) como factores absolutos —augment.py:1085-1134—, así que "
            "(0.25, 2.5) cubre 10x, justo la dispersión real. `scale` como float sólo daba "
            "0.4x-1.6x. `multi_scale` mueve además la resolución de entrada entre lotes: es "
            "una FRACCIÓN de imgsz (default.yaml:40), NO un booleano — con 0.25 los lotes van "
            "de 0.75x a 1.25x. NO se usa copy_paste: en ultralytics 8.4.117 es exclusivo de "
            "segmentación (default.yaml:131) y aquí las etiquetas son cajas."
        ),
        "degrees": 180.0,
        "flipud": 0.5,
        "scale": (0.25, 2.5),
        "multi_scale": 0.25,
        "hsv_h": 0.02,
        "hsv_s": 0.8,
        "hsv_v": 0.5,
        "mosaic": 1.0,
        "close_mosaic": 10,
    },
}

# El default de ultralytics es 300 (cfg/default.yaml:57) y en el holdout hay una imagen
# con 600 plantas reales: ahí el recall estaba topado al 50% por construcción, no por el
# modelo. Las fincas densas llegan a 328 cajas por imagen.
MAX_DET = 1000


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

    # UNA sola GPU por defecto, aunque haya dos. Con device="0,1" ultralytics no entrena en
    # el proceso actual: lanza `torch.distributed.run` (DDP) sobre un fichero temporal que
    # genera al vuelo, y eso muere dentro de un notebook de Kaggle. Medido el 2026-08-24:
    #   Command '[...torch.distributed.run --nproc_per_node 2 ...DDP/_temp_....py]'
    #   returned non-zero exit status 1
    # El entrenamiento arranco, hizo UNA iteracion y murio; el notebook siguio como si nada
    # y la sesion se perdio. La segunda T4 no compensa ese riesgo: se pide a proposito.
    device = "0"

    if batch is None:
        vram = info.get("vram_gb", 8)
        # medido en la RTX 5060 (8 GiB): con batch 8 a 768px CUDA paginaba a memoria
        # compartida y el paso iba 9 veces más lento. Se deja margen.
        batch = 4 if vram < 10 else (8 if vram < 16 else 16)

    if workers is None:
        # En Windows el DataLoader con workers>0 murió repetidamente (v2 y seg2), y tras
        # matar un entrenamiento quedaban segmentos de memoria compartida que reventaban
        # el arranque siguiente. En Linux (que es lo que hay en la nube) no pasa.
        workers = 0 if info["so"] == "Windows" else 8

    return device, batch, workers


def paciencia_efectiva(patience: int | None, horas: float | None) -> int:
    """Épocas sin mejorar antes de cortar. 0 = no cortar nunca (torch_utils.py:1003).

    Con --horas el RELOJ ya acota la tirada, y ahí la parada temprana no puede ahorrar
    nada: la sesión de nube se paga entera se use o no. Sólo puede quitar tiempo ya
    pagado. Medido en Kaggle el 2026-08-24, con la paciencia por defecto de 20:

        presupuesto 10,53 h  ->  murió a las 6,3 h, en la época 28, con 4,2 h sin usar

    y el "mejor" que dejó fue la época 8, elegida por una fitness que es 90% mAP50-95
    (metrics.py: 0.1*mAP50 + 0.9*mAP50-95). Esa época tuvo un pico de mAP50-95 de 0,307
    entre vecinas de 0,18-0,26; la 28 tenía MÁS mAP50 (0,753 frente a 0,713) y aún subía.
    O sea que la parada temprana cortó una tirada que mejoraba, midiendo el pico de una
    métrica ruidosa. Con tope de horas se desactiva salvo que se pida a mano.
    """
    if patience is not None:
        return max(0, patience)
    return 0 if horas else 20


def entrena(args: argparse.Namespace, data: Path, receta: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
    # multi_scale es una FRACCION de imgsz, no un interruptor. Poner True lo convierte en 1.0
    # y los lotes se sortean entre 32 px y 2*imgsz: a imgsz 1024 eso son lotes de 2048 px con
    # 4x las activaciones, y un CUDA out of memory en una epoca cualquiera. Peor aun, la
    # reduccion automatica de batch de ultralytics solo actua en la PRIMERA epoca y en una
    # sola GPU (trainer.py:522), asi que a partir de ahi no hay red y la sesion muere.
    if isinstance(receta.get("multi_scale"), bool):
        raise ValueError(
            "multi_scale es una fraccion de imgsz (0.25 = +/-25%), no un booleano. "
            "Con True se interpreta como 1.0 y sortea lotes de hasta 2*imgsz."
        )

    from ultralytics import YOLO

    if str(ROOT) not in sys.path:  # ejecutado como script, la raíz del repo no está
        sys.path.insert(0, str(ROOT))
    from cloud.rutas import resuelve

    # Sin esto, un data.yaml con `path:` relativo no lo abre ultralytics: lo busca contra su
    # propio datasets_dir y no contra el fichero. El nombre de la tirada se saca del yaml
    # ORIGINAL, no del resuelto, para que no cambie segun donde se lance.
    data_original = data
    data = resuelve(data)

    device, batch, workers = decide(info, args.batch, args.workers)
    nombre = args.nombre or f"{data_original.stem}_{args.receta}_{args.modelo.replace('.pt','')}_{args.imgsz}"
    proyecto = args.proyecto

    ultimo = Path(proyecto) / nombre / "weights" / "last.pt"
    reanudar = ultimo.exists() and not args.desde_cero
    punto = str(ultimo) if reanudar else args.modelo
    if reanudar:
        print(f"  reanudando desde {ultimo}")

    pac = paciencia_efectiva(args.patience, args.horas)
    if pac == 0:
        print("  parada temprana DESACTIVADA: manda el tope de horas")

    hiper = {k: v for k, v in receta.items() if not k.startswith("_")}

    if args.horas:
        # `time` de ultralytics manda sobre `epochs` y lo comprueba al final de cada epoca
        # (engine/trainer.py:547). Es la forma honesta de encajar en una sesion con limite:
        # en vez de adivinar cuantas epocas caben —la primera estimacion dio 24 h para 40—,
        # se le dice cuanto tiempo tiene y para solo, dejando el best.pt escrito.
        hiper["time"] = args.horas
    t0 = time.time()
    modelo = YOLO(punto)
    modelo.train(
        data=str(data),
        epochs=args.epochs,
        patience=pac,
        imgsz=args.imgsz,
        batch=batch,
        workers=workers,
        device=device,
        # También en el ENTRENAMIENTO, no sólo en la evaluación final: ultralytics valida al
        # terminar cada época y de ahí sale el best.pt (val.py:125 usa este mismo argumento).
        # Con el default de 300 y una finca de 328 cajas por imagen, el recall de esa
        # validación está topado por construcción, así que el "mejor" modelo se elegía con
        # una métrica recortada. Encontrado por revisión adversarial el 2026-08-24.
        max_det=args.max_det,
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

    metricas = modelo.val(
        data=str(data), imgsz=args.imgsz, device=device,
        max_det=args.max_det, verbose=False, plots=False,
    )
    resultado = {
        "data": data_original.name,
        "receta": args.receta,
        "receta_hiper": hiper,
        "modelo_base": args.modelo,
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "horas_tope": args.horas,
        "patience": pac,
        "batch": batch,
        "workers": workers,
        "device": device,
        "max_det": args.max_det,
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
    ap.add_argument("--max-det", type=int, default=MAX_DET, help="tope de detecciones por imagen")
    ap.add_argument("--modelo", default="yolo11m.pt")
    ap.add_argument("--imgsz", type=int, default=768)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--horas", type=float, default=None,
                    help="tope de horas; manda sobre --epochs y para solo al agotarse")
    ap.add_argument("--patience", type=int, default=None,
                    help="épocas sin mejorar antes de cortar; 0 desactiva. Por defecto, "
                         "20 sin --horas y 0 con --horas (el reloj ya acota)")
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
