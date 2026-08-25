"""De cero a entrenando, en cualquier máquina con GPU. Un solo comando.

Por qué existe. El pipeline vive dentro del notebook de Kaggle, y eso lo ata a Kaggle: si
un día se agota la cuota —pasó el 25 de agosto de 2026, con 0 h libres hasta el sábado— no
hay forma rápida de seguir en otra parte. Aquí está el mismo camino sin depender de ningún
notebook: sirve en un Studio de Lightning, en una VM alquilada, en un contenedor, o en
cualquier sitio donde haya una GPU y Python.

    python cloud/arranque.py --horas 8

Hace, por orden: comprobar la GPU · descargar los datos de Roboflow · repartirlos por finca
· comprobar que no hay fugas · entrenar · dejar los pesos donde se puedan recoger.

Sobre la clave de Roboflow: se lee de ROBOFLOW_API_KEY o de ~/.roboflow_key, igual que en
todos los demás scripts. En una máquina prestada, lo limpio es exportarla en el entorno.

    export ROBOFLOW_API_KEY=...      # Linux/Mac
    $env:ROBOFLOW_API_KEY = "..."    # PowerShell

DESATENDIDO. Si la máquina se puede quedar sola, lánzalo de fondo y desconéctate:

    nohup python cloud/arranque.py --horas 8 > arranque.log 2>&1 &

`--horas` es un TOPE, no un objetivo: ultralytics lo comprueba al final de cada época y
para dejando el modelo escrito, así que la tirada siempre cabe en el tiempo que le des.
Ponle menos de lo que dure tu sesión o tu crédito.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Un YOLO11m a 1024 px con lote 8 no cabe cómodo por debajo de esto. Con menos, el script
# avisa y baja el lote en vez de morir a mitad con un CUDA out of memory.
VRAM_MINIMA_GB = 12.0


def paso(n: int, total: int, texto: str) -> None:
    print(f"\n[{n}/{total}] {texto}", flush=True)


def corre(cmd: list[str | Path], donde: Path = ROOT) -> int:
    print("  $ " + " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], cwd=str(donde)).returncode


def revisa_gpu() -> tuple[bool, float]:
    try:
        import torch
    except ImportError:
        print("  torch no está instalado todavía", flush=True)
        return False, 0.0
    if not torch.cuda.is_available():
        print("  NO hay GPU visible", flush=True)
        return False, 0.0
    props = torch.cuda.get_device_properties(0)
    vram = props.total_memory / 1024**3
    print(f"  {torch.cuda.device_count()}x {props.name}, {vram:.1f} GB, "
          f"sm_{props.major}{props.minor}", flush=True)
    # is_available() puede decir True y reventar en la primera operación real: le pasa a la
    # P100 con la imagen de Kaggle, que trae PyTorch sin kernels de Pascal.
    torch.zeros(8, device="cuda").sum().item()
    print("  operación CUDA de prueba: OK", flush=True)
    return True, vram


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--horas", type=float, default=8.0, help="tope de horas de entrenamiento")
    ap.add_argument("--datos", type=Path, default=Path("/tmp/realdata"),
                    help="dónde dejar las imágenes (fuera del repo)")
    ap.add_argument("--splits", type=Path, default=ROOT / "splits_local")
    ap.add_argument("--data", default="lofo_armah.yaml",
                    help="qué reparto entrenar; lofo_armah deja la finca ciega fuera")
    ap.add_argument("--receta", default="escala")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--modelo", default="yolo11m.pt")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--nuevas", action="store_true", help="incluir las fincas nuevas sin usar")
    ap.add_argument("--salta-descarga", action="store_true", help="los datos ya están")
    args = ap.parse_args()

    t0 = time.time()
    TOTAL = 5

    paso(1, TOTAL, "GPU")
    hay_gpu, vram = revisa_gpu()
    if not hay_gpu:
        print("\n  Sin GPU esto tardaría días. Instala torch con CUDA o cambia de máquina.",
              file=sys.stderr)
        return 1
    batch = args.batch
    if batch is None and vram < VRAM_MINIMA_GB:
        batch = 4
        print(f"  AVISO: {vram:.1f} GB es poco para {args.imgsz} px; se baja el lote a 4.")

    paso(2, TOTAL, f"datos de Roboflow -> {args.datos}")
    if args.salta_descarga:
        print("  saltado a petición")
    else:
        cmd: list[str | Path] = [sys.executable, ROOT / "cloud" / "fetch_data.py",
                             "--destino", args.datos]
        if corre(cmd) != 0:
            print("\n  Falló la descarga. ¿Está puesta ROBOFLOW_API_KEY?", file=sys.stderr)
            return 1
        if args.nuevas:
            corre(cmd + ["--nuevas"])

    paso(3, TOTAL, f"repartir por finca -> {args.splits}")
    if corre([sys.executable, ROOT / "cloud" / "make_splits.py",
              "--raiz", args.datos, "--salida", args.splits,
              "--raiz-declarada", args.datos]) != 0:
        return 1

    paso(4, TOTAL, "comprobar fugas antes de entrenar")
    # Si una imagen de validación está en el entrenamiento, la cifra final no vale nada y
    # más vale saberlo ANTES de gastar las horas, no después.
    for yaml in sorted(args.splits.glob("lofo_*.yaml")):
        r = subprocess.run(
            [sys.executable, str(ROOT / "deep" / "leak_audit.py"), "--data", str(yaml)],
            capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(ROOT),
        )
        linea = next((ln.strip() for ln in r.stdout.splitlines() if "TOTAL" in ln), "?")
        print(f"  {yaml.name:28s} {linea}")

    paso(5, TOTAL, f"entrenar {args.data} durante {args.horas} h como mucho")
    cmd = [sys.executable, ROOT / "cloud" / "train.py",
           "--data", args.splits / args.data,
           "--receta", args.receta, "--modelo", args.modelo,
           "--imgsz", str(args.imgsz), "--horas", str(args.horas),
           "--epochs", "300", "--proyecto", ROOT / "runs_cloud" / "arranque"]
    if batch:
        cmd += ["--batch", str(batch)]
    rc = corre(cmd)

    pesos = sorted((ROOT / "runs_cloud" / "arranque").rglob("best.pt"))
    print(f"\nTerminado en {(time.time() - t0) / 3600:.2f} h (código {rc})")
    for p in pesos:
        print(f"  pesos: {p}  ({p.stat().st_size / 1e6:.1f} MB)")
    if not pesos:
        print("  NO hay pesos: mira el error de arriba.", file=sys.stderr)
        return rc or 1
    print("\nBájatelos antes de apagar la máquina: en una VM prestada, al pararla se van.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
