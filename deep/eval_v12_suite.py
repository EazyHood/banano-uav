r"""Suite de evaluación de un modelo multi-finca en los 4 protocolos del repo.

Por qué existe: en julio-2026 el 0.747 de v10 no quedó escrito en ningún fichero,
y en agosto-2026 esta misma suite vivía en un directorio temporal que se perdió al
cerrarse la sesión. Ahora vive en el repo y deja huella en real_eval/.

Los 4 protocolos, de más fácil a más honesto:
  1. samefarm   - test de la finca original (t768.yaml). Lo que casi todos publican.
  2. seenfarms  - holdout de las fincas que SÍ están en el train (holdout.yaml).
  3. newfarms   - test de las 3 fincas añadidas en v12 (holdout_newfarms.yaml).
  4. armah      - finca que NINGÚN modelo ha visto (holdout_armah.yaml). La cifra dura.

Además del mAP mide el ERROR DE CONTEO con barrido de confianza, que es lo que de
verdad le importa a una finca, y compara contra los baselines de v10 ya registrados.

Uso:
  python deep/eval_v12_suite.py
  python deep/eval_v12_suite.py --weights runs12/.../last.pt --tag v12b
  python deep/eval_v12_suite.py --batch 2   # GPU con poca VRAM libre

En Windows, con el venv del proyecto: .venv\Scripts\python.exe deep\eval_v12_suite.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# sys.executable, no una ruta fija: `.venv/Scripts/python.exe` sólo existe en Windows, así
# que la suite entera moria en la primera linea al correrla en la nube (Linux la pone en
# `.venv/bin/python`). Y ademas usa el MISMO interprete que lanzo esta suite, que es lo
# correcto: si alguien la ejecuta desde otro entorno, los subprocesos van a ese entorno.
PY = Path(sys.executable)
V10 = ROOT / "runs10" / "banana_v10_multifarm" / "weights" / "best.pt"
V12 = ROOT / "runs12" / "banana_v12_26m" / "weights" / "best.pt"

# (clave del protocolo, data.yaml, baseline de v10 ya registrado en real_eval/)
#
# OJO con `seenfarms`: su holdout.yaml valida en parte con extra/prueba2rgb, que resulto ser
# byte a byte el mismo dataset que extra/etiquetasnuevas, presente en el train. El 100% de sus
# 25 imagenes estaban vistas (25 de las 99 del protocolo). Ver deep/leak_audit.py y
# docs/escala-y-fugas.md. Los otros tres protocolos dan 0% de fuga.
PROTOCOLOS = [
    ("samefarm", "realdata/t768.yaml", None),
    ("seenfarms", "realdata/holdout.yaml", "v10_holdout_seenfarms"),
    ("newfarms", "realdata/holdout_newfarms.yaml", "v10_newfarms_zeroshot"),
    ("armah", "realdata/holdout_armah.yaml", "v10_armah_zeroshot"),
]


def _run(script: str, weights: Path, data: str, name: str, imgsz: int,
         batch: int) -> bool:
    """Lanza un evaluador y devuelve si terminó bien. No aborta la suite si falla."""
    cmd = [str(PY), str(ROOT / "deep" / script), "--weights", str(weights),
           "--data", str(ROOT / data), "--imgsz", str(imgsz), "--name", name]
    if script == "eval_record.py":
        cmd += ["--batch", str(batch)]
    print(f"\n=== {name} ({script}) ===", flush=True)
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(f"  FALLO ({r.returncode}): {(r.stderr or '')[-700:]}", flush=True)
        return False
    print("  ok", flush=True)
    return True


def _load(name: str) -> dict | None:
    f = ROOT / "real_eval" / f"{name}.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _fmt_delta(nuevo: float, viejo: float | None) -> str:
    if viejo is None:
        return ""
    d = nuevo - viejo
    return f"  ({viejo:.3f} -> {d:+.3f})"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=str(V12), help="pesos a evaluar")
    ap.add_argument("--tag", default="v12", help="prefijo de los JSON en real_eval/")
    ap.add_argument("--imgsz", type=int, default=768)
    ap.add_argument("--batch", type=int, default=8, help="bájalo si la GPU va justa")
    ap.add_argument("--solo", default=None,
                    help="un protocolo suelto: samefarm|seenfarms|newfarms|armah")
    ap.add_argument("--sin-conteo", action="store_true", help="solo mAP, sin conteo")
    ap.add_argument("--baseline-v10", action="store_true",
                    help="mide también el conteo de v10 (para comparar de tú a tú)")
    args = ap.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        sys.exit(f"No existe: {weights}")

    protos = [p for p in PROTOCOLOS if args.solo is None or p[0] == args.solo]
    if not protos:
        sys.exit(f"Protocolo desconocido: {args.solo}")

    fallos = 0
    for clave, data, _ in protos:
        if not (ROOT / data).exists():
            print(f"\n=== {clave}: SALTADO, falta {data} ===", flush=True)
            continue
        fallos += not _run("eval_record.py", weights, data,
                           f"{args.tag}_{clave}_map", args.imgsz, args.batch)
        if not args.sin_conteo:
            fallos += not _run("eval_count.py", weights, data,
                               f"{args.tag}_{clave}_count", args.imgsz, args.batch)
        if args.baseline_v10 and V10.exists():
            fallos += not _run("eval_count.py", V10, data,
                               f"v10_{clave}_count", args.imgsz, args.batch)

    print("\n\n########## RESUMEN ##########")
    print(f"pesos: {weights}")
    print(f"\n{'protocolo':12s} {'mAP50':>7s} {'mAP50-95':>9s} {'P':>6s} {'R':>6s}"
          f"   vs v10 (mAP50)")
    for clave, _, base_name in protos:
        rec = _load(f"{args.tag}_{clave}_map")
        if not rec or "metrics" not in rec:
            continue
        m = rec["metrics"]
        base = _load(base_name) if base_name else None
        prev = base["metrics"]["mAP50"] if base and "metrics" in base else None
        print(f"{clave:12s} {m['mAP50']:7.3f} {m['mAP50_95']:9.3f} "
              f"{m['precision']:6.3f} {m['recall']:6.3f}{_fmt_delta(m['mAP50'], prev)}")

    if not args.sin_conteo:
        # Dos columnas de error a proposito: la in-sample (umbral elegido mirando el
        # mismo conjunto) y la honesta (calibracion cruzada). La publicable es la 2a.
        print(f"\n{'protocolo':12s} {'conf*':>6s} {'err_insample':>13s} "
              f"{'err_HONESTO':>12s} {'MAPE_hon':>9s}   pred/gt (honesto)")
        for clave, _, _ in protos:
            rec = _load(f"{args.tag}_{clave}_count")
            if not rec or not rec.get("mejor"):
                continue
            b = rec["mejor"]
            h = rec.get("honesto") or {}
            nan = float("nan")
            err_h = h.get("error_conteo_total")
            mape_h = h.get("MAPE_por_imagen")
            par = (f"{h['total_pred']}/{h['total_gt']}"
                   if h.get("total_gt") else f"{b['total_pred']}/{b['total_gt']} (in-sample)")
            print(f"{clave:12s} {b['conf']:6.2f} "
                  f"{(b['error_conteo_total'] if b['error_conteo_total'] is not None else nan):13.4f} "
                  f"{(err_h if err_h is not None else nan):12.4f} "
                  f"{(mape_h if mape_h is not None else nan):9.4f}   {par}")

    if fallos:
        print(f"\n⚠  {fallos} evaluación(es) fallaron; el resumen solo cubre las que no.")
        sys.exit(1)


if __name__ == "__main__":
    main()
