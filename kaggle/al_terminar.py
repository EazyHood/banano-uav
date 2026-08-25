"""Vigila la corrida de Kaggle y, en cuanto acabe, recoge los pesos y los mide.

Por qué existe: una corrida de 10 h termina de madrugada. Kaggle manda un correo, pero eso
sólo dice "ha terminado", no si el modelo sirve. Esto deja el veredicto escrito y listo:

    C:\\Users\\jhona\\Desktop\\BANANO-RESULTADO.md

Pensado para una tarea programada cada 30 min. Es idempotente: si la corrida sigue en marcha
no hace nada, y si ya escribió el veredicto tampoco lo repite.

    python kaggle/al_terminar.py            # una pasada
    python kaggle/al_terminar.py --instalar # lo deja programado cada 30 min
    python kaggle/al_terminar.py --quitar   # lo desprograma

LA PREGUNTA QUE CONTESTA. El modelo publicado (v10) mide esto en `armah`, la finca que
ningún modelo ha visto, a la resolución que mejor le va (1024 px):

    mAP50 0.2847   recall 0.2290

Si la tirada nueva no pasa de ahí, no se publica. Lo decide el número, no la impresión de
que "ha entrenado mucho".
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
ROOT = AQUI.parent
VEREDICTO = Path.home() / "Desktop" / "BANANO-RESULTADO.md"
TAREA = "banano-uav-al-terminar"

# v10 sobre armah a 1024 px (real_eval/scale_sweep_lofo_v10.json). Es la vara.
VARA = {"mAP50": 0.2847, "recall": 0.2290}


def kaggle_cli() -> str:
    cand = Path(sys.executable).parent / "kaggle.exe"
    return str(cand) if cand.exists() else "kaggle"


def estado(kid: str) -> str:
    r = subprocess.run([kaggle_cli(), "kernels", "status", kid],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    for marca in ("COMPLETE", "ERROR", "CANCEL", "RUNNING", "QUEUED"):
        if marca in (r.stdout + r.stderr).upper():
            return marca
    return "DESCONOCIDO"


def avisa(titulo: str, cuerpo: str) -> None:
    """Notificación en el área de notificación de Windows.

    Hace falta porque el fichero del Escritorio no se ve solo: si nadie lo abre, el
    resultado está pero nadie se entera. Se usa NotifyIcon de WinForms, que viene con
    Windows y no exige instalar nada (BurntToast habría que instalarlo). Si falla —otro
    sistema operativo, sesión no interactiva— no pasa nada: el fichero sigue escrito.
    """
    if os.name != "nt":
        return
    ps = (
        "[void][System.Reflection.Assembly]::LoadWithPartialName('System.Windows.Forms');"
        "$n=New-Object System.Windows.Forms.NotifyIcon;"
        "$n.Icon=[System.Drawing.SystemIcons]::Information;"
        f"$n.BalloonTipTitle='{titulo}';"
        f"$n.BalloonTipText='{cuerpo}';"
        "$n.Visible=$true;$n.ShowBalloonTip(60000);Start-Sleep -Seconds 12;$n.Dispose()"
    )
    with contextlib.suppress(Exception):
        subprocess.run(["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", ps],
                       capture_output=True, timeout=40)


def escribe(texto: str, titulo: str = "Banano: la corrida ha terminado") -> None:
    VEREDICTO.parent.mkdir(parents=True, exist_ok=True)
    VEREDICTO.write_text(texto, encoding="utf-8")
    print(f"veredicto -> {VEREDICTO}")
    # la primera línea con contenido resume el veredicto
    resumen = next((ln.strip("# *") for ln in texto.splitlines()[2:] if ln.strip()), "listo")
    avisa(titulo, f"{resumen[:120]} -- detalle en el fichero BANANO-RESULTADO.md del Escritorio")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kernel", default=None, help="por defecto, el del kernel-metadata.json")
    ap.add_argument("--instalar", action="store_true")
    ap.add_argument("--quitar", action="store_true")
    args = ap.parse_args()

    kid = args.kernel or json.loads(
        (AQUI / "entrenar" / "kernel-metadata.json").read_text(encoding="utf-8")
    )["id"]

    if args.quitar:
        r = subprocess.run(["schtasks", "/Delete", "/TN", TAREA, "/F"],
                           capture_output=True, text=True)
        print((r.stdout + r.stderr).strip())
        return r.returncode

    if args.instalar:
        cmd = f'"{sys.executable}" "{Path(__file__).resolve()}"'
        r = subprocess.run(
            ["schtasks", "/Create", "/TN", TAREA, "/TR", cmd,
             "/SC", "MINUTE", "/MO", "30", "/F"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        print((r.stdout + r.stderr).strip())
        if r.returncode == 0:
            print(f"Comprobará cada 30 min y escribirá el veredicto en {VEREDICTO}")
        return r.returncode

    st = estado(kid)
    print(f"{kid}: {st}")
    if st in ("RUNNING", "QUEUED"):
        return 0
    if st == "DESCONOCIDO":
        return 1

    if st in ("ERROR", "CANCEL"):
        # Una cancelación casi siempre es la cuota agotada, no un fallo del código. Decirlo
        # aquí importa: el 25-ago este mismo aviso salió como "no llegó a buen puerto" y
        # mandaba a leer el log — y el log no dice ni una palabra de cuota. La causa está a
        # un comando de distancia, así que se consulta y se escribe.
        sys.path.insert(0, str(AQUI))
        try:
            from cuota import caben_horas
            from cuota import lee as lee_cuota

            q = lee_cuota()
        except Exception:
            q = None

        if q and q[1] <= 0.01:
            causa = (
                f"**Se agotó la cuota de GPU** ({q[0]:.2f} h usadas de 30). No falló el "
                "código: Kaggle corta la corrida cuando la cuota llega a cero. Se gasta "
                "~1,7x más rápido que el reloj porque asigna dos T4 y cobra por las dos, "
                "así que las 30 h semanales son ~17 h de reloj. **Vuelve el sábado a "
                "medianoche UTC** (19:00 del viernes en Colombia)."
            )
            orden = "kaggle\\cuota.py     # y cuando haya cuota:  kaggle\\lanzar.py"
        elif q:
            causa = (
                f"Cuota disponible: {q[1]:.2f} h ({caben_horas(q[1]):.1f} h de reloj), así "
                "que **no fue la cuota**. Hay que mirar el log."
            )
            orden = "kaggle\\lanzar.py --log"
        else:
            causa = "No pude leer la cuota para descartar que fuera eso."
            orden = "kaggle\\cuota.py     # y luego:  kaggle\\lanzar.py --log"

        escribe(
            f"# Banano: la corrida terminó con {st}\n\n{causa}\n\n"
            f"```\ncd {ROOT}\n.venv\\Scripts\\python.exe {orden}\n```\n",
            titulo="Banano: la corrida se cortó",
        )
        return 0

    # COMPLETE: recoger y medir
    destino = ROOT / "runs_cloud" / "kaggle_lofo"
    subprocess.run([sys.executable, str(AQUI / "lanzar.py"), "--recoger", "--destino", str(destino)],
                   cwd=str(ROOT))

    pesos = sorted(destino.rglob("*best*.pt"))
    if not pesos:
        escribe(
            "# Banano: la corrida terminó, pero no se recogieron pesos\n\n"
            "Suele pasar si hay una pestaña del notebook abierta en el navegador: la API de\n"
            "Kaggle mira la SESIÓN, no la versión guardada, y contesta cero ficheros.\n\n"
            f"Ciérrala y reintenta:\n\n```\ncd {ROOT}\n"
            ".venv\\Scripts\\python.exe kaggle\\lanzar.py --recoger\n```\n"
        )
        return 0

    modelo = max(pesos, key=lambda p: p.stat().st_size)
    data = ROOT / "splits" / "lofo_armah.yaml"
    salida = ROOT / "real_eval" / "lofo_armah_modelo_nuevo.json"
    subprocess.run(
        [sys.executable, str(ROOT / "cloud" / "scale_sweep.py"),
         "--pesos", str(modelo), "--data", str(data),
         "--imgsz", "768", "1024", "1280", "--salida", str(salida)],
        cwd=str(ROOT),
    )

    if not salida.exists():
        escribe(f"# Banano: pesos recogidos en `{destino}`, pero la medición falló\n")
        return 0

    d = json.loads(salida.read_text(encoding="utf-8"))
    filas = next(iter(d["fincas"].values()))["barrido"]
    mejor = max(filas, key=lambda f: f["mAP50"])
    gana = mejor["mAP50"] > VARA["mAP50"] and mejor["recall"] > VARA["recall"]

    tabla = "\n".join(
        f"| {f['imgsz']} | {f['mAP50']:.4f} | {f['recall']:.4f} | {f['precision']:.4f} |"
        for f in filas
    )
    escribe(f"""# Banano: resultado de la corrida en la nube

**{'✅ EL MODELO NUEVO GANA' if gana else '❌ NO SUPERA AL QUE YA TIENES'}**

Medido sobre `armah`, la finca que ningún modelo vio, con el modelo entrenado dejándola
entera fuera. Es la comparación honesta.

| | mAP50 | recall |
|---|---:|---:|
| v10 (el publicado, a 1024 px) | {VARA['mAP50']:.4f} | {VARA['recall']:.4f} |
| **modelo nuevo** (a {mejor['imgsz']} px) | **{mejor['mAP50']:.4f}** | **{mejor['recall']:.4f}** |

Barrido completo del modelo nuevo:

| imgsz | mAP50 | recall | precisión |
|---:|---:|---:|---:|
{tabla}

Pesos: `{modelo}`
Medición: `{salida.relative_to(ROOT)}`

{'Toca decidir si se publica: el número lo respalda.' if gana else
 'No se publica. Que haya entrenado muchas horas no es un argumento; la cifra en finca nueva es la que manda.'}

> Recuerda regenerar el token de Kaggle: https://www.kaggle.com/settings/api
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
