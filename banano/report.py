"""Generacion de entregables: CSV, GeoJSON, mapa e informe HTML autocontenido.

Estos son los productos que un agronomo espera: una tabla de plantas, una capa GIS
que puede abrir en QGIS/Google Earth, un mapa y un informe legible. Todo offline.
"""

from __future__ import annotations

import base64
import csv
import html
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def write_csv(path, result):
    """Tabla de macollas: id, fila, columna, lon, lat, nº de pseudotallos."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "fila_px", "columna_px", "lon", "lat", "pseudotallos"])
        for i, m in enumerate(result.mats, start=1):
            cy, cx = m["centroid"]
            w.writerow(
                [
                    i,
                    round(float(cy), 1),
                    round(float(cx), 1),
                    m.get("lon", ""),
                    m.get("lat", ""),
                    m["n_pseudostems"],
                ]
            )
    return path


def write_geojson(path, result):
    """Capa GIS de puntos (una macolla = un punto). EPSG:4326 si hay georreferencia."""
    features = []
    for i, m in enumerate(result.mats, start=1):
        cy, cx = m["centroid"]
        if result.has_geo and "lon" in m:
            coords = [m["lon"], m["lat"]]
        else:
            coords = [float(cx), float(cy)]  # pixel (x, y) si no hay geo
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": coords},
                "properties": {"id": i, "pseudotallos": m["n_pseudostems"]},
            }
        )
    fc = {"type": "FeatureCollection", "features": features}
    if result.has_geo:
        fc["crs"] = {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
        }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(fc, fh, ensure_ascii=False)
    return path


def render_map(path, overview_img, sy, sx, result, title=None):
    """Mapa: overview del ortomosaico con macollas (circulos) y pseudotallos (puntos).

    sy, sx = factores de reduccion reales por eje (fila, columna) del overview.
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(overview_img)

    mat_r = max(4, (result.params.get("mat_eps_px", 20) / (0.5 * (sy + sx))))
    for m in result.mats:
        cy, cx = m["centroid"]
        ax.add_patch(Circle((cx / sx, cy / sy), mat_r, fill=False, edgecolor="yellow", lw=0.8))
    if result.n_pseudostems:
        ax.scatter(
            result.pseudostems_px[:, 1] / sx,
            result.pseudostems_px[:, 0] / sy,
            s=6,
            c="red",
            marker=".",
        )
    ax.set_title(title or f"{result.n_mats} macollas · {result.n_pseudostems} pseudotallos")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _b64(path):
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("ascii")


def write_html(path, result, map_png=None, input_name=""):
    """Informe HTML autocontenido (imagen embebida en base64)."""
    s = result.summary()
    warn_html = ""
    if s["avisos"]:
        items = "".join(f"<li>{html.escape(a)}</li>" for a in s["avisos"])
        warn_html = f'<div class="warn"><b>Avisos</b><ul>{items}</ul></div>'

    rows = [
        ("Macollas detectadas", s["n_macollas"]),
        ("Pseudotallos (estimado)", s["pseudotallos_estimados"]),
        (
            "Pseudotallos (rango honesto)",
            f"{s['pseudotallos_rango'][0]} – {s['pseudotallos_rango'][1]}",
        ),
        ("Pseudotallos por macolla", s["pseudotallos_por_macolla"]),
        ("Cobertura de dosel", f"{s['cobertura_dosel_%']} %"),
        ("GSD (cm/píxel)", s["gsd_cm"] if s["gsd_cm"] else "no provisto"),
        ("Georreferenciado", "sí" if s["georreferenciado"] else "no"),
        ("Regularidad de siembra", s["regularidad_siembra"]),
    ]
    table = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>" for k, v in rows
    )
    img_html = ""
    if map_png and os.path.exists(map_png):
        img_html = f'<img src="data:image/png;base64,{_b64(map_png)}" alt="mapa de detección">'

    doc = f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<title>Informe de detección de banano</title>
<style>
 body{{font-family:system-ui,Segoe UI,Roboto,sans-serif;max-width:1000px;margin:2rem auto;padding:0 1rem;color:#1a1a1a}}
 h1{{color:#2e7d32}} .sub{{color:#666;margin-top:-0.6rem}}
 table{{border-collapse:collapse;width:100%;margin:1rem 0}}
 td{{border:1px solid #ddd;padding:.5rem .8rem}} td:first-child{{font-weight:600;width:45%;background:#f6f8f6}}
 img{{max-width:100%;border:1px solid #ccc;border-radius:6px}}
 .warn{{background:#fff8e1;border:1px solid #ffe082;border-radius:6px;padding:.6rem 1rem;margin:1rem 0}}
 footer{{color:#888;font-size:.85rem;margin-top:2rem;border-top:1px solid #eee;padding-top:1rem}}
</style></head><body>
<h1>🍌 Detección de cultivo de banano</h1>
<p class="sub">Archivo: {html.escape(input_name)}</p>
{warn_html}
<table>{table}</table>
{img_html}
<footer>Generado por <b>banano-drone</b> (AGPL-3.0). El conteo fiable es a nivel de
<b>macolla</b>; los pseudotallos se reportan como rango porque separar plantas dentro de
una macolla es intrínsecamente ambiguo. Para precisión comercial, entrena el modelo del
paquete <code>deep/</code>.</footer>
</body></html>"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(doc)
    return path


def write_all(outdir, result, raster=None, input_name=""):
    """Escribe todos los entregables en outdir y devuelve el dict de rutas."""
    os.makedirs(outdir, exist_ok=True)
    paths = {}
    paths["resumen"] = os.path.join(outdir, "resumen.json")
    with open(paths["resumen"], "w", encoding="utf-8") as fh:
        json.dump(result.summary(), fh, indent=2, ensure_ascii=False)

    paths["csv"] = write_csv(os.path.join(outdir, "plantas.csv"), result)
    paths["geojson"] = write_geojson(os.path.join(outdir, "plantas.geojson"), result)

    map_png = None
    if raster is not None:
        overview, sy, sx = raster.read_overview()
        map_png = render_map(os.path.join(outdir, "mapa.png"), overview, sy, sx, result)
        paths["mapa"] = map_png

    paths["html"] = write_html(
        os.path.join(outdir, "informe.html"), result, map_png=map_png, input_name=input_name
    )
    return paths
