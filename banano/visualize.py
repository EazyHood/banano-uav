"""Visualizacion de resultados de deteccion."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # backend sin ventana (guardar a archivo)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from .indices import to_rgb_float


def overlay(rgb, result, out_path=None, title=None, show_mask=True):
    """Dibuja mascara de dosel, macollas (circulos) y pseudotallos (cruces)."""
    rgb = to_rgb_float(rgb)
    fig, ax = plt.subplots(figsize=(11, 11))
    ax.imshow(rgb)

    if show_mask and result.mask is not None:
        m = np.ma.masked_where(~result.mask, result.mask.astype(float))
        ax.imshow(m, alpha=0.18, cmap="cool")

    mat_r = result.params.get("mat_radius_px", 15)
    n = max(1, result.n_mats)
    colors = plt.cm.tab20(np.linspace(0, 1, n))
    for i, mat in enumerate(result.mats):
        cy, cx = mat["centroid"]
        ax.add_patch(
            Circle(
                (cx, cy),
                mat_r,
                fill=False,
                edgecolor=colors[i % len(colors)],
                lw=1.6,
            )
        )

    if result.n_pseudostems:
        ax.scatter(
            result.centers[:, 1],
            result.centers[:, 0],
            s=32,
            c="red",
            marker="+",
            linewidths=1.3,
        )

    if title is None:
        title = (
            f"Banano — {result.n_mats} macollas, "
            f"{result.n_pseudostems} pseudotallos"
        )
    ax.set_title(title, fontsize=13)
    ax.axis("off")
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
    return out_path


def save_score_map(result, out_path):
    """Guarda el mapa de respuesta de simetria radial (util para depurar)."""
    fig, ax = plt.subplots(figsize=(9, 9))
    im = ax.imshow(result.score_map, cmap="magma")
    ax.set_title("Mapa de simetria radial (FRST)")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path
