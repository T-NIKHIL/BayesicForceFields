"""Supervised VAE and independent, comparable W&B grid sweeps per data fold."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from umap import UMAP
import plotly.graph_objects as go

def _plot_umap(
    vectors,
    gnll,
    umap_params,
    output_path='.',
    color_label="Total GNLL",
    show=False,
):
    """Fit UMAP, save an interactive HTML plot, and return the coordinates."""
    vectors = np.asarray(vectors)
    gnll = np.asarray(gnll).reshape(-1)

    if vectors.ndim != 2 or len(vectors) != len(gnll):
        raise ValueError("Expected a 2D array and one GNLL value per row.")
    if not (np.isfinite(vectors).all() and np.isfinite(gnll).all()):
        raise ValueError("Vectors and GNLL must contain finite values.")

    n_components = umap_params["n_components"]
    if n_components not in (2, 3):
        raise ValueError("The plot supports 2 or 3 UMAP components.")

    coords = UMAP(**umap_params).fit_transform(vectors)

    points = {
        "x": coords[:, 0],
        "y": coords[:, 1],
        "mode": "markers",
        "marker": {
            "size": 3,
            "color": gnll,
            "colorscale": "Viridis",
            "showscale": True,
            "colorbar": {"title": color_label},
        },
    }

    if n_components == 3:
        fig = go.Figure(go.Scatter3d(z=coords[:, 2], **points))
        fig.update_layout(scene={
            "xaxis_title": "UMAP 1",
            "yaxis_title": "UMAP 2",
            "zaxis_title": "UMAP 3",
        })
    else:
        fig = go.Figure(go.Scattergl(**points))
        fig.update_layout(xaxis_title="UMAP 1", yaxis_title="UMAP 2")

    settings = [f"{key}={value}" for key, value in umap_params.items()]
    settings_text = "<br>".join(
        " · ".join(settings[i:i + 4])
        for i in range(0, len(settings), 4)
    )

    fig.update_layout(
        title={
            "text": f"UMAP colored by {color_label}",
            "subtitle": {"text": settings_text},
            "x": 0.5,
            "xanchor": "center",
        },
        margin={"t": 100},
    )

    fig.write_html(output_path, include_plotlyjs=True)
    if show:
        fig.show()

    return coords
