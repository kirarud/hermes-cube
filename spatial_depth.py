"""spatial_depth.py — Золотая спираль, 4 слоя Z, конус сингулярности.

Превращает куб в золотую спираль с 4 смысловыми слоями глубины.
Перенос из Spatial Depth Workspace (React/Canvas → Python/NumPy).

Integration:
  SpatialDepthGenerator() — функция генерации точек для CubeEngine
  Z_COLORS — 4 слоя с цветами
  create_spiral_shape(pts) — новая форма для SHAPE_GENERATORS
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import math
import numpy as np
from numpy.typing import NDArray


# ── 4 Z-layers with semantic colors ──────────────────────────────────────

Z_LAYERS: Dict[str, Dict[str, float]] = {
    'surface':  {'z': 0.12, 'r': 0.96, 'g': 0.62, 'b': 0.04},   # #f59e0b
    'vector':   {'z': 0.37, 'r': 0.13, 'g': 0.77, 'b': 0.37},   # #22c55e
    'semantic': {'z': 0.62, 'r': 0.39, 'g': 0.40, 'b': 0.95},   # #6366f1
    'singularity': {'z': 0.87, 'r': 0.93, 'g': 0.29, 'b': 0.60}, # #ec4899
}

# Pre-computed RGB tuples for each layer
Z_COLORS: List[Tuple[int, int, int]] = [
    (245, 158, 11),    # surface — золотой
    (34, 197, 94),     # vector — зелёный
    (99, 102, 241),    # semantic — индиго
    (236, 72, 153),    # singularity — розовый
]


# ═══════════════════════════════════════════════════════════════════════════
# Shape generator: golden spiral
# ═══════════════════════════════════════════════════════════════════════════


def create_spiral_shape(pts: NDArray[np.float64]) -> NDArray[np.float64]:
    """Generate points along a golden spiral.

    Replaces cube particle positions with points on a logarithmic spiral
    expanding from Z=0 (center) to Z=1 (surface).

    Args:
        pts: Original cube grid (N, 3) — used only for count N.

    Returns:
        (N, 3) array of spiral positions, normalized to [-1, 1].
    """
    n: int = len(pts)
    t: NDArray[np.float64] = np.linspace(0, 1.0, n, dtype=np.float64)

    # Golden spiral: r = exp(-z * 1.5), angle = z * 16 * pi
    angles: NDArray[np.float64] = t * 16.0 * math.pi
    z: NDArray[np.float64] = t
    radii: NDArray[np.float64] = np.exp(-z * 1.5)

    # X, Y from spiral
    x: NDArray[np.float64] = np.cos(angles) * radii * 0.8
    y: NDArray[np.float64] = np.sin(angles) * radii * 0.8

    # Z: map from 0..1 to -1..1
    z_mapped: NDArray[np.float64] = z * 2.0 - 1.0

    return np.column_stack((x, y, z_mapped))


def create_spiral_cone_shape(pts: NDArray[np.float64]) -> NDArray[np.float64]:
    """Generate spiral points WITH singularity cone gravitational attraction.

    Near Z=0.5, points are pulled toward center (singularity).
    """
    n: int = len(pts)
    t: NDArray[np.float64] = np.linspace(0, 1.0, n, dtype=np.float64)

    angles: NDArray[np.float64] = t * 16.0 * math.pi
    z: NDArray[np.float64] = t

    # Singularity attraction factor: strongest at Z=0.5
    singularity_z: float = 0.5
    attract: NDArray[np.float64] = np.exp(
        -((z - singularity_z) ** 2) * 50.0
    )

    # Radius: spiral base * (1 - attract) — points collapse near singularity
    base_radii: NDArray[np.float64] = np.exp(-z * 1.5)
    collapse: NDArray[np.float64] = 1.0 - attract * 0.6
    radii: NDArray[np.float64] = base_radii * collapse

    x: NDArray[np.float64] = np.cos(angles) * radii * 0.8
    y: NDArray[np.float64] = np.sin(angles) * radii * 0.8

    # Z: points near singularity get pulled toward center of cube
    z_pull: NDArray[np.float64] = z - attract * 0.15 * (z - singularity_z)
    z_mapped: NDArray[np.float64] = z_pull * 2.0 - 1.0

    return np.column_stack((x, y, z_mapped))


# ═══════════════════════════════════════════════════════════════════════════
# Layer coloring
# ═══════════════════════════════════════════════════════════════════════════


def get_layer_color(pz: NDArray[np.float64]) -> NDArray[np.float64]:
    """Assign RGB color to each point based on its Z-layer.

    Args:
        pz: Z coordinates in [-1, 1] range.

    Returns:
        (N, 3) RGB uint8 array — each row = (R, G, B).
    """
    n: int = len(pz)
    colors: NDArray[np.float64] = np.zeros((n, 3), dtype=np.float64)

    # Normalize z from [-1, 1] to [0, 1]
    z_norm: NDArray[np.float64] = (pz + 1.0) / 2.0

    for i, (name, layer) in enumerate(Z_LAYERS.items()):
        layer_z: float = layer['z']
        # Points within ±0.15 of each layer center get that color
        mask: NDArray[np.bool_] = np.abs(z_norm - layer_z) < 0.15
        colors[mask, 0] = layer['r'] * 255
        colors[mask, 1] = layer['g'] * 255
        colors[mask, 2] = layer['b'] * 255

    # Fallback: uncolored points blend between nearest layers
    uncolored: NDArray[np.bool_] = np.all(colors == 0, axis=1)
    if np.any(uncolored):
        z_u: NDArray[np.float64] = z_norm[uncolored]
        # Interpolate between layer colors based on Z
        for j in range(len(z_u)):
            zj: float = z_u[j]
            # Find nearest two layers
            layer_zs: List[float] = [v['z'] for v in Z_LAYERS.values()]
            idx: int = min(range(len(layer_zs)),
                           key=lambda i: abs(layer_zs[i] - zj))
            colors[uncolored, 0] = Z_COLORS[idx][0]
            colors[uncolored, 1] = Z_COLORS[idx][1]
            colors[uncolored, 2] = Z_COLORS[idx][2]

    return np.clip(colors, 0, 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════
# Plenum / Vacuum (from Spatial Depth)
# ═══════════════════════════════════════════════════════════════════════════


def apply_plenum_vacuum(
    pts: NDArray[np.float64],
    colors: NDArray[np.float64],
    cy: float,
) -> NDArray[np.float64]:
    """Apply Plenum/Vacuum effect to points.

    Upper half (y < cy) = PLENUM — normal visibility.
    Lower half (y > cy) = VACUUM — opacity decays toward bottom.

    Args:
        pts: (N, 2) screen-space (x, y).
        colors: (N, 3) RGB uint8.
        cy: screen center Y.

    Returns:
        Modified colors with vacuum dimming.
    """
    vacuum_mask: NDArray[np.bool_] = pts[:, 1] > cy
    if not np.any(vacuum_mask):
        return colors

    # Vacuum decay: stronger toward bottom
    vacuum_strength: NDArray[np.float64] = (
        (pts[vacuum_mask, 1] - cy) / max(1.0, cy)
    )
    vacuum_strength = np.clip(vacuum_strength, 0.0, 1.0)

    colors_out: NDArray[np.float64] = colors.copy().astype(np.float64)
    fade: NDArray[np.float64] = 1.0 - vacuum_strength * 0.96
    for c in range(3):
        colors_out[vacuum_mask, c] = (
            colors_out[vacuum_mask, c] * fade
        )

    return np.clip(colors_out, 0, 255).astype(np.uint8)
