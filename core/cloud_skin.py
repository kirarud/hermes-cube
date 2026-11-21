"""Deterministic volumetric cloud-avatar skin for Hermes Cube."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def generate_cloud(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Spread the existing indexed particles through five overlapping cloud lobes.

    Pure, stable, no per-frame randomness. The shape can morph to/from cube.
    """
    n = len(points)
    if n == 0:
        return points.copy()

    centers = np.array([
        [-0.73, -0.17, -0.05],
        [-0.39,  0.13,  0.03],
        [ 0.00,  0.30,  0.04],
        [ 0.42,  0.13,  0.03],
        [ 0.77, -0.17, -0.05],
    ], dtype=np.float64)
    radii = np.array([
        [0.46, 0.37, 0.39],
        [0.54, 0.52, 0.46],
        [0.59, 0.56, 0.47],
        [0.54, 0.49, 0.44],
        [0.45, 0.38, 0.39],
    ], dtype=np.float64)

    # Low-discrepancy pseudo-random positions create full volume rather than
    # wireframe shells.  Index (not time) controls the positions.
    idx = np.arange(n, dtype=np.float64) + 0.5
    longitude = 2.0 * np.pi * np.mod(idx * 0.618033988749895, 1.0)
    latitude = 2.0 * np.mod(idx * 0.754877666246693, 1.0) - 1.0
    radial = np.cbrt(np.mod(idx * 0.569840290998, 1.0))
    horizontal = np.sqrt(np.maximum(0.0, 1.0 - latitude * latitude))
    directions = np.column_stack((
        horizontal * np.cos(longitude),
        latitude,
        horizontal * np.sin(longitude),
    ))
    lobe = np.arange(n) % len(centers)
    result = centers[lobe] + directions * (radii[lobe] * radial[:, None])
    return result
