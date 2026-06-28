"""systems/grid_generator.py — Генерация сетки частиц куба.

Пишет в sim.base_position — исходная сетка на 6 гранях [-1, 1].
Копирует во все downstream буферы (morphed, animated, world_position).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

from core.world import World


def update(world: World, dt: float) -> None:
    """Перестроить сетку если density изменился."""
    cfg = world.meta.config
    new_density: int = int(cfg.get('particle_density', 12))
    expected_n = new_density * new_density * 6

    if world.sim.active_count == expected_n and world.sim.shape_cache:
        return

    _rebuild(world, new_density)


def _rebuild(world: World, density: int) -> None:
    """Сгенерировать новую сетку, цвета, shape_cache, копировать во все буферы."""
    pts = _generate_cube_grid(density)
    n = len(pts)

    world.sim.active_count = n
    world.sim.base_position[:n] = pts
    world.sim.morphed[:n] = pts
    world.sim.animated[:n] = pts
    world.sim.world_position[:n] = pts
    world.sim.color[:n, 0] = ((pts[:, 0] + 1.0) / 2.0 * 255.0)
    world.sim.color[:n, 1] = ((pts[:, 1] + 1.0) / 2.0 * 255.0)
    world.sim.color[:n, 2] = ((pts[:, 2] + 1.0) / 2.0 * 255.0)

    world.sim.alive[:] = False
    world.sim.alive[:n] = True

    world.sim.shape_cache = {
        'cube': pts.copy(),
        'sphere': _gen_sphere(pts),
        'torus': _gen_torus(pts),
        'dna': _gen_dna(pts),
        'metaball': _gen_metaball(pts),
    }


def _generate_cube_grid(n: int) -> NDArray[np.float64]:
    pts: List[Tuple[float, float, float]] = []
    u = np.linspace(-1.0, 1.0, n)
    v = np.linspace(-1.0, 1.0, n)
    for ui in u:
        for vi in v:
            pts.append((1.0, ui, vi))
            pts.append((-1.0, vi, ui))
            pts.append((ui, 1.0, vi))
            pts.append((ui, -1.0, vi))
            pts.append((ui, vi, 1.0))
            pts.append((ui, vi, -1.0))
    return np.array(pts, dtype=np.float64)


def _gen_sphere(points: NDArray[np.float64]) -> NDArray[np.float64]:
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / np.clip(norms, 1e-8, None)


def _gen_torus(points: NDArray[np.float64]) -> NDArray[np.float64]:
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    R, r = 1.5, 0.5
    theta = np.arctan2(z, x)
    phi = np.arcsin(np.clip(y, -1, 1)) * 2.0
    result = np.zeros_like(points)
    result[:, 0] = (R + r * np.cos(phi)) * np.cos(theta)
    result[:, 1] = r * np.sin(phi)
    result[:, 2] = (R + r * np.cos(phi)) * np.sin(theta)
    return result


def _gen_dna(points: NDArray[np.float64]) -> NDArray[np.float64]:
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    twists = 3.0
    angle = z * twists * np.pi
    radius = 0.7 + 0.3 * np.abs(np.sin(angle * 0.5))
    result = np.zeros_like(points)
    result[:, 0] = radius * np.cos(angle)
    result[:, 1] = y * 0.5
    result[:, 2] = radius * np.sin(angle)
    return result


def _gen_metaball(points: NDArray[np.float64]) -> NDArray[np.float64]:
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    attractors = [
        (0.5, 0.5, 0.5, 0.6), (-0.5, -0.5, 0.5, 0.6),
        (0.5, -0.5, -0.5, 0.6), (-0.5, 0.5, -0.5, 0.6),
    ]
    field = np.zeros(len(points))
    for ax, ay, az, strength in attractors:
        dist = np.sqrt((x - ax)**2 + (y - ay)**2 + (z - az)**2)
        field += strength / (dist + 0.1)
    field /= field.max()
    scale = 0.6 + 0.4 * field
    result = points.copy()
    result[:, 0] *= scale
    result[:, 1] *= scale
    result[:, 2] *= scale
    return result
