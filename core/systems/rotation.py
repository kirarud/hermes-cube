"""systems/rotation.py — 3D-вращение частиц.

Чистая функция: sim.position × angles → повёрнутая позиция.
Поворот по трём осям с разными скоростями (как в CubeEngine).
"""

from __future__ import annotations

import math
import numpy as np
from numpy.typing import NDArray

from core.world import World


def update(world: World, dt: float) -> None:
    """Повернуть все частицы по X/Y/Z.

    Скорости берутся из meta.config['rotation_speed'].
    Аналогично CubeEngine.get_frame().
    """
    cfg = world.meta.config
    speed: float = cfg.get('rotation_speed', 0.28)
    t: float = world.meta.t

    ang_x: float = t * 0.20 * (speed / 0.28)
    ang_y: float = t * speed
    ang_z: float = t * 0.08 * (speed / 0.28)

    pos = world.sim.position
    n = world.sim.active_count
    if n == 0:
        return
    pts = pos[:n]

    rotated = _rotate_x(pts, ang_x)
    rotated = _rotate_y(rotated, ang_y)
    rotated = _rotate_z(rotated, ang_z)

    pos[:n] = rotated


def _rotate_x(points: NDArray[np.float64], angle: float) -> NDArray[np.float64]:
    c, s = math.cos(angle), math.sin(angle)
    result = points.copy()
    y = points[:, 1] * c - points[:, 2] * s
    z = points[:, 1] * s + points[:, 2] * c
    result[:, 1] = y
    result[:, 2] = z
    return result


def _rotate_y(points: NDArray[np.float64], angle: float) -> NDArray[np.float64]:
    c, s = math.cos(angle), math.sin(angle)
    result = points.copy()
    x = points[:, 0] * c + points[:, 2] * s
    z = -points[:, 0] * s + points[:, 2] * c
    result[:, 0] = x
    result[:, 2] = z
    return result


def _rotate_z(points: NDArray[np.float64], angle: float) -> NDArray[np.float64]:
    c, s = math.cos(angle), math.sin(angle)
    result = points.copy()
    x = points[:, 0] * c - points[:, 1] * s
    y = points[:, 0] * s + points[:, 1] * c
    result[:, 0] = x
    result[:, 1] = y
    return result
