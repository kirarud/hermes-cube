"""systems/rotation.py — 3D-вращение частиц.

Читает: sim.animated
Пишет:  sim.world_position

Без копирования — world_position отдельный буфер.
Поворот по трём осям с разными скоростями (как CubeEngine).
"""

from __future__ import annotations

import math
import numpy as np
from numpy.typing import NDArray

from core.world import World


def update(world: World, dt: float) -> None:
    cfg = world.meta.config
    speed: float = cfg.get('rotation_speed', 0.28)
    t: float = world.meta.t

    ang_x: float = t * 0.20 * (speed / 0.28)
    ang_y: float = t * speed
    ang_z: float = t * 0.08 * (speed / 0.28)

    n = world.sim.active_count
    if n == 0:
        return

    inp = world.sim.animated[:n]
    out = world.sim.world_position[:n]

    # One-shot copy + rotate in-place on out
    out[:] = inp
    _rotate_x_inplace(out, ang_x)
    _rotate_y_inplace(out, ang_y)
    _rotate_z_inplace(out, ang_z)


def _rotate_x_inplace(pts: NDArray[np.float64], angle: float) -> None:
    c, s = math.cos(angle), math.sin(angle)
    y = pts[:, 1] * c - pts[:, 2] * s
    z = pts[:, 1] * s + pts[:, 2] * c
    pts[:, 1] = y
    pts[:, 2] = z


def _rotate_y_inplace(pts: NDArray[np.float64], angle: float) -> None:
    c, s = math.cos(angle), math.sin(angle)
    x = pts[:, 0] * c + pts[:, 2] * s
    z = -pts[:, 0] * s + pts[:, 2] * c
    pts[:, 0] = x
    pts[:, 2] = z


def _rotate_z_inplace(pts: NDArray[np.float64], angle: float) -> None:
    c, s = math.cos(angle), math.sin(angle)
    x = pts[:, 0] * c - pts[:, 1] * s
    y = pts[:, 0] * s + pts[:, 1] * c
    pts[:, 0] = x
    pts[:, 1] = y
