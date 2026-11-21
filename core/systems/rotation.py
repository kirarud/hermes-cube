"""systems/rotation.py — 3D-вращение частиц.

Читает: sim.animated, meta.text_freeze
Пишет:  sim.world_position

Без копирования — world_position отдельный буфер.
Поворот по трём осям с разными скоростями (как CubeEngine).

text_freeze (0..1) плавно возвращает частицы к неповёрнутому виду: при 1.0
надпись развёрнута прямо к камере и не крутится. Это lerp РЕЗУЛЬТАТА, а не
обнуление угла — угол = t * speed, поэтому менять speed на ходу нельзя
(получается скачок на t*Δspeed радиан).
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

    n = world.sim.active_count
    if n == 0:
        return

    inp = world.sim.animated[:n]
    out = world.sim.world_position[:n]

    freeze = float(getattr(world.meta, 'text_freeze', 0.0))
    freeze = min(1.0, max(0.0, freeze))

    if freeze >= 0.999:
        # Полностью «в лоб» к камере — вращения нет
        out[:] = inp
        return

    # A face should look at the user, not spin away every few seconds.
    if cfg.get('shape_preset') == 'face' and float(cfg.get('morph_progress',0)) >= .99:
        ang_x = 0.035 * math.sin(t * 0.45)
        ang_y = 0.15 * math.sin(t * 0.28)
        ang_z = 0.025 * math.sin(t * 0.37)
    else:
        ang_x = t * 0.20 * (speed / 0.28)
        ang_y = t * speed
        ang_z = t * 0.08 * (speed / 0.28)

    # One-shot copy + rotate in-place on out
    out[:] = inp
    _rotate_x_inplace(out, ang_x)
    _rotate_y_inplace(out, ang_y)
    _rotate_z_inplace(out, ang_z)

    if freeze > 0.001:
        # Плавно «доворачиваем» к неповёрнутой позиции
        out *= (1.0 - freeze)
        out += inp * freeze


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
