"""systems/trail.py — Шлейф частиц (трейлы).

Читает:
  render.projected_x/y — текущие позиции
  render.final_rgb — текущие цвета
  meta.config — cell_size для трейлов

Пишет:
  render.trail_history — кольцевой буфер (maxlen=12) позиций+цветов
  render.trail_layer — flat-массивы для штамповки
"""

from __future__ import annotations

from collections import deque
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from core.world import World


def update(world: World, dt: float) -> None:
    """Обновить трейлы: добавить текущий кадр, собрать затухающий слой."""
    if not world.render.trail_enabled:
        world.render.trail_history.clear()
        world.render.trail_layer = None
        return

    n = world.sim.active_count
    if n == 0:
        return

    # Push current frame
    world.render.trail_history.append({
        'px': world.render.projected_x[:n].copy(),
        'py': world.render.projected_y[:n].copy(),
        'rgb': world.render.final_rgb[:n].copy(),
    })

    # Build trail layer (aged, faded)
    trail_len = len(world.render.trail_history)
    if trail_len < 2:
        world.render.trail_layer = None
        return

    chunks_x: list[NDArray[np.float64]] = []
    chunks_y: list[NDArray[np.float64]] = []
    chunks_rgb: list[NDArray[np.uint8]] = []
    maxlen = world.render.trail_history.maxlen or 12

    for age in range(1, trail_len):
        frame_data = world.render.trail_history[trail_len - 1 - age]
        fade: float = 1.0 - (age / maxlen)
        if fade < 0.05:
            continue
        fc = len(frame_data['px'])
        if fc == 0:
            continue
        col = np.clip(frame_data['rgb'] * fade, 0, 255).astype(np.uint8)
        chunks_x.append(frame_data['px'])
        chunks_y.append(frame_data['py'])
        chunks_rgb.append(col)

    if chunks_x:
        world.render.trail_layer = (
            np.concatenate(chunks_x),
            np.concatenate(chunks_y),
            np.concatenate(chunks_rgb),
        )
    else:
        world.render.trail_layer = None
