"""systems/projection.py — Проекция 3D → 2D экранные координаты.

Читает:
  sim.position — 3D-позиции (уже повёрнутые, с анимацией)
  meta.config — cube_scale, pulse_rate, pulse_amplitude
  meta.t — время для пульсации
  meta.cube_ox/oy — drag offset

Пишет:
  render.projected_x/y — (N,) экранные координаты
  render.depth — (N,) z-глубина для painter's algorithm
"""

from __future__ import annotations

import math
import numpy as np

from core.world import World


def update(world: World, dt: float) -> None:
    """Спроецировать 3D-позиции на экран."""
    cfg = world.meta.config
    n = world.sim.active_count
    if n == 0:
        world.render.projected_x = np.array([], dtype=np.float64)
        world.render.projected_y = np.array([], dtype=np.float64)
        world.render.depth = np.array([], dtype=np.float64)
        return

    pts = world.sim.position[:n]
    w = world.meta.w
    h = world.meta.h
    t = world.meta.t

    # Pulse
    pulse_rate: float = cfg.get('pulse_rate', 1.8)
    pulse_amp: float = cfg.get('pulse_amplitude', 0.12)
    pulse: float = 1.0 + pulse_amp * math.sin(t * pulse_rate)

    # Scale
    scale_val: float = cfg.get('cube_scale', 0.27)
    base: float = float(min(w, h))
    scale: float = (base * scale_val / (1.0 + pulse_amp) * pulse)

    cx_s: float = w / 2.0 + world.meta.cube_ox
    cy_s: float = h / 2.0 + world.meta.cube_oy

    world.render.projected_x = pts[:, 0] * scale + cx_s
    world.render.projected_y = pts[:, 1] * scale + cy_s
    world.render.depth = pts[:, 2]
