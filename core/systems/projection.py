"""systems/projection.py — 3D → 2D проекция.

Читает: sim.world_position
Пишет:  render.projected_x, render.projected_y, render.depth
"""

from __future__ import annotations

import math
import numpy as np

from core.world import World


def update(world: World, dt: float) -> None:
    cfg = world.meta.config
    n = world.sim.active_count
    if n == 0:
        world.render.projected_x = np.array([], dtype=np.float64)
        world.render.projected_y = np.array([], dtype=np.float64)
        world.render.depth = np.array([], dtype=np.float64)
        return

    pts = world.sim.world_position[:n]
    w = world.meta.w
    h = world.meta.h
    t = world.meta.t

    pulse_rate = cfg.get('pulse_rate', 1.8)
    pulse_amp = cfg.get('pulse_amplitude', 0.12)
    pulse = 1.0 + pulse_amp * math.sin(t * pulse_rate)

    scale_val = cfg.get('cube_scale', 0.27)
    base = float(min(w, h))
    scale = base * scale_val / (1.0 + pulse_amp) * pulse

    # ── Защита от «невидимой рамки» ──────────────────────────────────────
    # cube_scale — ДОЛЯ окна, поэтому при большом значении куб выходит за
    # границы кадра независимо от размера окна (FBO == размер окна, всё
    # снаружи физически отсутствует). Ограничиваем масштаб так, чтобы самая
    # далёкая частица оставалась внутри с запасом под размер ячейки.
    if cfg.get('fit_to_window', True):
        margin = float(max(2, int(cfg.get('cell_size', 6)))) + 4.0
        half = min(w, h) / 2.0 - margin
        if half > 1.0:
            max_abs = float(max(
                np.abs(pts[:, 0]).max(),
                np.abs(pts[:, 1]).max(),
            ))
            if max_abs > 1e-6:
                # Учитываем пик пульсации, чтобы не обрезало на вдохе
                peak = scale / pulse * (1.0 + pulse_amp)
                if peak * max_abs > half:
                    scale *= half / (peak * max_abs)

    cx_s = w / 2.0 + world.meta.cube_ox
    cy_s = h / 2.0 + world.meta.cube_oy

    world.render.projected_x = pts[:, 0] * scale + cx_s
    world.render.projected_y = pts[:, 1] * scale + cy_s
    world.render.depth = pts[:, 2]
