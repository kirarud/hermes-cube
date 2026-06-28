"""systems/color.py — Цвет частиц.

Читает: sim.color (базовые цвета), sim.world_position (для глубины)
Пишет:  render.final_rgb
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from core.world import World


def update(world: World, dt: float) -> None:
    n = world.sim.active_count
    if n == 0:
        world.render.final_rgb = np.array([], dtype=np.uint8).reshape(0, 3)
        return

    r0 = world.sim.color[:n, 0]
    g0 = world.sim.color[:n, 1]
    b0 = world.sim.color[:n, 2]
    pz = world.sim.world_position[:n, 2]

    depth_factor = 0.6 + 0.4 * (pz + 1.0) / 2.0

    r_p = np.clip(r0 * depth_factor, 0, 255)
    g_p = np.clip(g0 * depth_factor, 0, 255)
    b_p = np.clip(b0 * depth_factor, 0, 255)

    shift = world.meta.color_shift
    if shift > 0.01:
        r_p, g_p, b_p = _apply_hsv_shift(r_p, g_p, b_p, shift)

    world.render.final_rgb = np.column_stack(
        (r_p, g_p, b_p),
    ).astype(np.uint8)


def _apply_hsv_shift(r, g, b, shift):
    rn, gn, bn = r / 255.0, g / 255.0, b / 255.0
    mx = np.maximum(np.maximum(rn, gn), bn)
    mn = np.minimum(np.minimum(rn, gn), bn)
    delta = mx - mn

    h = np.zeros_like(rn)
    mask = delta > 1e-6
    rm = mask & (mx == rn)
    gm = mask & (mx == gn)
    bm = mask & (mx == bn)
    h[rm] = ((gn[rm] - bn[rm]) / delta[rm]) % 6.0
    h[gm] = ((bn[gm] - rn[gm]) / delta[gm]) + 2.0
    h[bm] = ((rn[bm] - gn[bm]) / delta[bm]) + 4.0
    h = h / 6.0

    s = np.zeros_like(rn)
    s[mask] = delta[mask] / mx[mask]
    v = mx
    h = (h + shift) % 1.0

    h6 = h * 6.0
    hi = np.floor(h6).astype(np.int32)
    f = h6 - hi.astype(np.float64)
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    r_out = np.clip(np.select(
        [hi == 0, hi == 1, hi == 2, hi == 3, hi == 4, hi == 5],
        [v, q, p, p, t, v]) * 255.0, 0, 255)
    g_out = np.clip(np.select(
        [hi == 0, hi == 1, hi == 2, hi == 3, hi == 4, hi == 5],
        [t, v, v, q, p, p]) * 255.0, 0, 255)
    b_out = np.clip(np.select(
        [hi == 0, hi == 1, hi == 2, hi == 3, hi == 4, hi == 5],
        [p, p, t, v, v, q]) * 255.0, 0, 255)
    return r_out, g_out, b_out
