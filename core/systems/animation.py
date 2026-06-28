"""systems/animation.py — Анимации частиц.

Читает: sim.morphed
Пишет:  sim.animated

Никаких copy() — animated свой отдельный буфер.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from core.world import World


def update(world: World, dt: float) -> None:
    cfg = world.meta.config
    anim_name: str = cfg.get('particle_mode', 'off')
    n = world.sim.active_count
    if n == 0:
        return

    inp = world.sim.morphed[:n]
    out = world.sim.animated[:n]

    if anim_name == 'off':
        out[:] = inp
        return

    speed: float = cfg.get('wave_speed', 1.5)
    amp: float = cfg.get('wave_amp', 0.12)
    t: float = world.meta.t

    if anim_name == 'wave':
        _apply_wave(inp, out, t, speed, amp)
    elif anim_name == 'breathe':
        _apply_breathe(inp, out, t, speed, amp)
    elif anim_name == 'orbit':
        _apply_orbit(inp, out, t, speed, amp)
    elif anim_name == 'geyser':
        _apply_geyser(inp, out, t, speed, amp)
    else:
        out[:] = inp


def _apply_wave(inp: NDArray[np.float64], out: NDArray[np.float64],
                t: float, speed: float, amp: float) -> None:
    if amp < 0.001:
        out[:] = inp
        return
    out[:] = inp
    out[:, 0] += (np.sin(inp[:, 1] * 3.0 + t * speed * 2.5) * amp * 0.5
                  + np.cos(inp[:, 2] * 2.0 + t * speed * 1.3) * amp * 0.3)
    out[:, 1] += (np.cos(inp[:, 0] * 2.5 + t * speed * 1.7) * amp * 0.7
                  + np.sin(inp[:, 0] * 3.0 + t * speed * 1.1) * amp * 0.4)
    out[:, 2] += (np.sin(inp[:, 2] * 3.2 + t * speed * 2.0) * amp * 0.5
                  + np.cos(inp[:, 1] * 2.8 + t * speed * 1.9) * amp * 0.3)


def _apply_breathe(inp: NDArray[np.float64], out: NDArray[np.float64],
                   t: float, speed: float, amp: float) -> None:
    if amp < 0.001:
        out[:] = inp
        return
    phase = inp[:, 0] * 1.7 + inp[:, 1] * 2.3 + inp[:, 2] * 1.1
    out[:] = inp
    out[:, 0] += np.sin(phase + t * speed * 1.5) * amp
    out[:, 1] += np.cos(phase * 1.3 + t * speed * 1.1) * amp
    out[:, 2] += np.sin(phase * 0.7 + t * speed * 1.8) * amp


def _apply_orbit(inp: NDArray[np.float64], out: NDArray[np.float64],
                 t: float, speed: float, amp: float) -> None:
    if amp < 0.001:
        out[:] = inp
        return
    phase = inp[:, 0] * 2.7 + inp[:, 1] * 3.1 + inp[:, 2] * 1.9
    out[:] = inp
    out[:, 0] += (np.cos(phase + t * speed) * amp
                  + np.sin(phase * 0.5 + t * speed * 0.9) * amp * 0.4)
    out[:, 1] += np.sin(phase * 1.3 + t * speed * 0.7) * amp
    out[:, 2] += (np.cos(phase * 0.7 + t * speed * 1.4) * amp
                  + np.cos(phase * 0.9 + t * speed * 1.1) * amp * 0.4)


def _apply_geyser(inp: NDArray[np.float64], out: NDArray[np.float64],
                  t: float, speed: float, amp: float) -> None:
    if amp < 0.001:
        out[:] = inp
        return
    height = (inp[:, 1] + 1.0) * 0.5
    spray = np.sin(t * speed * 2.5 + inp[:, 0] * 4.0 + inp[:, 2] * 4.0)
    spread = spray * amp * (0.3 + height * 0.7)
    out[:] = inp
    out[:, 0] += spread
    out[:, 2] += spread
    wobble = np.sin(t * speed * 3.0 + inp[:, 0] * 5.0 + inp[:, 2] * 5.0)
    out[:, 1] += wobble * amp * 0.25 * height
