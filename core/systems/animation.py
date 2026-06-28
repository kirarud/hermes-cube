"""systems/animation.py — Анимации частиц (смещения от базовой позиции).

5 режимов: off, wave, breathe, orbit, geyser.
Применяется ПОСЛЕ morph, ДО rotation.
Каждая функция — numpy-векторизованное смещение positions × t × params.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from core.world import World


def update(world: World, dt: float) -> None:
    """Применить анимацию частиц к sim.position."""
    cfg = world.meta.config
    anim_name: str = cfg.get('particle_mode', 'off')
    if anim_name == 'off':
        return

    speed: float = cfg.get('wave_speed', 1.5)
    amp: float = cfg.get('wave_amp', 0.12)
    t: float = world.meta.t

    n = world.sim.active_count
    if n == 0:
        return

    pts = world.sim.position[:n]

    if anim_name == 'wave':
        world.sim.position[:n] = _animate_wave(pts, t, speed, amp)
    elif anim_name == 'breathe':
        world.sim.position[:n] = _animate_breathe(pts, t, speed, amp)
    elif anim_name == 'orbit':
        world.sim.position[:n] = _animate_orbit(pts, t, speed, amp)
    elif anim_name == 'geyser':
        world.sim.position[:n] = _animate_geyser(pts, t, speed, amp)


def _animate_wave(points: NDArray[np.float64], t: float,
                  speed: float, amp: float) -> NDArray[np.float64]:
    """3D Lissajous wave field."""
    if amp < 0.001:
        return points.copy()
    result = points.copy()
    result[:, 0] += (np.sin(points[:, 1] * 3.0 + t * speed * 2.5) * amp * 0.5
                     + np.cos(points[:, 2] * 2.0 + t * speed * 1.3) * amp * 0.3)
    result[:, 1] += (np.cos(points[:, 0] * 2.5 + t * speed * 1.7) * amp * 0.7
                     + np.sin(points[:, 0] * 3.0 + t * speed * 1.1) * amp * 0.4)
    result[:, 2] += (np.sin(points[:, 2] * 3.2 + t * speed * 2.0) * amp * 0.5
                     + np.cos(points[:, 1] * 2.8 + t * speed * 1.9) * amp * 0.3)
    return result


def _animate_breathe(points: NDArray[np.float64], t: float,
                     speed: float, amp: float) -> NDArray[np.float64]:
    if amp < 0.001:
        return points.copy()
    phase = points[:, 0] * 1.7 + points[:, 1] * 2.3 + points[:, 2] * 1.1
    result = points.copy()
    result[:, 0] += np.sin(phase + t * speed * 1.5) * amp
    result[:, 1] += np.cos(phase * 1.3 + t * speed * 1.1) * amp
    result[:, 2] += np.sin(phase * 0.7 + t * speed * 1.8) * amp
    return result


def _animate_orbit(points: NDArray[np.float64], t: float,
                   speed: float, amp: float) -> NDArray[np.float64]:
    if amp < 0.001:
        return points.copy()
    phase = points[:, 0] * 2.7 + points[:, 1] * 3.1 + points[:, 2] * 1.9
    result = points.copy()
    result[:, 0] += (np.cos(phase + t * speed) * amp
                     + np.sin(phase * 0.5 + t * speed * 0.9) * amp * 0.4)
    result[:, 1] += np.sin(phase * 1.3 + t * speed * 0.7) * amp
    result[:, 2] += (np.cos(phase * 0.7 + t * speed * 1.4) * amp
                     + np.cos(phase * 0.9 + t * speed * 1.1) * amp * 0.4)
    return result


def _animate_geyser(points: NDArray[np.float64], t: float,
                    speed: float, amp: float) -> NDArray[np.float64]:
    if amp < 0.001:
        return points.copy()
    height = (points[:, 1] + 1.0) * 0.5
    spray = np.sin(t * speed * 2.5 + points[:, 0] * 4.0 + points[:, 2] * 4.0)
    spread = spray * amp * (0.3 + height * 0.7)
    result = points.copy()
    result[:, 0] += spread
    result[:, 2] += spread
    wobble = np.sin(t * speed * 3.0 + points[:, 0] * 5.0 + points[:, 2] * 5.0)
    result[:, 1] += wobble * amp * 0.25 * height
    return result
