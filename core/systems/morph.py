"""systems/morph.py — Плавный морфинг между формами.

Берёт кубическую сетку из world.sim.position и интерполирует
к целевой форме из shape_cache.
"""

from __future__ import annotations

import numpy as np

from core.world import World


def update(world: World, dt: float) -> None:
    """Применить морфинг: lerp(cube, target, morph_progress).

    Если morph_progress == 0.0 — ничего не делает (чистый куб).
    Если morph_progress == 1.0 — чистая целевая форма.
    """
    cfg = world.meta.config
    morph: float = cfg.get('morph_progress', 0.0)
    if morph <= 0.0:
        return

    shape_name: str = cfg.get('shape_preset', 'cube')
    target = world.sim.shape_cache.get(shape_name)
    if target is None:
        return

    n = world.sim.active_count
    if n == 0:
        return

    pts = world.sim.position[:n]
    target_pts = target[:n]
    world.sim.position[:n] = pts * (1.0 - morph) + target_pts * morph
