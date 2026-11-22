"""systems/morph.py — Морфинг: lerp(base_position → target shape).

Читает: sim.base_position
Пишет:  sim.morphed

Никогда не мутирует base_position.
"""

from __future__ import annotations

import numpy as np

from core.face_avatar import regions
from core.world import World


def update(world: World, dt: float) -> None:
    cfg = world.meta.config
    morph: float = cfg.get('morph_progress', 0.0)
    n = world.sim.active_count
    if n == 0:
        return

    if morph <= 0.0:
        # Копируем base → morphed (без морфинга)
        world.sim.morphed[:n] = world.sim.base_position[:n]
        return

    shape_name: str = cfg.get('shape_preset', 'cube')
    target = world.sim.shape_cache.get(shape_name)
    if target is None:
        world.sim.morphed[:n] = world.sim.base_position[:n]
        return

    base = world.sim.base_position[:n]
    tgt = target[:n]
    # lerp: base * (1-t) + target * t
    world.sim.morphed[:n] = base * (1.0 - morph) + tgt * morph

    # The friendly companion has a neutral mouth by default. Only an explicit
    # happy mood creates a modest smile, without changing the cached shape.
    if (shape_name == 'face' and morph >= 0.99 and
            world.meta.mood == 'happy' and not world.meta.text_mode):
        _, _, c, d, _ = regions(n)
        if d > c:
            u = np.linspace(-1.0, 1.0, d-c)
            world.sim.morphed[c:d, 1] -= .055 * (1.0-u*u)
