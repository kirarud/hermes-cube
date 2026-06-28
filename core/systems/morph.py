"""systems/morph.py — Морфинг: lerp(base_position → target shape).

Читает: sim.base_position
Пишет:  sim.morphed

Никогда не мутирует base_position.
"""

from __future__ import annotations

import numpy as np

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
