#!/usr/bin/env python3
"""core/world.py — Единый data-слой Hermes Engine.

ВСЁ состояние проекта живёт здесь. Никакая логика не хранит своё
состояние отдельно — только World.

Три зоны:
  sim    — simulation state (положение, цвет, форма)
  render — presentation state (готово к отрисовке)
  meta   — management (конфиг, время, AI, события)

Stage-буферы (sim):
  Каждая стадия конвейера пишет в свой буфер, читает из предыдущего.
  Это исключает copy(), side-effects и ошибки порядка.

  base_position  ← GridGenerator (неизменна после генерации)
  morphed        ← Morph (lerp base→target)
  animated       ← Animation (смещение от morphed)
  world_position ← Rotation (поворот → финальная 3D-позиция)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class SimState:
    """Симуляционное состояние — stage-буферы конвейера.

    Все массивы — плоские (N, 3), индексированные по entity_id.
    N — активное количество частиц (<= pool_size).
    """

    # ── Stage buffers (конвейер: base → morphed → animated → world) ──
    base_position: NDArray[np.float64]   # (N, 3) — исходная сетка куба
    morphed: NDArray[np.float64]         # (N, 3) — после морфинга
    animated: NDArray[np.float64]        # (N, 3) — после анимации
    world_position: NDArray[np.float64]   # (N, 3) — после поворота, финал 3D

    # Цвет (базовый, не зависит от стадии)
    color: NDArray[np.float64]           # (N, 3) — r0/g0/b0 (0-255)

    # Мета-данные частиц
    alive: NDArray[np.bool_]             # (N,) — active/dormant
    symbol_idx: NDArray[np.int32]        # (N,) — индекс символа в symbol_set

    # Ресурсы (не привязаны к entity)
    shape_cache: Dict[str, NDArray[np.float64]] = field(default_factory=dict)

    # Pool management
    pool_size: int = 4096
    active_count: int = 0


@dataclass
class RenderState:
    """Состояние, готовое к отрисовке — заполняется View Stage."""
    projected_x: NDArray[np.float64] = field(default_factory=lambda: np.array([], dtype=np.float64))
    projected_y: NDArray[np.float64] = field(default_factory=lambda: np.array([], dtype=np.float64))
    final_rgb: NDArray[np.uint8] = field(default_factory=lambda: np.array([], dtype=np.uint8))
    depth: NDArray[np.float64] = field(default_factory=lambda: np.array([], dtype=np.float64))
    cell: int = 6
    symbol: str = 'square'

    # Трейлы
    trail_enabled: bool = False
    trail_history: deque = field(default_factory=lambda: deque(maxlen=12))
    trail_layer: Optional[Tuple[NDArray[np.float64],
                                 NDArray[np.float64],
                                 NDArray[np.uint8]]] = None


@dataclass
class MetaState:
    """Управляющее состояние — конфиг, время, AI, события."""
    config: Dict[str, Any] = field(default_factory=dict)
    t: float = 0.0
    dt: float = 0.0
    frame: int = 0

    # AI
    mood: str = 'idle'
    color_shift: float = 0.0
    ai_response: str = ''
    ai_thinking: bool = False
    ai_requested: bool = False
    ai_ready: bool = False
    chat_history: List[Dict[str, str]] = field(default_factory=list)

    # Input / Events
    events: deque = field(default_factory=deque)
    input_buffer: str = ''

    # Cube offset (drag)
    cube_ox: float = 0.0
    cube_oy: float = 0.0
    draggable: bool = False

    w: int = 800
    h: int = 600


@dataclass
class World:
    """Единый data-слой Hermes Engine.

    Все три зоны — простые dataclass-ы. Никакой логики.
    Мутируется ТОЛЬКО системами (System → world).
    """
    sim: SimState
    render: RenderState
    meta: MetaState

    @classmethod
    def create(cls, config: Dict[str, Any], n_particles: int = 864,
               pool_size: int = 4096) -> World:
        """Создать World с инициализированными массивами."""
        z = np.zeros((n_particles, 3), dtype=np.float64)
        sim = SimState(
            base_position=z.copy(),
            morphed=z.copy(),
            animated=z.copy(),
            world_position=z.copy(),
            color=z.copy(),
            alive=np.ones(n_particles, dtype=np.bool_),
            symbol_idx=np.zeros(n_particles, dtype=np.int32),
            pool_size=pool_size,
            active_count=n_particles,
        )
        render = RenderState(
            projected_x=np.zeros(n_particles, dtype=np.float64),
            projected_y=np.zeros(n_particles, dtype=np.float64),
            final_rgb=np.zeros((n_particles, 3), dtype=np.uint8),
            depth=np.zeros(n_particles, dtype=np.float64),
            cell=config.get('cell_size', 6),
            symbol=config.get('symbol', 'square'),
        )
        meta = MetaState(config=config)
        return cls(sim=sim, render=render, meta=meta)

    def resize_pool(self, new_count: int) -> None:
        """Изменить активное количество частиц (массивы не перевыделяются)."""
        if new_count > self.sim.pool_size:
            raise ValueError(
                f"Cannot grow beyond pool_size={self.sim.pool_size}")
        self.sim.active_count = max(0, new_count)
        n = self.sim.active_count
        self.render.projected_x = self.render.projected_x[:n]
        self.render.projected_y = self.render.projected_y[:n]
        self.render.final_rgb = self.render.final_rgb[:n]
        self.render.depth = self.render.depth[:n]
