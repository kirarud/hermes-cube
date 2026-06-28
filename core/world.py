#!/usr/bin/env python3
"""core/world.py — Единый data-слой Hermes Engine.

ВСЁ состояние проекта живёт здесь. Никакая логика не хранит своё
состояние отдельно — только World.

Три зоны:
  sim    — simulation state (физика, положение, цвет)
  render — presentation state (готово к отрисовке)
  meta   — management (конфиг, время, AI, события)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class SimState:
    """Симуляционное состояние — меняется системами каждый кадр.

    Все массивы — плоские (N, 3) или (N,), индексированные по entity_id.
    N — активное количество частиц (<= pool_size).
    """
    position: NDArray[np.float64]   # (N, 3) — 3D-координаты [-1, 1]
    velocity: NDArray[np.float64]   # (N, 3) — скорости для анимаций
    color: NDArray[np.float64]      # (N, 3) — базовые цвета r0/g0/b0 (0-255)

    # Мета-данные частиц
    alive: NDArray[np.bool_]        # (N,) — active/dormant
    symbol_idx: NDArray[np.int32]   # (N,) — индекс символа в symbol_set

    # Ресурсы (не привязаны к entity)
    shape_cache: Dict[str, NDArray[np.float64]] = field(default_factory=dict)

    # Pool management
    pool_size: int = 4096
    active_count: int = 0

    # Shape generators
    shape_cache: Dict[str, NDArray[np.float64]] = field(default_factory=dict)


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
    t: float = 0.0           # глобальное время (сек)
    dt: float = 0.0          # шаг кадра (сек)
    frame: int = 0           # номер кадра

    # AI
    mood: str = 'idle'
    color_shift: float = 0.0
    ai_response: str = ''
    ai_thinking: bool = False
    ai_requested: bool = False

    # Input / Events
    events: deque = field(default_factory=deque)
    input_buffer: str = ''

    # Cube offset (drag)
    cube_ox: float = 0.0
    cube_oy: float = 0.0
    draggable: bool = False

    # Размеры окна
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
        """Создать World с инициализированными массивами.

        Args:
            config: Начальная конфигурация (из load_config()).
            n_particles: Начальное количество частиц.
            pool_size: Максимальный размер пула (для spawner-ов).
        """
        sim = SimState(
            position=np.zeros((n_particles, 3), dtype=np.float64),
            velocity=np.zeros((n_particles, 3), dtype=np.float64),
            color=np.zeros((n_particles, 3), dtype=np.float64),
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
