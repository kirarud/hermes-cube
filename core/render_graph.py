#!/usr/bin/env python3
"""core/render_graph.py — Data-driven render pipeline for Hermes Cube.

АРХИТЕКТУРА
-----------
Render Graph отделяет логику рендера (Pass) от оркестрации (RenderGraph)
и бэкенда вывода (renderer.PointCloudRenderer).

Каждый Pass штампует в shared RGBA-буфер. Pass-ы выполняются
в порядке регистрации; позже зарегистрированный — поверх предыдущих.

Жизненный цикл кадра:
  1. CubeApp вычисляет проекции, цвета, трейлы, AI-состояние
  2. Упаковывает всё в FrameContext
  3. Вычисляет bbox (через PointCloudRenderer.compute_bbox)
  4. Выделяет RGBA-буфер (через PointCloudRenderer.allocate_rgba)
  5. RenderGraph.execute(buf, x0, y0, ctx) — очищает буфер и запускает Pass-ы
  6. PointCloudRenderer.blit(buf, x0, y0) — выводит на canvas

ПЛАН ЭВОЛЮЦИИ (следующие шаги):
  - Blend modes (add/screen для трейлов)
  - Post-process pass (bloom, glow, gaussian blur)
  - Tile-based chunky rasterisation
  - GPU via moderngl (Pass → GL shader)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from renderer import PointCloudRenderer


# ═══════════════════════════════════════════════════════════════════════════
# FrameContext — всё изменяемое состояние одного кадра
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class FrameContext:
    """Входные данные для пассов рендера — всё, что меняется кадр→кадр.

    Заполняется CubeApp перед каждым вызовом RenderGraph.execute().
    """
    # Проекции частиц (экранные координаты)
    px: NDArray[np.float64]      # (N,) screen x
    py: NDArray[np.float64]      # (N,) screen y
    pz: NDArray[np.float64]      # (N,) depth (для painter's algorithm)
    rgb: NDArray[np.uint8]       # (N, 3) per-particle цвет (уже с depth + HSV)

    # Параметры отрисовки
    cell: int                    # размер частицы в px
    symbol: str                  # 'square' | 'circle' | 'dot'

    # Трейлы
    trail_enabled: bool
    trail_layer: Optional[Tuple[NDArray[np.float64],
                                 NDArray[np.float64],
                                 NDArray[np.uint8]]]  # (tx, ty, trgb) или None

    # Символьный режим
    using_chars: bool
    char_list: Optional[List[str]]  # [char, ...] по одному на частицу
    symbols_set: List[str]          # из SYMBOL_SETS

    # Текущий конфиг (для будущих пассов, которые читают настройки)
    config: Dict[str, Any]

    # Размер canvas (для клиппинга)
    w: int
    h: int


# ═══════════════════════════════════════════════════════════════════════════
# Pass — один слой/эффект
# ═══════════════════════════════════════════════════════════════════════════


class Pass:
    """Базовый класс render-пасса.

    Подклассы переопределяют render() — штампуют пиксели в RGBA-буфер.
    """

    name: str = 'unnamed_pass'
    enabled: bool = True

    def render(
        self,
        buf: NDArray[np.uint8],  # (H, W, 4) RGBA
        x0: int,                 # offset X буфера (экранные → буфер)
        y0: int,                 # offset Y буфера
        ctx: FrameContext,
    ) -> None:
        """Штамповать слой в буфер. buf модифицируется на месте."""
        raise NotImplementedError


# ═══════════════════════════════════════════════════════════════════════════
# GeometryPass — штамповка частиц куба (точки или символы)
# ═══════════════════════════════════════════════════════════════════════════


class GeometryPass(Pass):
    """Основной пасс: рисует частицы куба (точки или символы)."""

    name: str = 'geometry'

    def render(
        self,
        buf: NDArray[np.uint8],
        x0: int, y0: int,
        ctx: FrameContext,
    ) -> None:
        if len(ctx.px) == 0:
            return

        if ctx.using_chars and ctx.char_list is not None:
            PointCloudRenderer.stamp_chars(
                buf, x0, y0,
                ctx.px, ctx.py, ctx.rgb,
                ctx.cell, ctx.char_list,
            )
        else:
            PointCloudRenderer.stamp_points(
                buf, x0, y0,
                ctx.px, ctx.py, ctx.rgb,
                ctx.cell, ctx.symbol,
            )


# ═══════════════════════════════════════════════════════════════════════════
# TrailPass — шлейф частиц (под кубом)
# ═══════════════════════════════════════════════════════════════════════════


class TrailPass(Pass):
    """Пасс трейлов: рисует шлейф ПОД геометрией куба."""

    name: str = 'trails'

    def render(
        self,
        buf: NDArray[np.uint8],
        x0: int, y0: int,
        ctx: FrameContext,
    ) -> None:
        if not ctx.trail_enabled or ctx.trail_layer is None:
            return
        tx, ty, trgb = ctx.trail_layer
        if len(tx) == 0:
            return
        # Трейлы — одиночные пиксели (cell=1)
        PointCloudRenderer.stamp_points(
            buf, x0, y0,
            tx, ty, trgb,
            1, 'square',
        )


# ═══════════════════════════════════════════════════════════════════════════
# RenderGraph — оркестратор пассов
# ═══════════════════════════════════════════════════════════════════════════


class RenderGraph:
    """Data-driven render pipeline.

    Пример использования:
        graph = RenderGraph()
        graph.add_pass(TrailPass())
        graph.add_pass(GeometryPass())
        ...
        rgba = graph.execute(ctx, layers_info)
    """

    def __init__(self) -> None:
        self.passes: List[Pass] = []

    def add_pass(self, p: Pass) -> None:
        """Добавить пасс в конец очереди (поверх предыдущих)."""
        self.passes.append(p)

    def get_pass(self, name: str) -> Optional[Pass]:
        """Найти пасс по имени."""
        for p in self.passes:
            if p.name == name:
                return p
        return None

    def execute(
        self,
        ctx: FrameContext,
        renderer: Optional[Any] = None,
    ) -> Tuple[Optional[NDArray[np.uint8]], Optional[Tuple[int, int, int, int]]]:
        """Выполнить все пассы → готовый RGBA-буфер + bbox.

        Шаги:
          1. Собрать информацию о слоях для bbox
          2. Выделить RGBA-буфер (из пула renderer, если передан)
          3. Очистить
          4. Запустить пассы по порядку

        Возвращает (rgba_buf, (x0, y0, x1, y1)) или (None, None) если пусто.
        """
        # --- Сбор bbox ---
        layers_info: List[Dict[str, Any]] = []

        # Трейлы
        if ctx.trail_enabled and ctx.trail_layer is not None:
            tx, ty, _ = ctx.trail_layer
            if len(tx) > 0:
                layers_info.append({'px': tx, 'py': ty, 'cell': 1})

        # Геометрия
        if len(ctx.px) > 0:
            layers_info.append({'px': ctx.px, 'py': ctx.py, 'cell': ctx.cell})

        bbox = PointCloudRenderer.compute_bbox(layers_info, margin=8)
        if bbox is None:
            return (None, None)

        x0, y0, x1, y1 = bbox

        # --- Выделение буфера (из пула, если есть renderer) ---
        w = x1 - x0
        h = y1 - y0
        if renderer is not None and hasattr(renderer, 'get_buffer'):
            buf = renderer.get_buffer(w, h)
        else:
            buf = PointCloudRenderer.allocate_rgba(x0, y0, x1, y1)

        # --- Выполнение пассов ---
        for p in self.passes:
            if p.enabled:
                try:
                    p.render(buf, x0, y0, ctx)
                except Exception:
                    pass  # изолируем сломанный пасс

        return (buf, bbox)
