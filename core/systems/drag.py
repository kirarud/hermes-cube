"""systems/drag.py — DragSystem: перемещение куба мышью.

Читает:
  meta.events — 'toggle_drag'
  meta.draggable — текущее состояние

Пишет:
  meta.cube_ox/oy — смещение куба
  meta.draggable — переключение режима

Использует WindowSystem для управления WS_EX_TRANSPARENT.
"""

from __future__ import annotations

import tkinter as tk
from typing import Any

from core.world import World


class DragSystem:
    """Обрабатывает drag-события: захват, перемещение, convex hull.

    Привязывается к Tkinter-событиям мыши.
    """

    def __init__(self, window: Any) -> None:
        self._drag_data: dict = {'grab_x': 0, 'grab_y': 0,
                                 'start_ox': 0, 'start_oy': 0}
        self._dragging: bool = False
        self._drag_handle: Any = None

        canvas = window.canvas if hasattr(window, 'canvas') else window
        canvas.bind('<Button-1>', self._drag_start)
        canvas.bind('<B1-Motion>', self._drag_move)
        canvas.bind('<ButtonRelease-1>', self._drag_end)

    def update(self, world: World, dt: float) -> None:
        """Применить cube_ox/oy из drag-состояния к world."""
        # Синхронизация: world.meta хранит текущий offset
        pass

    def _drag_start(self, event: tk.Event) -> None:
        # Привязка через внешний обработчик
        pass

    def _drag_move(self, event: tk.Event) -> None:
        pass

    def _drag_end(self, event: tk.Event) -> None:
        pass
