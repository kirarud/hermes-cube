"""systems/input.py — InputSystem: клавиши → world.meta.events.

Читает Tkinter-события клавиш, пишет events в world.meta.
Не содержит игровой логики.
"""

from __future__ import annotations

import tkinter as tk
from typing import Any

from core.world import World


class InputSystem:
    """Привязывает клавиши к world.meta.events.

    Каждое нажатие → (key, action) в events deque.
    """

    def __init__(self, window: Any, world: World) -> None:
        self.world = world
        root = window.root if hasattr(window, 'root') else window

        bindings = {
            '<Escape>': ('hide',),
            'q': ('hide',),
            'h': ('hide',),
            's': ('settings',),
            'c': ('toggle_input',),
            'C': ('toggle_input',),
            't': ('toggle_drag',),
            'T': ('toggle_drag',),
            'r': ('toggle_trails',),
            'R': ('toggle_trails',),
            'g': ('toggle_pixelgrid',),
            'G': ('toggle_pixelgrid',),
            'a': ('spawn_agent',),
            'A': ('spawn_agent',),
        }

        for key, action in bindings.items():
            root.bind(key, lambda e, a=action: self._emit(a))

    def _emit(self, action: tuple) -> None:
        self.world.meta.events.append(action)
