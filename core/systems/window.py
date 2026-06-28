"""systems/window.py — WindowSystem: управление Tkinter-окном.

Владеет:
  - Tk root (создание, конфигурация)
  - Fullscreen overlay (overrideredirect, transparent)
  - Canvas для рендера
  - WS_EX_TRANSPARENT (click-through)

Не содержит:
  - Игровую логику
  - Pipeline
  - AI
"""

from __future__ import annotations

import ctypes
import sys
import tkinter as tk
from typing import Any, Optional

from core.world import World

TRANSPARENT_COLOR: str = '#000001'

# Win32 helpers
_user32 = None
if sys.platform == 'win32':
    _user32 = ctypes.windll.user32


class WindowSystem:
    """Управляет Tkinter-окном куба.

    Создаёт полноэкранное прозрачное окно с Canvas.
    Управляет click-through (WS_EX_TRANSPARENT) и drag-режимом.
    """

    def __init__(self, title: str = '♢ Hermes Cube') -> None:
        self.root = tk.Tk()
        self.root.title(title)
        self.root.protocol('WM_DELETE_WINDOW', self.hide)

        # Fullscreen transparent
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.geometry(f'{sw}x{sh}+0+0')
        self.root.resizable(True, True)
        self.root.overrideredirect(True)
        self.root.configure(bg=TRANSPARENT_COLOR)
        self.root.attributes('-transparentcolor', TRANSPARENT_COLOR)

        # Canvas
        self.canvas = tk.Canvas(
            self.root, bg=TRANSPARENT_COLOR, highlightthickness=0,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # WS_EX_TRANSPARENT — click-through by default
        self._clickthrough: bool = True
        if _user32 is not None:
            hwnd = ctypes.c_void_p(self.root.winfo_id())
            GWL_EXSTYLE = -20
            ex_style = _user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            ex_style |= 0x00000020  # WS_EX_TRANSPARENT
            _user32.SetWindowLongW(hwnd, GWL_EXSTYLE, ex_style)

        self.root.update()

    def show(self) -> None:
        """Показать окно."""
        self.root.deiconify()
        self.root.lift()
        self.root.lift()
        self.root.lift()
        self.root.update()

    def hide(self) -> None:
        """Скрыть окно (оставляет трей живым)."""
        self.root.withdraw()

    def toggle_visible(self) -> None:
        """Flip между скрытым и видимым."""
        if self.root.state() == 'withdrawn':
            self.show()
        else:
            self.hide()

    def set_topmost(self, on: bool = True) -> None:
        """Поверх всех окон."""
        self.root.attributes('-topmost', on)

    def set_clickthrough(self, on: bool = True) -> None:
        """Включить/выключить click-through."""
        if _user32 is None:
            return
        if on == self._clickthrough:
            return
        self._clickthrough = on
        hwnd = ctypes.c_void_p(self.root.winfo_id())
        GWL_EXSTYLE = -20
        ex_style = _user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        if on:
            ex_style |= 0x00000020
            self.canvas.config(cursor='')
        else:
            ex_style &= ~0x00000020
            self.canvas.config(cursor='fleur')
        _user32.SetWindowLongW(hwnd, GWL_EXSTYLE, ex_style)

    def destroy(self) -> None:
        """Полностью уничтожить окно."""
        try:
            self.root.destroy()
        except Exception:
            pass

    @property
    def w(self) -> int:
        return max(10, self.canvas.winfo_width())

    @property
    def h(self) -> int:
        return max(10, self.canvas.winfo_height())
