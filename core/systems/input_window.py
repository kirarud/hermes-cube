"""systems/input_window.py — Окно ввода текста для AI.

Создаёт Tkinter-окно внизу экрана при вызове toggle().
Пишет введённый текст в world.meta.input_buffer.
Не зависит от ai_module, LM Studio, TextOverlay.
"""

from __future__ import annotations

import tkinter as tk
from typing import Any, Callable, Optional

from core.world import World


class InputWindowSystem:
    """Компактное окно ввода внизу экрана.

    Открывается по C / контекстному меню.
    При Enter пишет текст в world.meta.input_buffer и закрывается.
    """

    WIDTH: int = 440
    HEIGHT: int = 48

    def __init__(self, root: tk.Tk) -> None:
        self.root: tk.Tk = root
        self.window: Optional[tk.Toplevel] = None
        self._entry: Optional[tk.Entry] = None
        self._var: tk.StringVar = tk.StringVar()

    def toggle(self) -> None:
        """Открыть/закрыть окно ввода."""
        if self.window and self.window.winfo_exists():
            self.hide()
        else:
            self.show()

    def show(self) -> None:
        """Создать и показать окно ввода."""
        self.hide()
        self.window = tk.Toplevel(self.root)
        self.window.title('♢ Hermes Cube — Ввод')
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        ix = (sw - self.WIDTH) // 2
        iy = sh - self.HEIGHT - 50
        self.window.geometry(f'{self.WIDTH}x{self.HEIGHT}+{ix}+{iy}')
        self.window.configure(bg='#0d0d1a')
        self.window.attributes('-topmost', True)
        self.window.resizable(False, False)
        self.window.overrideredirect(True)

        frame = tk.Frame(self.window, bg='#0d0d1a',
                         highlightbackground='#e94560',
                         highlightthickness=1, bd=0)
        frame.pack(fill='both', expand=True)

        self._var.set('')
        self._entry = tk.Entry(
            frame, textvariable=self._var,
            bg='#0d0d1a', fg='#e0e0e0', relief=tk.FLAT,
            font=('Segoe UI', 14), insertbackground='#e94560',
            highlightthickness=0, bd=4,
        )
        self._entry.pack(fill='both', expand=True, padx=6, pady=4)
        self._entry.focus_set()
        self._entry.bind('<Return>', self._on_enter)
        self._entry.bind('<Escape>', lambda e: self.hide())

    def hide(self) -> None:
        """Закрыть окно ввода."""
        if self.window:
            try:
                self.window.destroy()
            except Exception:
                pass
            self.window = None

    def _on_enter(self, _event: Any = None) -> None:
        text = self._var.get().strip()
        if not text:
            return
        self.hide()
        # Пишем в мировой буфер — AISystem прочитает
        # Для этого нужен доступ к world.
        # Временно: используем callback.
        if self._on_submit:
            self._on_submit(text)

    # Временный коллбэк-мост (пока world не подключён напрямую)
    _on_submit: Optional[Callable[[str], None]] = None

    def connect_world(self, world: World) -> None:
        """Подключить world.meta.input_buffer."""
        self._on_submit = lambda text: setattr(world.meta, 'input_buffer', text)
