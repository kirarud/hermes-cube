"""systems/text_overlay.py — Парящие буквы над кубом.

Читает:
  meta.ai_response — текст для отображения
  meta.ai_thinking — если True, буквы ещё не готовы

Пишет:
  meta.ai_response — очищается после отображения

Управляет Tkinter-окном с анимированными буквами.
Отделено от AI-системы: не знает про ai_module, LM Studio.
"""

from __future__ import annotations

import math
import random
import tkinter as tk
from typing import Any, Dict, List, Optional

from core.world import World


class TextOverlaySystem:
    """System для парящих букв AI-ответа.

    Каждая буква — canvas текст, летит от центра в случайную сторону,
    замедляется, затухает, падает вниз (gravity).
    """

    BG_COLOR: str = '#000001'
    FONT: tuple = ('Segoe UI', 18, 'bold')
    LETTER_LIFE: float = 2.5
    LETTER_SPEED: float = 80.0
    MAX_PARTICLES: int = 60

    def __init__(self, root: tk.Tk) -> None:
        self.root: tk.Tk = root
        self.window: Optional[tk.Toplevel] = None
        self.canvas: Optional[tk.Canvas] = None
        self.particles: List[Dict[str, Any]] = []
        self._running: bool = False
        self._last_response: str = ''
        self._setup()

    def _setup(self) -> None:
        """Создать прозрачное overlay-окно."""
        self.window = tk.Toplevel(self.root)
        self.window.title('♢ Hermes Text')
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.window.geometry(f'{sw}x{sh}+0+0')
        self.window.configure(bg=self.BG_COLOR)
        self.window.attributes('-transparentcolor', self.BG_COLOR)
        self.window.overrideredirect(True)
        self.window.withdraw()

        self.canvas = tk.Canvas(
            self.window, bg=self.BG_COLOR, highlightthickness=0,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

    def update(self, world: World, dt: float) -> None:
        """Проверить ai_response → spawn букв, обновить анимацию."""
        # Спавн нового текста
        if (world.meta.ai_response
                and world.meta.ai_response != self._last_response
                and not world.meta.ai_thinking):
            self._last_response = world.meta.ai_response
            cx = world.meta.w / 2.0 + world.meta.cube_ox
            cy = world.meta.h / 2.0 + world.meta.cube_oy
            self._spawn_text(world.meta.ai_response, int(cx), int(cy))
            world.meta.ai_response = ''

        # Анимация существующих частиц
        self._animate(dt)

    def _spawn_text(self, text: str, from_x: int, from_y: int) -> None:
        if not self._running or not self.canvas:
            return
        if self.window and self.window.state() == 'withdrawn':
            self._show()

        for ch in text:
            if ch.strip() or ch in '.,!?—…':
                angle = random.uniform(0, 2 * math.pi)
                speed = self.LETTER_SPEED * random.uniform(0.7, 1.3)
                r = random.randint(200, 255)
                g = random.randint(150, 255)
                b = random.randint(100, 255)
                colour = f'#{r:02x}{g:02x}{b:02x}'
                if self.canvas is None:
                    continue
                item = self.canvas.create_text(
                    from_x, from_y, text=ch,
                    fill=colour, font=self.FONT,
                )
                self.particles.append({
                    'item': item,
                    'x': float(from_x),
                    'y': float(from_y),
                    'vx': math.cos(angle) * speed,
                    'vy': math.sin(angle) * speed,
                    'life': self.LETTER_LIFE,
                    'max_life': self.LETTER_LIFE,
                    'r': r, 'g': g, 'b': b,
                })

    def _animate(self, dt: float) -> None:
        if not self._running or not self.canvas:
            return

        alive = []
        for p in self.particles:
            p['life'] -= dt
            if p['life'] <= 0:
                try:
                    self.canvas.delete(p['item'])
                except Exception:
                    pass
                continue
            p['x'] += p['vx'] * dt
            p['y'] += p['vy'] * dt
            p['vy'] += 40.0 * dt  # gravity
            fade = max(0.0, p['life'] / p['max_life'])
            colour = f'#{int(p["r"]):02x}{int(p["g"]):02x}{int(p["b"]):02x}'
            try:
                self.canvas.coords(p['item'], int(p['x']), int(p['y']))
                self.canvas.itemconfig(p['item'], fill=colour)
            except Exception:
                pass
            alive.append(p)

        self.particles = alive[-self.MAX_PARTICLES:]

        # Auto-hide when empty
        if not self.particles and self.window and self.window.state() != 'withdrawn':
            self._hide()

    def _show(self) -> None:
        if self.window:
            self.window.deiconify()
            self.window.lift()
            self.window.attributes('-topmost', True)
        self._running = True

    def _hide(self) -> None:
        self._running = False
        if self.window:
            self.window.withdraw()
        self.particles.clear()

    def close(self) -> None:
        self._running = False
        self.particles.clear()
        if self.window:
            try:
                self.window.destroy()
            except Exception:
                pass
