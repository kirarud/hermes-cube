#!/usr/bin/env python3
"""hermes_hud.py — Нативный мини-HUD Гермеса на рабочем столе macOS.

Аватар Куба:
  - Компактный прозрачный плавающий виджет с частицами куба
  - Анимированное дыхание, пульсация, вращение
  - Реакция цвета и света на состояние агента (idle, thinking, speaking)
  - Голосовая озвучка (macOS say) при клике или ответе
  - Мини-меню с кнопками быстрого действия:
      [ 🎙 Сказать / Голос ]
      [ 🧠 Внимание / Мысли ]
      [ 🌐 Граф Obsidian ]
      [ ⚙ Настройки ]
      [ ✕ Скрыть ]
"""

from __future__ import annotations

import math
import os
import random
import subprocess
import threading
import time
import tkinter as tk
from typing import List, Tuple, Dict, Any

# Цвета состояний агента
MOOD_THEMES = {
    'idle': {
        'primary': (80, 200, 255),    # Неоново-голубой
        'secondary': (40, 100, 255),
        'glow': '#2266aa',
        'speed': 0.03,
        'pulse_freq': 1.8,
        'label': 'Hermes: Спит / Наблюдает',
    },
    'thinking': {
        'primary': (255, 180, 50),     # Янтарно-золотой
        'secondary': (255, 100, 30),
        'glow': '#aa6611',
        'speed': 0.08,
        'pulse_freq': 4.0,
        'label': 'Hermes: Анализирует...',
    },
    'speaking': {
        'primary': (120, 255, 180),    # Изумрудно-зеленый
        'secondary': (50, 255, 120),
        'glow': '#22aa66',
        'speed': 0.05,
        'pulse_freq': 2.8,
        'label': 'Hermes: Говорит',
    },
}

class HermesAvatarHUD:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Hermes HUD")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)

        # Прозрачность окна на macOS
        try:
            self.root.attributes("-transparent", True)
            self.root.config(bg="systemTransparent")
            self.bg_color = "systemTransparent"
        except Exception:
            self.root.attributes("-alpha", 0.94)
            self.bg_color = "#12141c"
            self.root.config(bg=self.bg_color)

        self.width = 240
        self.height = 290

        # Позиция: правый верхний угол экрана
        screen_w = self.root.winfo_screenwidth()
        pos_x = screen_w - self.width - 30
        pos_y = 60
        self.root.geometry(f"{self.width}x{self.height}+{pos_x}+{pos_y}")

        # Состояние
        self.mood = "idle"
        self.angle_x = 0.3
        self.angle_y = 0.4
        self.angle_z = 0.1
        self.menu_visible = True
        self.status_text = "Hermes Online"

        # Вершины 3D куба
        self.vertices = [
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1]
        ]
        self.edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]

        # Внутреннее облако частиц
        random.seed(42)
        self.particles: List[List[float]] = []
        for _ in range(48):
            px = random.uniform(-0.85, 0.85)
            py = random.uniform(-0.85, 0.85)
            pz = random.uniform(-0.85, 0.85)
            self.particles.append([px, py, pz])

        # Drag & drop окна
        self._drag_start_x = 0
        self._drag_start_y = 0

        self._build_ui()
        self._bind_events()

        # Запуск цикла анимации
        self.start_time = time.time()
        self._animate()

    def _build_ui(self) -> None:
        self.main_frame = tk.Frame(self.root, bg="#10121a", bd=1, highlightthickness=1, highlightbackground="#252a3a")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Верхняя панель заголовка (хэндл для перетаскивания)
        self.header = tk.Frame(self.main_frame, bg="#161924", height=24)
        self.header.pack(fill=tk.X, side=tk.TOP)

        self.title_label = tk.Label(
            self.header, text="♢ HERMES AVATAR",
            font=("Menlo", 9, "bold"), fg="#88aaff", bg="#161924"
        )
        self.title_label.pack(side=tk.LEFT, padx=8, pady=3)

        self.close_btn = tk.Label(
            self.header, text="✕",
            font=("Menlo", 10, "bold"), fg="#667799", bg="#161924", cursor="hand2"
        )
        self.close_btn.pack(side=tk.RIGHT, padx=8, pady=2)
        self.close_btn.bind("<Button-1>", lambda e: self.root.destroy())

        # Холст для 3D Куба
        self.canvas = tk.Canvas(
            self.main_frame, width=220, height=130,
            bg="#0d0f15", highlightthickness=0
        )
        self.canvas.pack(fill=tk.X, padx=6, pady=4)

        # Статус
        self.status_lbl = tk.Label(
            self.main_frame, text=self.status_text,
            font=("Menlo", 8), fg="#64748b", bg="#10121a"
        )
        self.status_lbl.pack(pady=(0, 4))

        # Панель быстрых кнопок
        self.btn_panel = tk.Frame(self.main_frame, bg="#10121a")
        self.btn_panel.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

        # Стиль кнопок
        btn_style = {
            "font": ("Menlo", 8, "bold"),
            "bd": 0,
            "cursor": "hand2",
            "activebackground": "#2a334d",
            "relief": "flat",
            "padx": 4, "pady": 4
        }

        row1 = tk.Frame(self.btn_panel, bg="#10121a")
        row1.pack(fill=tk.X, pady=2)

        self.b_voice = tk.Button(
            row1, text="🎙 Голос", bg="#1e2333", fg="#38bdf8",
            command=self._on_voice_click, **btn_style
        )
        self.b_voice.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))

        self.b_think = tk.Button(
            row1, text="🧠 Импульс", bg="#1e2333", fg="#fbbf24",
            command=self._on_think_click, **btn_style
        )
        self.b_think.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

        row2 = tk.Frame(self.btn_panel, bg="#10121a")
        row2.pack(fill=tk.X, pady=2)

        self.b_graph = tk.Button(
            row2, text="🌐 Граф Obsidian", bg="#1e2333", fg="#a78bfa",
            command=self._on_graph_click, **btn_style
        )
        self.b_graph.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))

        self.b_dash = tk.Button(
            row2, text="⚡ Dashboard", bg="#1e2333", fg="#34d399",
            command=self._on_dash_click, **btn_style
        )
        self.b_dash.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

    def _bind_events(self) -> None:
        for widget in (self.header, self.title_label):
            widget.bind("<ButtonPress-1>", self._start_drag)
            widget.bind("<B1-Motion>", self._on_drag)

        self.canvas.bind("<Button-1>", lambda e: self._cycle_mood())

    def _start_drag(self, event) -> None:
        self._drag_start_x = event.x_root - self.root.winfo_x()
        self._drag_start_y = event.y_root - self.root.winfo_y()

    def _on_drag(self, event) -> None:
        new_x = event.x_root - self._drag_start_x
        new_y = event.y_root - self._drag_start_y
        self.root.geometry(f"+{new_x}+{new_y}")

    def _cycle_mood(self) -> None:
        moods = list(MOOD_THEMES.keys())
        idx = (moods.index(self.mood) + 1) % len(moods)
        self.set_mood(moods[idx])

    def set_mood(self, mood: str) -> None:
        if mood in MOOD_THEMES:
            self.mood = mood
            theme = MOOD_THEMES[mood]
            self.status_lbl.config(text=theme['label'])
            self._update_glow()

    def _update_glow(self) -> None:
        theme = MOOD_THEMES[self.mood]
        self.main_frame.config(highlightbackground=theme['glow'])

    def _on_voice_click(self) -> None:
        self.set_mood('speaking')
        threading.Thread(target=self._speak_phrase, daemon=True).start()

    def _speak_phrase(self) -> None:
        phrases = [
            "Гермес на связи. Физическое воплощение активировано.",
            "Система готова. Окружение в фокусе.",
            "Архитектурный мост синхронизирован."
        ]
        text = random.choice(phrases)
        subprocess.run(["say", "-v", "Milena", text])
        self.root.after(500, lambda: self.set_mood('idle'))

    def _on_think_click(self) -> None:
        self.set_mood('thinking')
        self.root.after(3000, lambda: self.set_mood('idle'))

    def _on_graph_click(self) -> None:
        subprocess.Popen(
            ["open", "obsidian://open"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

    def _on_dash_click(self) -> None:
        subprocess.Popen(
            ["open", "http://127.0.0.1:9119"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

    def _project_point(self, x: float, y: float, z: float, cx: float, cy: float, scale: float) -> Tuple[float, float]:
        # Вращение
        rad_x, rad_y, rad_z = self.angle_x, self.angle_y, self.angle_z

        # Rot X
        y1 = y * math.cos(rad_x) - z * math.sin(rad_x)
        z1 = y * math.sin(rad_x) + z * math.cos(rad_x)

        # Rot Y
        x2 = x * math.cos(rad_y) + z1 * math.sin(rad_y)
        z2 = -x * math.sin(rad_y) + z1 * math.cos(rad_y)

        # Rot Z
        x3 = x2 * math.cos(rad_z) - y1 * math.sin(rad_z)
        y3 = x2 * math.sin(rad_z) + y1 * math.cos(rad_z)

        # Перспектива
        distance = 3.2
        fov = distance / (distance + z2)
        px = cx + x3 * scale * fov
        py = cy + y3 * scale * fov
        return px, py

    def _animate(self) -> None:
        t = time.time() - self.start_time
        theme = MOOD_THEMES[self.mood]

        # Вращение
        self.angle_y += theme['speed']
        self.angle_x += theme['speed'] * 0.45
        self.angle_z += theme['speed'] * 0.2

        # Дыхание / пульс
        pulse = 1.0 + 0.12 * math.sin(t * theme['pulse_freq'])
        scale = 36.0 * pulse

        self.canvas.delete("all")
        cx = 110
        cy = 65

        r1, g1, b1 = theme['primary']
        r2, g2, b2 = theme['secondary']

        # Отрисовка внутреннего облака частиц
        for pt in self.particles:
            px, py = self._project_point(pt[0], pt[1], pt[2], cx, cy, scale)
            pr = 1.5
            p_col = f"#{r2:02x}{g2:02x}{b2:02x}"
            self.canvas.create_oval(px - pr, py - pr, px + pr, py + pr, fill=p_col, outline="")

        # Проекция вершин куба
        proj_vertices = [
            self._project_point(v[0], v[1], v[2], cx, cy, scale)
            for v in self.vertices
        ]

        # Отрисовка ребер куба
        edge_col = f"#{r1:02x}{g1:02x}{b1:02x}"
        for e1, e2 in self.edges:
            p1 = proj_vertices[e1]
            p2 = proj_vertices[e2]
            self.canvas.create_line(p1[0], p1[1], p2[0], p2[1], fill=edge_col, width=1.5)

        # Отрисовка узловых точек
        for p in proj_vertices:
            self.canvas.create_oval(p[0] - 2.5, p[1] - 2.5, p[0] + 2.5, p[1] + 2.5, fill="#ffffff", outline=edge_col)

        self.root.after(33, self._animate)

    def run(self) -> None:
        self.root.mainloop()

if __name__ == "__main__":
    app = HermesAvatarHUD()
    app.run()
