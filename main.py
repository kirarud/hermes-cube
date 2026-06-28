#!/usr/bin/env python3
"""main.py — Hermes Engine v2 entry point.

Заменяет CubeApp. Создаёт World, Pipeline, Window, запускает loop.

Жизненный цикл:
  1. Создать Window (Tk root, fullscreen, transparent)
  2. Создать World с конфигом
  3. Собрать Pipeline (Sim → FX → View)
  4. Создать RenderGraph (Trails → Geometry)
  5. Создать AI-системы (TextOverlay, InputWindow)
  6. Запустить mainloop (pipeline.run() → render_graph.execute() → blit)
"""

from __future__ import annotations

import os
import sys
import threading
import tkinter as tk
from typing import Any, Dict, List, Optional

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from core.world import World
from core.pipeline import Pipeline, Stage, Schedule, build_default_pipeline
from core.render_graph import RenderGraph, GeometryPass, TrailPass, FrameContext
from core.systems.window import WindowSystem
from core.systems.text_overlay import TextOverlaySystem
from core.systems.input_window import InputWindowSystem
from core.systems.drag import DragSystem
from renderer import PointCloudRenderer
from char_cube import SYMBOL_SETS
from cube_app import (
    load_config, save_config, DEFAULT_CONFIG,
    MIN_CELL_SIZE, MAX_CELL_SIZE, FRAME_MS,
    _convex_hull_2d, _expand_hull, _remove_tray_icon_force,
    UI_ACCENT, SettingsWindow,
)


def _create_tray_image():
    """Создать иконку трея (64×64). Копия из cube_app.py."""
    from PIL import Image, ImageDraw
    tray_path = os.path.join(
        os.environ.get('APPDATA', os.path.expanduser('~')),
        'HermesCube', 'tray_icon.png',
    )
    if os.path.isfile(tray_path):
        try:
            return Image.open(tray_path).convert('RGBA').resize((64, 64), Image.LANCZOS)
        except Exception:
            pass
    img = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    pixel_data = [
        (16, 8, 255, 50, 50), (18, 8, 255, 100, 50), (20, 8, 50, 255, 50),
        (22, 8, 50, 200, 100), (24, 8, 50, 50, 255), (26, 8, 100, 50, 200),
        (28, 8, 200, 50, 100), (14, 10, 255, 80, 80), (16, 10, 255, 150, 50),
        (18, 10, 100, 255, 100), (20, 10, 80, 200, 120), (22, 10, 80, 80, 255),
        (24, 10, 150, 50, 200), (26, 10, 200, 80, 150), (28, 10, 200, 100, 100),
        (30, 10, 150, 150, 50), (12, 12, 255, 100, 100), (14, 12, 255, 200, 80),
        (16, 12, 150, 255, 150), (18, 12, 100, 255, 200), (20, 12, 100, 100, 255),
        (22, 12, 200, 80, 255), (24, 12, 255, 100, 200), (26, 12, 255, 150, 100),
        (28, 12, 200, 200, 80), (30, 12, 150, 200, 100),
    ]
    for px, py, r, g, b in pixel_data:
        draw.rectangle([px, py, px + 3, py + 3], fill=(r, g, b, 255))
    draw.text((2, 52), '♢', fill=(150, 150, 255, 200))
    return img


def _setup_tray_icon_engine(app_ref: Any) -> Optional[Any]:
    """Создать иконку трея для HermesEngine."""
    import pystray
    try:
        image = _create_tray_image()
        menu = pystray.Menu(
            pystray.MenuItem('♢ Показать/Скрыть', lambda i, m: app_ref.toggle_window()),
            pystray.MenuItem('↕ Переместить', lambda i, m: app_ref._toggle_draggable()),
            pystray.MenuItem('💬 Ввод (C)', lambda i, m: app_ref.input_win.toggle()),
            pystray.MenuItem('🌠 Трейлы', lambda i, m: app_ref._toggle_trails()),
            pystray.MenuItem('⚙ Настройки', lambda i, m: app_ref.show_settings()),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem('✕ Выход', lambda i, m: app_ref.quit_app()),
        )
        icon = pystray.Icon('HermesCube', image, '♢ Hermes Cube', menu)
        if hasattr(icon, 'run_detached'):
            icon.run_detached()
        else:
            threading.Thread(target=icon.run, daemon=False).start()
        app_ref._tray_guid = 'HermesCube'
        return icon
    except Exception as e:
        print(f"Tray icon failed: {e}", flush=True)
        return None


TRANSPARENT_COLOR: str = '#000001'

# Гарантированная очистка трея при любом завершении
import atexit
atexit.register(_remove_tray_icon_force)


class HermesEngine:
    """Hermes Engine v2 — main application controller.

    Владеет жизненным циклом: окно → мир → пайплайн → рендер → выход.
    """

    def __init__(self) -> None:
        self.config: Dict[str, Any] = load_config()
        self.world: World = World.create(
            self.config,
            n_particles=self.config['particle_density'] ** 2 * 6,
        )

        # --- Window ---
        self.window = WindowSystem()
        self.canvas = self.window.canvas

        # --- Renderer ---
        self.renderer = PointCloudRenderer()
        self.renderer.attach(self.canvas)

        # --- Render Graph ---
        self.render_graph = RenderGraph()
        self.render_graph.add_pass(TrailPass())
        self.render_graph.add_pass(GeometryPass())

        # --- Pipeline ---
        self.pipeline = build_default_pipeline()

        # --- AI Systems ---
        self.text_overlay = TextOverlaySystem(self.window.root)
        self.input_win = InputWindowSystem(self.window.root)
        self.input_win.connect_world(self.world)

        # --- AISystem (chat) — добавляется в pipeline ---

        # --- State ---
        self.running: bool = True
        self.anim_running: bool = False
        self.t0: float = 0.0
        self.frame_count: int = 0
        self._show_hint: bool = True
        self._last_mood: str = 'idle'

        # Convex hull drag handle
        self._drag_handle: Optional[int] = None
        self._cube_ox: float = 0.0
        self._cube_oy: float = 0.0
        self.draggable: bool = False

        # Trails
        self._trail_enabled: bool = False

        # PixelGrid (legacy, kept for compatibility)
        self._pixel_anim_active: bool = False
        self._pixel_anim_frame: int = 0

        # Bindings (simplified)
        self.window.root.bind('<Button-1>', self._drag_start)
        self.window.root.bind('<B1-Motion>', self._drag_move)
        self.window.root.bind('<ButtonRelease-1>', self._drag_end)
        self.window.root.bind('<Escape>', lambda e: self.window.hide())
        self.window.root.bind('q', lambda e: self.window.hide())
        self.window.root.bind('h', lambda e: self.window.hide())
        self.window.root.bind('s', lambda e: self.show_settings())
        self.window.root.bind('c', lambda e: self.input_win.toggle())
        self.window.root.bind('C', lambda e: self.input_win.toggle())
        self.window.root.bind('t', lambda e: self._toggle_draggable())
        self.window.root.bind('T', lambda e: self._toggle_draggable())
        self.window.root.bind('r', lambda e: self._toggle_trails())
        self.window.root.bind('R', lambda e: self._toggle_trails())

        # Tray
        self.tray_icon: Optional[Any] = None
        threading.Thread(target=lambda: setattr(
            self, 'tray_icon', _setup_tray_icon_engine(self)), daemon=False).start()

    # ── Drag ─────────────────────────────────────────────────────────

    def _drag_start(self, event: tk.Event) -> None:
        if not self.draggable:
            return
        self._drag_grab_x = event.x_root
        self._drag_grab_y = event.y_root
        self._drag_start_ox = self._cube_ox
        self._drag_start_oy = self._cube_oy

    def _drag_move(self, event: tk.Event) -> None:
        if not self.draggable:
            return
        self._cube_ox = self._drag_start_ox + (event.x_root - self._drag_grab_x)
        self._cube_oy = self._drag_start_oy + (event.y_root - self._drag_grab_y)

    def _drag_end(self, event: tk.Event) -> None:
        pass

    def _toggle_draggable(self) -> None:
        self.draggable = not self.draggable
        self.window.set_clickthrough(not self.draggable)
        if self.draggable:
            if self._drag_handle is None:
                self._drag_handle = self.canvas.create_polygon(
                    0, 0, 0, 0, fill='#000000', outline='', width=0, tags='drag_handle',
                )
            self.canvas.itemconfig('drag_handle', state='normal')
        else:
            if self._drag_handle is not None:
                self.canvas.itemconfig('drag_handle', state='hidden')
        self._show_mode_overlay('↕ ДРАГ' if self.draggable else 'ПРОЗРАЧНЫЙ')

    def _show_mode_overlay(self, text: str) -> None:
        overlay = self.canvas.create_text(
            10, 10, anchor='nw', text=text,
            fill='#ffffff', font=('Segoe UI', 14, 'bold'), tags='mode_overlay',
        )
        self.window.root.after(1200, lambda: self.canvas.delete('mode_overlay'))

    def _toggle_trails(self) -> None:
        self._trail_enabled = not self._trail_enabled
        self.world.render.trail_enabled = self._trail_enabled
        if not self._trail_enabled:
            self.world.render.trail_history.clear()
        self._show_mode_overlay('ТРЕЙЛЫ ВКЛ' if self._trail_enabled else 'ТРЕЙЛЫ ВЫКЛ')

    # ── Settings ─────────────────────────────────────────────────────

    def show_settings(self) -> None:
        SettingsWindow(self)

    # ── Render loop ──────────────────────────────────────────────────

    def _render_loop(self) -> None:
        if not self.running:
            return

        now = self.window.root.tk.call('clock', 'milliseconds')
        if self.t0 == 0:
            self.t0 = now
            self.window.root.after(100, self._render_loop)
            return

        elapsed = (now - self.t0) / 1000.0
        w = self.window.w
        h = self.window.h

        if w < 10 or h < 10:
            self.window.root.after(FRAME_MS, self._render_loop)
            return

        # Update world
        self.world.meta.t = elapsed
        self.world.meta.w = w
        self.world.meta.h = h
        self.world.meta.cube_ox = self._cube_ox
        self.world.meta.cube_oy = self._cube_oy
        self.world.meta.config = self.config

        # Pipeline
        self.pipeline.run(self.world, FRAME_MS / 1000.0)

        n = self.world.sim.active_count
        px = self.world.render.projected_x[:n]
        py = self.world.render.projected_y[:n]
        pz = self.world.render.depth[:n]
        rgb_arr = self.world.render.final_rgb[:n]

        # Depth sort
        if n > 0:
            order = np.argsort(pz)
            px, py, rgb_arr = px[order], py[order], rgb_arr[order]

        # Convex hull
        if self._drag_handle is not None and self.draggable and n >= 3:
            xy = np.column_stack((px, py))
            hull = _convex_hull_2d(xy)
            if len(hull) >= 3:
                hull = _expand_hull(hull, pad=24.0)
                self.canvas.coords(self._drag_handle, *hull.ravel().tolist())
                self.canvas.tag_lower('drag_handle')
        elif self._drag_handle is not None:
            self.canvas.coords(self._drag_handle, 0, 0, 0, 0)

        # Render Graph
        ctx = FrameContext(
            px=px, py=py, pz=pz, rgb=rgb_arr,
            cell=max(MIN_CELL_SIZE, int(self.config.get('cell_size', MIN_CELL_SIZE))),
            symbol=self.config.get('symbol', 'square'),
            trail_enabled=self.world.render.trail_enabled,
            trail_layer=self.world.render.trail_layer,
            using_chars=False, char_list=None, symbols_set=[],
            config=self.config, w=w, h=h,
        )
        rgba_buf, bbox = self.render_graph.execute(ctx)
        if rgba_buf is not None and rgba_buf.size > 0:
            self.renderer.blit(rgba_buf, bbox[0], bbox[1])
        else:
            self.renderer.hide()

        # Hint overlay (first frame)
        if self._show_hint:
            hint = self.canvas.create_text(
                w // 2, h - 20,
                text='C — ввод  |  S — настройки  |  R — трейлы  |  H — скрыть',
                fill=UI_ACCENT, font=('Segoe UI', 9), anchor='center',
            )
            self.window.root.after(5000, lambda: self.canvas.delete(hint) if self.canvas.winfo_exists() else None)
            self._show_hint = False

        # AI text overlay
        self.text_overlay.update(self.world, FRAME_MS / 1000.0)

        # Mood change overlay
        if self.world.meta.mood != self._last_mood:
            self._last_mood = self.world.meta.mood
            labels = {'idle': '😐', 'thinking': '🤔', 'speaking': '💬', 'happy': '😊', 'sad': '😢'}
            self._show_mode_overlay(labels.get(self.world.meta.mood, ''))

        self.frame_count += 1
        self.window.root.after(FRAME_MS, self._render_loop)

    # ── Public interface (for tray) ──────────────────────────────────

    def toggle_window(self) -> None:
        self.window.toggle_visible()

    def quit_app(self) -> None:
        self.running = False
        save_config(self.config)

        # Stop tray icon FIRST (pystray), then force-remove via Win32
        if hasattr(self, 'tray_icon') and self.tray_icon is not None:
            try:
                self.tray_icon.stop()
            except Exception:
                pass
        _remove_tray_icon_force()

        self.text_overlay.close()
        self.window.destroy()
        os._exit(0)

    def run(self) -> None:
        """Запустить engine."""
        self.window.show()
        self.window.root.after(100, self._render_loop)
        self.window.root.mainloop()


if __name__ == '__main__':
    engine = HermesEngine()
    engine.run()
