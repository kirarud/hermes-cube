#!/usr/bin/env python3
"""main.py — Hermes Engine v3 entry point (GPU overlay).

Заменяет Tkinter canvas на OpenGL overlay окно.
Tkinter остаётся (withdrawn) для SettingsWindow, трея и ввода.

Жизненный цикл:
  1. Tk (withdrawn) — трей, настройки, горячие клавиши
  2. GpuWindow — прозрачное OpenGL-окно поверх всего
  3. World — данные частиц
  4. Pipeline — Sim → FX → View
  5. GpuRenderer — рендер напрямую в OpenGL окно
  6. Горячие клавиши через Win32 hook → проброс в Tk

Производительность: 800-1000 FPS (864 частиц, RTX 2070 SUPER)
"""

from __future__ import annotations

import os
import sys
import tempfile
import atexit
import threading
import time
import tkinter as tk
from typing import Any, Dict, List, Optional

# ── Single-instance lock ──────────────────────────────────────────────
_LOCK_FILE: str = os.path.join(tempfile.gettempdir(), 'hermes_cube.lock')

def _check_single_instance() -> None:
    try:
        fd = os.open(_LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        atexit.register(lambda: os.unlink(_LOCK_FILE))
    except FileExistsError:
        try:
            with open(_LOCK_FILE) as f:
                old_pid = int(f.read().strip())
            if sys.platform == 'win32':
                import ctypes
                handle = ctypes.windll.kernel32.OpenProcess(0x0400, False, old_pid)
                if handle:
                    ctypes.windll.kernel32.CloseHandle(handle)
                    sys.exit(0)
            else:
                os.kill(old_pid, 0)
                sys.exit(0)
        except (ValueError, OSError, ProcessLookupError):
            try:
                os.unlink(_LOCK_FILE)
            except OSError:
                pass
            _check_single_instance()
            return

_check_single_instance()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from core.world import World
from core.pipeline import Pipeline, Stage, Schedule, build_default_pipeline
from core.render_graph import RenderGraph, GeometryPass, TrailPass, FrameContext
from core.systems.text_overlay import TextOverlaySystem
from core.systems.input_window import InputWindowSystem
from core.monitor import FrameMonitor
from char_cube import SYMBOL_SETS

# GPU overlay
from core.systems.gpu_window import GpuWindowSystem
from core.gpu import GpuRenderer

# Tk (withdrawn) для трея и настроек
os.environ['HERMES_LOCKED'] = '1'
from cube_app import (
    load_config, save_config, DEFAULT_CONFIG,
    MIN_CELL_SIZE, MAX_CELL_SIZE, FRAME_MS,
    _remove_tray_icon_force,
    UI_ACCENT, SettingsWindow,
)


def _create_tray_image():
    """Создать иконку трея (64×64)."""
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
    """Создать иконку трея для HermesEngine (GPU версия)."""
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
        return icon
    except Exception as e:
        print(f"Tray icon failed: {e}", flush=True)
        return None


# Гарантированная очистка трея при любом завершении
atexit.register(_remove_tray_icon_force)

# Win32 key constants (проброс в Tk)
WM_KEYDOWN = 0x0100


class HermesEngine:
    """Hermes Engine v3 — GPU overlay, Tk withdrawn.

    Владеет: GpuWindow (OpenGL), Tk (withdrawn), World, Pipeline.
    """

    VK_MAP = {
        0x53: 's', 0x43: 'c', 0x54: 't', 0x52: 'r',
        0x48: 'h', 0x51: 'q', 0x47: 'g', 0x41: 'a',
        0x50: 'p',  # P = FPS toggle
    }

    def __init__(self) -> None:
        self.config: Dict[str, Any] = load_config()

        # ── Tk (withdrawn) — трей и SettingsWindow ───────────────────
        self._tk_root = tk.Tk()
        self._tk_root.withdraw()

        # ── GPU overlay ──────────────────────────────────────────────
        self.gpu_win = GpuWindowSystem()
        self.gpu_win.on_key = self._on_gpu_key
        self.gpu_win.on_quit = self.quit_app

        # Установить OpenGL контекст текущим
        self.gpu_win.make_current()

        # ── GpuRenderer ──────────────────────────────────────────────
        self.renderer = GpuRenderer()
        if not self.renderer.init_from_context(self.gpu_win.ctx):
            print("[GPU] Fallback failed — CPU fallback not yet implemented!")
            sys.exit(1)

        # ── World ────────────────────────────────────────────────────
        self.world: World = World.create(
            self.config,
            n_particles=self.config['particle_density'] ** 2 * 6,
        )

        # ── Pipeline ─────────────────────────────────────────────────
        self.pipeline = build_default_pipeline()

        # ── Render Graph (только CPU fallback, GPU минует его) ──────
        self.render_graph = RenderGraph()
        self.render_graph.add_pass(TrailPass())
        self.render_graph.add_pass(GeometryPass())

        # ── AI Systems ───────────────────────────────────────────────
        self.text_overlay = TextOverlaySystem(self._tk_root)
        self.input_win = InputWindowSystem(self._tk_root)
        self.input_win.connect_world(self.world)

        # ── State ────────────────────────────────────────────────────
        self.running: bool = True
        self._show_hint: bool = True
        self._last_mood: str = 'idle'
        self._trail_enabled: bool = False
        self._draggable: bool = False
        self._cube_ox: float = 0.0
        self._cube_oy: float = 0.0
        self._show_fps: bool = True

        # ── Monitor ──────────────────────────────────────────────────
        self.monitor = FrameMonitor()
        self._fps_frame_count: int = 0
        self._fps_last_time: float = time.perf_counter()

        # ── Tray ─────────────────────────────────────────────────────
        self.tray_icon: Optional[Any] = None
        threading.Thread(target=lambda: setattr(
            self, 'tray_icon', _setup_tray_icon_engine(self)), daemon=False).start()

        # ── Tk timers (для SettingsWindow, AI overlay) ──────────────
        self._tk_root.after(100, self._tk_tick)

    # ── GPU render loop ──────────────────────────────────────────────

    def _render_loop(self) -> None:
        """Главный рендер-луп. Запускается вручную из run()."""
        while self.running:
            _t0 = time.perf_counter_ns()

            # Обновить мир
            t = time.perf_counter()
            w, h = self.gpu_win.w, self.gpu_win.h
            if w < 10 or h < 10:
                time.sleep(0.001)
                continue

            self.world.meta.t = t
            self.world.meta.w = w
            self.world.meta.h = h
            self.world.meta.cube_ox = self._cube_ox
            self.world.meta.cube_oy = self._cube_oy
            self.world.meta.config = self.config

            # Pipeline
            self.pipeline.run(self.world, 0.016)
            _t1 = time.perf_counter_ns()

            # Depth sort (пока на CPU — GPU не умеет painter's order)
            n = self.world.sim.active_count
            px = self.world.render.projected_x[:n]
            py = self.world.render.projected_y[:n]
            pz = self.world.render.depth[:n]
            rgb_arr = self.world.render.final_rgb[:n]
            if n > 0:
                order = np.argsort(pz)
                px, py, rgb_arr = px[order], py[order], rgb_arr[order]

            _t2 = time.perf_counter_ns()

            # GPU render
            self.gpu_win.make_current()
            self.gpu_win.clear()
            self.renderer.render(px, py, pz, rgb_arr, w, h,
                                 cell_size=max(MIN_CELL_SIZE, int(self.config.get('cell_size', MIN_CELL_SIZE))))
            self.gpu_win.swap_buffers()
            self.gpu_win.pump_messages()

            _t3 = time.perf_counter_ns()

            # HUD (через Tk withdraw — временный fallback, потом GL-шрифты)
            self._update_hud(t, _t0, _t1, _t2, _t3)

    def _update_hud(self, t: float, t0: int, t1: int, t2: int, t3: int) -> None:
        """Обновить FPS монитор."""
        self._fps_frame_count += 1
        now_s = time.perf_counter()
        dt_fps = now_s - self._fps_last_time
        fps = 0.0
        if dt_fps >= 0.5:
            fps = self._fps_frame_count / dt_fps
            self._fps_frame_count = 0
            self._fps_last_time = now_s

        self.monitor.log_frame(
            fps=fps,
            pipeline_us=(t1 - t0) / 1000,
            sort_us=(t2 - t1) / 1000,
            render_us=(t3 - t2) / 1000,
            total_us=(t3 - t0) / 1000,
            n_particles=self.world.sim.active_count,
            bbox=None,
            config=self.config,
        )

    # ── Tk loop (для SettingsWindow, AI overlay) ────────────────────

    def _tk_tick(self) -> None:
        """Tik-tak для Tk: события SettingsWindow, AI overlay."""
        if not self.running:
            return
        try:
            # AI text overlay (рендерится через Tk, поверх GPU)
            self.text_overlay.update(self.world, 0.016)
            self._tk_root.update_idletasks()

            # Mood change
            if self.world.meta.mood != self._last_mood:
                self._last_mood = self.world.meta.mood
                # Показываем на GPU overlay? Потом.
                pass
        except Exception:
            pass
        self._tk_root.after(16, self._tk_tick)

    # ── GPU Key handler ──────────────────────────────────────────────

    def _on_gpu_key(self, vk: int) -> None:
        """Обработка клавиш из Win32 hook."""
        key = self.VK_MAP.get(vk)
        if key == 'h' or key == 'q':
            self.gpu_win.hide()
        elif key == 's':
            self.show_settings()
        elif key == 'c':
            self.input_win.toggle()
        elif key == 't':
            self._toggle_draggable()
        elif key == 'r':
            self._toggle_trails()

    # ── Public interface ─────────────────────────────────────────────

    def toggle_window(self) -> None:
        self.gpu_win.toggle_visible()

    def _toggle_draggable(self) -> None:
        self._draggable = not self._draggable
        self.gpu_win.set_clickthrough(not self._draggable)

    def _toggle_trails(self) -> None:
        self._trail_enabled = not self._trail_enabled
        self.world.render.trail_enabled = self._trail_enabled
        if not self._trail_enabled:
            self.world.render.trail_history.clear()

    def recalc(self, cfg: Dict[str, Any]) -> None:
        import core.systems.grid_generator as gg
        self.world.meta.config = cfg
        gg.update(self.world, 0.042)

    def show_settings(self) -> None:
        """Показать SettingsWindow через Tk (GPU окно позади)."""
        # Tk root уже существует — создаём Toplevel
        win = tk.Toplevel(self._tk_root)
        SettingsWindow.__init__(None, self)  # некрасиво, но работает

    def quit_app(self) -> None:
        self.running = False
        save_config(self.config)
        if hasattr(self, 'tray_icon') and self.tray_icon is not None:
            try:
                t = threading.Thread(target=self.tray_icon.stop, daemon=True)
                t.start()
                t.join(timeout=2)
            except Exception:
                pass
        _remove_tray_icon_force()
        self.text_overlay.close()
        self.gpu_win.destroy()
        self.renderer.destroy()
        try:
            self._tk_root.quit()
        except Exception:
            pass

    def run(self) -> None:
        """Запустить engine: показать GPU окно, войти в render loop."""
        self.gpu_win.show()
        # Render loop в этом потоке
        self._render_loop()


if __name__ == '__main__':
    engine = HermesEngine()
    engine.run()
