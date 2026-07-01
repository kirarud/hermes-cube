#!/usr/bin/env python3
"""main.py — Hermes Engine v3 entry point (GPU → FBO → DIB overlay).

Архитектура:
  1. GPU рендерит instanced quads в FBO (moderngl через GpuWindowSystem)
  2. PBO readback → memmove → DIB → UpdateLayeredWindow (60 FPS)
  3. Win32 прозрачное окно с color-key — click-through, topmost
  4. Tk (withdrawn) — трей, настройки, горячие клавиши
  5. World — данные частиц
  6. Pipeline — Sim → FX → View
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

print("[main.py] import complete", flush=True)

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
from core.gpu import GpuRenderer

print("[main.py] imports ok", flush=True)

# Tk (withdrawn) для трея и настроек
os.environ['HERMES_LOCKED'] = '1'
from cube_app import (
    load_config, save_config, DEFAULT_CONFIG,
    MIN_CELL_SIZE, MAX_CELL_SIZE, FRAME_MS,
    _remove_tray_icon_force,
    SettingsWindow,
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


def _setup_tray_icon(app_ref: Any) -> Optional[Any]:
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
        return icon
    except Exception as e:
        print(f"Tray icon failed: {e}", flush=True)
        return None


# _remove_tray_icon_force уже зарегистрирован в cube_app.py при импорте


class HermesEngine:
    """Hermes Engine v3 — GPU → FBO → DIB overlay, Tk только для UI.

    Владеет: GpuWindowSystem (Win32 DIB overlay), Tk (трей/настройки),
    World, Pipeline, трей, SettingsWindow.
    """


    def __init__(self) -> None:
        print("[HermesEngine] init start", flush=True)
        self.config: Dict[str, Any] = load_config()
        print("[HermesEngine] config loaded", flush=True)

        # ── Размеры окна ─────────────────────────────────────────────
        self._win_w: int = self.config.get('window_width', 700)
        self._win_h: int = self.config.get('window_height', 550)
        win_x: int = self.config.get('x', 100)
        win_y: int = self.config.get('y', 100)

        # ── Tk root (withdrawn — трей, настройки, input) ─────────────
        self._tk_root = tk.Tk()
        self._tk_root.title('♢ Hermes Cube')
        self._tk_root.withdraw()
        print("[HermesEngine] tk root created (withdrawn)", flush=True)

        # ── GPU Window (DIB overlay) ─────────────────────────────────
        from core.systems.gpu_window import GpuWindowSystem
        self.win: GpuWindowSystem = GpuWindowSystem(
            width=self._win_w, height=self._win_h,
            x=win_x, y=win_y)
        print("[HermesEngine] gpu window created", flush=True)

        self._renderer = GpuRenderer()
        if not self._renderer.init_from_context(self.win.ctx):
            print("[GPU] Shader init failed!")
            sys.exit(1)
        print("[HermesEngine] gpu renderer init ok", flush=True)

        # ── Keyboard через GpuWindow ────────────────────────────────
        self.win.on_key = self._on_gl_key
        print("[HermesEngine] keyboard bindings ok", flush=True)

        # ── World ────────────────────────────────────────────────────
        n_particles = self.config['particle_density'] ** 2 * 6
        self.world: World = World.create(self.config, n_particles=n_particles)
        self._renderer.upload(self.world.sim.active_count)
        print(f"[HermesEngine] world created: {n_particles} particles", flush=True)

        # ── Pipeline ─────────────────────────────────────────────────
        self.pipeline = build_default_pipeline()
        print("[HermesEngine] pipeline ok", flush=True)

        # ── Render Graph ─────────────────────────────────────────────
        self.render_graph = RenderGraph()
        self.render_graph.add_pass(TrailPass())
        self.render_graph.add_pass(GeometryPass())
        print("[HermesEngine] render graph ok", flush=True)

        # ── AI Systems ───────────────────────────────────────────────
        self.text_overlay = TextOverlaySystem(self._tk_root)
        self.input_win = InputWindowSystem(self._tk_root)
        self.input_win.connect_world(self.world)
        print("[HermesEngine] ai systems ok", flush=True)

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
        print("[HermesEngine] state initialized", flush=True)

        # ── Tray ─────────────────────────────────────────────────────
        self.tray_icon: Optional[Any] = None
        threading.Thread(target=lambda: setattr(
            self, 'tray_icon', _setup_tray_icon(self)), daemon=False).start()
        print("[HermesEngine] tray thread started", flush=True)

        # ── Tk timers ────────────────────────────────────────────────
        self._tk_root.after(100, self._tk_tick)
        print("[HermesEngine] init done", flush=True)

    # ── GPU render (вызывается из _tk_tick) ──────────────────────────

    def _render_frame(self) -> None:
        _t0 = time.perf_counter_ns()

        t = time.perf_counter()
        w, h = self.win.w, self.win.h

        self.world.meta.t = t
        self.world.meta.w = w
        self.world.meta.h = h
        self.world.meta.cube_ox = self._cube_ox
        self.world.meta.cube_oy = self._cube_oy
        self.world.meta.config = self.config

        # Pipeline
        self.pipeline.run(self.world, 0.016)
        _t1 = time.perf_counter_ns()

        # Depth sort
        n = self.world.sim.active_count
        px = self.world.render.projected_x[:n]
        py = self.world.render.projected_y[:n]
        pz = self.world.render.depth[:n]
        rgb_arr = self.world.render.final_rgb[:n]
        if n > 0:
            order = np.argsort(pz)
            px, py, rgb_arr = px[order], py[order], rgb_arr[order]

        _t2 = time.perf_counter_ns()

        # GPU render → FBO → DIB overlay (через GpuWindowSystem)
        self.win.make_current()
        cell = max(MIN_CELL_SIZE, int(self.config.get('cell_size', MIN_CELL_SIZE)))
        self._renderer.render(px, py, pz, rgb_arr, w, h, cell_size=cell)
        self.win.swap_buffers()

        _t3 = time.perf_counter_ns()

        # Monitor (без HUD на canvas — всё через DIB)
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
            pipeline_us=(_t1 - _t0) / 1000,
            sort_us=(_t2 - _t1) / 1000,
            render_us=(_t3 - _t2) / 1000,
            total_us=(_t3 - _t0) / 1000,
            n_particles=self.world.sim.active_count,
            bbox=None,
            config=self.config,
        )

    # ── Tk loop ─────────────────────────────────────────────────────

    def _tk_tick(self) -> None:
        if not self.running:
            return
        try:
            self.win.pump_messages()
            self._render_frame()
            self.text_overlay.update(self.world, 0.016)
            self._tk_root.update_idletasks()
            if self.world.meta.mood != self._last_mood:
                self._last_mood = self.world.meta.mood
        except Exception as e:
            print(f"[tk_tick] error: {e}", flush=True)
            import traceback
            traceback.print_exc()
        self._tk_root.after(16, self._tk_tick)

    # ── Events ──────────────────────────────────────────────────────

    def _on_gl_key(self, keycode: int) -> None:
        if keycode == 0x1B:  # Escape
            self.win.hide()
        elif keycode == 0x48 or keycode == 0x51:  # H, Q
            self.win.hide()
        elif keycode == 0x53:  # S
            self.show_settings()
        elif keycode == 0x43:  # C
            self.input_win.toggle()
        elif keycode == 0x54:  # T
            self._toggle_draggable()
        elif keycode == 0x52:  # R
            self._toggle_trails()

    # ── Public interface ────────────────────────────────────────────

    def toggle_window(self) -> None:
        self.win.toggle_visible()

    def _hide_window(self) -> None:
        self.win.hide()

    def _toggle_draggable(self) -> None:
        self._draggable = not self._draggable
        self.win.set_clickthrough(not self._draggable)

    def _toggle_trails(self) -> None:
        self._trail_enabled = not self._trail_enabled
        self.world.render.trail_enabled = self._trail_enabled
        if not self._trail_enabled:
            self.world.render.trail_history.clear()

    def recalc(self, cfg: Dict[str, Any]) -> None:
        import core.systems.grid_generator as gg
        world = self.world
        old_count = world.sim.active_count
        world.meta.config = cfg
        gg.update(world, 0.042)
        if world.sim.active_count != old_count:
            self._renderer.upload(world.sim.active_count)

    def show_settings(self) -> None:
        """Показать SettingsWindow."""
        class AppProxy:
            __init__ = lambda s: None
            config = self.config
            root = self._tk_root
            engine = type('e', (), {'recalc': lambda s, c: self.recalc(c)})()
            _auto_resize_window = lambda: None
        SettingsWindow(AppProxy())

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
        self._renderer.destroy()
        self.win.destroy()
        try:
            self._tk_root.quit()
            self._tk_root.destroy()
        except Exception:
            pass

    def run(self) -> None:
        """Запустить engine: показать окно, войти в mainloop."""
        self.win.show()
        self._tk_root.after(100, self._tk_tick)
        print("♢ Hermes Cube (GPU → DIB overlay) — H/Q/Esc = скрыть, S = настройки", flush=True)
        self._tk_root.mainloop()


if __name__ == '__main__':
    engine = HermesEngine()
    engine.run()
