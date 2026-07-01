#!/usr/bin/env python3
"""main.py — Hermes Engine v3 (GPU → FBO → PPM → Tk PhotoImage)."""

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
import moderngl

from core.world import World
from core.pipeline import Pipeline, Stage, Schedule, build_default_pipeline
from core.systems.text_overlay import TextOverlaySystem
from core.systems.input_window import InputWindowSystem
from core.monitor import FrameMonitor
from core.gpu import GpuRenderer

print("[main.py] imports ok", flush=True)

os.environ['HERMES_LOCKED'] = '1'
from cube_app import (
    load_config, save_config,
    MIN_CELL_SIZE, MAX_CELL_SIZE,
    TRANSPARENT_COLOR, UI_ACCENT,
    _remove_tray_icon_force,
    SettingsWindow,
)


def _create_tray_image():
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


class HermesEngine:
    def __init__(self) -> None:
        print("[HermesEngine] init start", flush=True)
        self.config: Dict[str, Any] = load_config()
        print("[HermesEngine] config loaded", flush=True)

        w = self.config.get('window_width', 700)
        h = self.config.get('window_height', 550)
        x = self.config.get('x', 100)
        y = self.config.get('y', 100)

        self._tk_root = tk.Tk()
        self._tk_root.title('♢ Hermes Cube')
        self._tk_root.geometry(f'{w}x{h}+{x}+{y}')
        self._tk_root.overrideredirect(True)
        self._tk_root.configure(bg=TRANSPARENT_COLOR)
        self._tk_root.attributes('-transparentcolor', TRANSPARENT_COLOR)
        self._tk_root.attributes('-topmost', True)
        self._tk_root.resizable(True, True)
        print("[HermesEngine] tk window created", flush=True)

        self._canvas = tk.Canvas(
            self._tk_root, bg=TRANSPARENT_COLOR, highlightthickness=0)
        self._canvas.pack(fill=tk.BOTH, expand=True)
        print("[HermesEngine] canvas created", flush=True)

        self._gl_ctx = moderngl.create_standalone_context(require=330)
        self._fbo = self._gl_ctx.simple_framebuffer((w, h))
        print("[HermesEngine] gl context ok", flush=True)

        self._renderer = GpuRenderer()
        if not self._renderer.init_from_context(self._gl_ctx):
            print("[GPU] Shader init failed!")
            sys.exit(1)
        print("[HermesEngine] gpu renderer ok", flush=True)

        # Font atlas (только при старте)
        from core.font_atlas import build_atlas
        atlas_rgba, char_maps = build_atlas(font_size=14)
        self._renderer.load_font_atlas(atlas_rgba, char_maps)
        self._char_map = char_maps
        print("[HermesEngine] font atlas loaded", flush=True)

        self._rgb_arr = np.empty((h, w, 3), dtype=np.uint8)
        self._ppm_header = f'P6\n{w} {h}\n255\n'.encode()
        self._tk_photo: Optional[tk.PhotoImage] = None
        self._canvas_image: Optional[int] = None
        print("[HermesEngine] ppm buffers ok", flush=True)

        n_particles = self.config['particle_density'] ** 2 * 6
        self.world: World = World.create(self.config, n_particles=n_particles)
        self._renderer.upload(self.world.sim.active_count)
        print(f"[HermesEngine] world: {n_particles} particles", flush=True)

        self.pipeline = build_default_pipeline()
        print("[HermesEngine] pipeline ok", flush=True)

        # ── AI Modules ───────────────────────────────────────────────
        from core.systems.ai import AISystem
        from core.systems.mood import MoodSystem
        from core.systems.lm_autostart import LMAutoStartSystem
        from core.systems.avatar_text import AvatarTextSystem
        self.ai_system = AISystem()
        self.mood_system = MoodSystem()
        self.lm_autostart = LMAutoStartSystem()
        self.avatar_text = AvatarTextSystem()
        print("[HermesEngine] ai modules ok", flush=True)

        self.text_overlay = TextOverlaySystem(self._tk_root)
        self.input_win = InputWindowSystem(self._tk_root)
        self.input_win.connect_world(self.world)
        print("[HermesEngine] ai systems ok", flush=True)

        self.running: bool = True
        self._last_mood: str = 'idle'
        self._trail_enabled: bool = False
        self._draggable: bool = False
        self._cube_ox: float = 0.0
        self._cube_oy: float = 0.0
        self._show_fps: bool = True

        self.monitor = FrameMonitor()
        self._fps_count: int = 0
        self._fps_time: float = time.perf_counter()
        print("[HermesEngine] state ok", flush=True)

        self.tray_icon: Optional[Any] = None
        threading.Thread(target=lambda: setattr(
            self, 'tray_icon', _setup_tray_icon(self)), daemon=False).start()
        print("[HermesEngine] tray started", flush=True)

        self._tk_root.bind('<Configure>', self._on_resize)
        self._tk_root.bind('<Escape>', lambda e: self._hide_window())
        self._tk_root.bind('q', lambda e: self._hide_window())
        self._tk_root.bind('h', lambda e: self._hide_window())
        self._tk_root.bind('s', lambda e: self.show_settings())
        self._tk_root.bind('c', lambda e: self.input_win.toggle())
        self._tk_root.bind('C', lambda e: self.input_win.toggle())
        self._tk_root.bind('t', lambda e: self._toggle_draggable())
        self._tk_root.bind('T', lambda e: self._toggle_draggable())
        self._tk_root.bind('r', lambda e: self._toggle_trails())
        self._tk_root.bind('R', lambda e: self._toggle_trails())
        # Drag
        self._tk_root.bind('<Button-1>', self._drag_start)
        self._tk_root.bind('<B1-Motion>', self._drag_move)
        self._tk_root.bind('<ButtonRelease-1>', self._drag_end)
        print("[HermesEngine] bindings ok", flush=True)

        self._tk_root.protocol('WM_DELETE_WINDOW', self._hide_window)

        # ── HTTP API ────────────────────────────────────────────────
        from api_server import start_api_server
        start_api_server(self, port=8081)
        print("[HermesEngine] api server started", flush=True)

        self._tk_root.after(100, self._tk_tick)
        print("[HermesEngine] init done", flush=True)

    def _render_frame(self) -> None:
        _t0 = time.perf_counter_ns()
        t = time.perf_counter()
        w, h = self._tk_root.winfo_width(), self._tk_root.winfo_height()

        self.world.meta.t = t
        self.world.meta.w = w
        self.world.meta.h = h
        self.world.meta.cube_ox = self._cube_ox
        self.world.meta.cube_oy = self._cube_oy
        self.world.meta.config = self.config
        self.pipeline.run(self.world, 0.016)
        _t1 = time.perf_counter_ns()

        n = self.world.sim.active_count
        px = self.world.render.projected_x[:n]
        py = self.world.render.projected_y[:n]
        pz = self.world.render.depth[:n]
        rgb_arr = self.world.render.final_rgb[:n]
        if n > 0:
            o = np.argsort(pz)
            px, py, rgb_arr = px[o], py[o], rgb_arr[o]
        _t2 = time.perf_counter_ns()
        # GPU render → FBO
        self._fbo.use()
        self._gl_ctx.clear(0.0, 0.0, 1.0 / 255.0, 0.0)
        cell = max(MIN_CELL_SIZE, int(self.config.get('cell_size', MIN_CELL_SIZE)))

        char_mode: str = self.config.get('char_mode', 'dots')
        using_chars = char_mode != 'dots'

        if using_chars:
            if self.world.meta.text_mode:
                # Аватар-режим: per-particle индексы из symbol_idx
                per_particle_indices = self.world.sim.symbol_idx[:n]
            else:
                # Обычный режим: циклический выбор из symbol_set
                symbol_set_name: str = self.config.get('symbol_set', 'default')
                char_indices_arr = self._char_map.get(symbol_set_name)
                if char_indices_arr is None:
                    char_indices_arr = self._char_map.get('default', np.array([0], dtype=np.int32))
                n_symbols = len(char_indices_arr)
                if n_symbols > 0:
                    sym_idx = np.arange(n, dtype=np.int32) % n_symbols
                    per_particle_indices = char_indices_arr[sym_idx]
                else:
                    per_particle_indices = np.zeros(n, dtype=np.int32)

            self._renderer.render(
                px, py, pz, rgb_arr, w, h, cell_size=cell,
                use_chars=True, char_indices=per_particle_indices,
            )

            # Трейлы (через dots, маленькие)
            if self._trail_enabled and self.world.render.trail_layer is not None:
                tx, ty, trgb = self.world.render.trail_layer
                if len(tx) > 0:
                    t_depth = np.zeros(len(tx), dtype=np.float64)
                    self._renderer.render(tx, ty, t_depth, trgb, w, h, cell_size=1)
        else:
            self._renderer._symbol = self.config.get('symbol', 'circle')
            self._renderer.render(px, py, pz, rgb_arr, w, h, cell_size=cell)

            # Трейлы
            if self._trail_enabled and self.world.render.trail_layer is not None:
                tx, ty, trgb = self.world.render.trail_layer
                if len(tx) > 0:
                    t_depth = np.zeros(len(tx), dtype=np.float64)
                    self._renderer.render(tx, ty, t_depth, trgb, w, h, cell_size=1)

        # Readback → PPM → Tk (общий для обоих режимов)
        fbo_data = self._fbo.read(components=4)
        arr = np.frombuffer(fbo_data, dtype=np.uint8).reshape((h, w, 4))
        self._rgb_arr[:, :, 0] = arr[:, :, 0]
        self._rgb_arr[:, :, 1] = arr[:, :, 1]
        self._rgb_arr[:, :, 2] = arr[:, :, 2]
        ppm_data = self._ppm_header + self._rgb_arr.tobytes()
        self._tk_photo = tk.PhotoImage(data=ppm_data, format='ppm')
        if self._canvas_image is None:
            self._canvas_image = self._canvas.create_image(
                0, 0, anchor='nw', image=self._tk_photo)
        else:
            self._canvas.itemconfig(self._canvas_image, image=self._tk_photo)

        _t3 = time.perf_counter_ns()

        self._fps_count += 1
        now_s = time.perf_counter()
        dt_fps = now_s - self._fps_time
        fps = 0.0
        if dt_fps >= 0.5:
            fps = self._fps_count / dt_fps
            self._fps_count = 0
            self._fps_time = now_s

        self.monitor.log_frame(
            fps=fps,
            pipeline_us=(_t1 - _t0) / 1000,
            sort_us=(_t2 - _t1) / 1000,
            render_us=(_t3 - _t2) / 1000,
            total_us=(_t3 - _t0) / 1000,
            n_particles=n,
            config=self.config,
        )

        if self._show_fps:
            self._canvas.delete('hud')
            tot_us = (_t3 - _t0) / 1000
            txt = f'FPS:{fps:.0f}  ptcl:{n}  frame:{tot_us:.0f}µs'
            self._canvas.create_text(
                8, 8, anchor='nw', text=txt,
                fill=UI_ACCENT, font=('Consolas', 9), tags='hud')

    def _tk_tick(self) -> None:
        if not self.running:
            return
        try:
            self._render_frame()
            self._tk_root.update_idletasks()

            # AI: сначала запрос → ответ → настроение → только потом text_overlay
            if self.lm_autostart:
                self.lm_autostart.update(self.world, 0.016)
            if self.ai_system:
                self.ai_system.update(self.world, 0.016)
            if self.mood_system:
                self.mood_system.update(self.world, 0.016)

            # Avatar text mode — отображение ответа частицами
            if self.avatar_text:
                self.avatar_text.update(self.world, 0.016)

            self.text_overlay.update(self.world, 0.016)
        except Exception as e:
            print(f"[tk_tick] {e}", flush=True)
            import traceback
            traceback.print_exc()
        self._tk_root.after(16, self._tk_tick)

    def _on_resize(self, event: tk.Event) -> None:
        if event.widget == self._tk_root:
            w, h = event.width, event.height
            if w != self._rgb_arr.shape[1] or h != self._rgb_arr.shape[0]:
                if w > 50 and h > 50:
                    self._resize_buffers(w, h)

    def _resize_buffers(self, w: int, h: int) -> None:
        self._fbo = self._gl_ctx.simple_framebuffer((w, h))
        self._rgb_arr = np.empty((h, w, 3), dtype=np.uint8)
        self._ppm_header = f'P6\n{w} {h}\n255\n'.encode()
        self._tk_photo = None
        self._canvas_image = None

    def toggle_window(self) -> None:
        if self._tk_root.state() == 'withdrawn':
            self._tk_root.deiconify()
            self._tk_root.lift()
            self._tk_root.attributes('-topmost', True)
        else:
            self._tk_root.withdraw()

    def _hide_window(self) -> None:
        self._tk_root.withdraw()

    def _toggle_draggable(self) -> None:
        self._draggable = not self._draggable
        if sys.platform == 'win32':
            import ctypes
            hwnd = ctypes.c_void_p(self._tk_root.winfo_id())
            user32 = ctypes.windll.user32
            ex = user32.GetWindowLongW(hwnd, -20)
            ex = ex & ~0x00000020 if self._draggable else ex | 0x00000020
            user32.SetWindowLongW(hwnd, -20, ex)
            user32.InvalidateRect(hwnd, None, True)

    def _drag_start(self, event):
        self._drag_x = event.x_root
        self._drag_y = event.y_root

    def _drag_move(self, event):
        dx = event.x_root - self._drag_x
        dy = event.y_root - self._drag_y
        self._drag_x = event.x_root
        self._drag_y = event.y_root
        x = self._tk_root.winfo_x() + dx
        y = self._tk_root.winfo_y() + dy
        self._tk_root.geometry(f'+{x}+{y}')

    def _drag_end(self, event):
        pass

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
        """Показать SettingsWindow (синглтон)."""
        if hasattr(self, '_settings_win') and self._settings_win is not None:
            try:
                self._settings_win.window.lift()
                self._settings_win.window.focus_set()
                return
            except Exception:
                pass
        class AppProxy:
            __init__ = lambda s: None
            config = self.config
            root = self._tk_root
            engine = type('e', (), {'recalc': lambda s, c: self.recalc(c)})()
            _auto_resize_window = lambda s: None
        win = SettingsWindow(AppProxy())
        self._settings_win = win
        # Cleanup ref on close
        def on_close(*_):
            self._settings_win = None
        win.window.protocol('WM_DELETE_WINDOW', lambda: (win.window.destroy(), on_close()))
        win.window.bind('<Destroy>', on_close, add='+')

    def quit_app(self) -> None:
        self.running = False
        save_config(self.config)
        self.config['x'] = self._tk_root.winfo_x()
        self.config['y'] = self._tk_root.winfo_y()
        self.config['window_width'] = self._tk_root.winfo_width()
        self.config['window_height'] = self._tk_root.winfo_height()
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
        try:
            self._tk_root.quit()
            self._tk_root.destroy()
        except Exception:
            pass

    def run(self) -> None:
        self._tk_root.deiconify()
        self._tk_root.lift()
        self._tk_root.after(100, self._tk_tick)
        print("♢ Hermes Cube (GPU → PPM → Tk) — H = скрыть, S = настройки", flush=True)
        self._tk_root.mainloop()


if __name__ == '__main__':
    engine = HermesEngine()
    engine.run()
