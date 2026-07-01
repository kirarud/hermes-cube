#!/usr/bin/env python3
"""run.py — Hermes Cube (GPU + Tk).

GPU рендерит частицы в FBO (instanced quads).
Tk выводит на экран через прозрачный оверлей.
"""

from __future__ import annotations
import os, sys, time, numpy as np, tkinter as tk
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['HERMES_LOCKED'] = '1'

from PIL import Image, ImageTk
import moderngl

from core.gpu import GpuRenderer
from core.world import World
from core.pipeline import build_default_pipeline
from cube_app import load_config, TRANSPARENT_COLOR, UI_ACCENT, MIN_CELL_SIZE, SettingsWindow, save_config

# ── Config ────────────────────────────────────────────────────────────
cfg: Dict[str, Any] = load_config()
W, H = 700, 550  # размер окна

# ── GPU ────────────────────────────────────────────────────────────────
gl_ctx = moderngl.create_standalone_context(require=330)
gpu = GpuRenderer()
gpu.init_from_context(gl_ctx)
fbo = gl_ctx.simple_framebuffer((W, H))

# Буфер для readback — один раз, переиспользуется
rgba_buf = np.zeros((H, W, 4), dtype=np.uint8)
pil_img = Image.frombuffer('RGBA', (W, H), rgba_buf, 'raw', 'RGBA', 0, 1)
tk_photo: Optional[ImageTk.PhotoImage] = None

# ── World + Pipeline ──────────────────────────────────────────────────
world = World.create(cfg, n_particles=cfg['particle_density'] ** 2 * 6)
pipeline = build_default_pipeline()
gpu.upload(world.sim.active_count)
print(f"World: {world.sim.active_count} particles", flush=True)

# ── Tk Window ─────────────────────────────────────────────────────────
root = tk.Tk()
root.title('♢ Hermes Cube')
root.geometry(f'{W}x{H}+100+100')
root.overrideredirect(True)
root.configure(bg=TRANSPARENT_COLOR)
root.attributes('-transparentcolor', TRANSPARENT_COLOR)
root.attributes('-topmost', True)

canvas = tk.Canvas(root, bg=TRANSPARENT_COLOR, highlightthickness=0)
canvas.pack(fill=tk.BOTH, expand=True)

# ── Hotkeys ───────────────────────────────────────────────────────────
def on_key(e: tk.Event) -> None:
    k = e.keysym.lower()
    if k in ('escape', 'q', 'h'): root.withdraw()
    elif k == 's':
        show_settings()
    elif k == 'c': print("[input] C pressed")

def show_settings() -> None:
    """Открыть SettingsWindow через прокси-объект."""
    class AppProxy:
        def __init__(self):
            self.config = cfg
            self.root = root
            self._auto_resize_window = lambda: None
            eng = type('engine', (), {})()
            eng.recalc = lambda c: update_config(c)
            self.engine = eng
    SettingsWindow(AppProxy())

root.bind('<Escape>', on_key)
root.bind('q', on_key)
root.bind('h', on_key)
root.bind('s', lambda e: (root.deiconify(), root.lift(), root.attributes('-topmost', True)))

# ── Render loop ───────────────────────────────────────────────────────
running = True
frame = 0
t0 = time.perf_counter()

def update_config(new_cfg: Dict[str, Any]) -> None:
    """Обновить конфиг и перестроить мир если density изменился."""
    global cfg
    old_density = cfg.get('particle_density', 12)
    cfg = new_cfg
    new_density = new_cfg.get('particle_density', 12)
    if new_density != old_density:
        import core.systems.grid_generator as gg
        core.systems.grid_generator # suppress unused
        world.meta.config = cfg
        gg.update(world, 0.042)
        n = world.sim.active_count
        gpu.upload(n)

def tick() -> None:
    global frame, tk_photo
    if not running:
        return

    _t0 = time.perf_counter_ns()
    t = time.perf_counter() - t0

    world.meta.t = t
    world.meta.w = W
    world.meta.h = H
    world.meta.cube_ox = 0.0
    world.meta.cube_oy = 0.0
    world.meta.config = cfg
    pipeline.run(world, 0.016)

    n = world.sim.active_count
    px = world.render.projected_x[:n]
    py = world.render.projected_y[:n]
    pz = world.render.depth[:n]
    rgb_arr = world.render.final_rgb[:n]
    if n > 0:
        order = np.argsort(pz)
        px, py, rgb_arr = px[order], py[order], rgb_arr[order]

    # GPU render → FBO
    fbo.use()
    gl_ctx.clear(0.0, 0.0, 1.0 / 255.0, 0.0)
    cell = max(MIN_CELL_SIZE, int(cfg.get('cell_size', 6)))
    gpu.render(px, py, pz, rgb_arr, W, H, cell_size=cell)

    # Readback FBO → PIL → Tk canvas
    fbo_data = fbo.read(components=4)
    arr = np.frombuffer(fbo_data, dtype=np.uint8, count=W * H * 4).reshape((H, W, 4))
    rgba_buf[:] = arr

    if tk_photo is None:
        tk_photo = ImageTk.PhotoImage(pil_img)
    else:
        tk_photo.paste(pil_img)
    canvas.create_image(0, 0, anchor='nw', image=tk_photo)

    # HUD
    frame += 1
    if frame % 30 == 0:
        fps = frame / (time.perf_counter() - t0)
        tot_us = (time.perf_counter_ns() - _t0) / 1000
        canvas.delete('hud')
        txt = f'FPS:{fps:.0f}  ptcl:{n}  frame:{tot_us:.0f}µs'
        canvas.create_text(8, 8, anchor='nw', text=txt,
                           fill=UI_ACCENT, font=('Consolas', 9), tags='hud')

    root.update_idletasks()
    root.after(16, tick)  # ~60 FPS

# ── Запуск ────────────────────────────────────────────────────────────
root.deiconify()
root.lift()
root.after(100, tick)
print("Running — окно открыто. H = скрыть, S = показать, Esc = скрыть", flush=True)
root.mainloop()

# ── Cleanup ───────────────────────────────────────────────────────────
running = False
gpu.destroy()
print("Done.", flush=True)
