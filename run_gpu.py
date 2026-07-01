#!/usr/bin/env python3
"""run_gpu.py — Minimal GPU Cube runner.

Запускает куб напрямую через OpenGL overlay, без Tk, без трея.
"""

import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['HERMES_LOCKED'] = '1'

from core.systems.gpu_window import GpuWindowSystem
from core.gpu import GpuRenderer
from core.world import World
from core.pipeline import build_default_pipeline
from cube_app import load_config

# Load config
cfg = load_config()

# Create GPU window (fullscreen transparent overlay)
win = GpuWindowSystem()
print(f"Window: {win.w}x{win.h}")

# Create GPU renderer (instanced quads)
gpu = GpuRenderer()
if not gpu.init_from_context(win.ctx):
    print("GPU init failed!")
    sys.exit(1)

# Create world + pipeline
world = World.create(cfg, n_particles=cfg['particle_density']**2 * 6)
pipeline = build_default_pipeline()
print(f"World: {world.sim.active_count} particles")

gpu.upload(world.sim.active_count)

# Show window
win.show()

# Render loop
try:
    frame = 0
    t0 = time.perf_counter()
    while True:
        t = time.perf_counter()
        elapsed = t - t0

        world.meta.t = elapsed
        world.meta.w = win.w
        world.meta.h = win.h
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

        win.make_current()
        gpu.render(px, py, pz, rgb_arr, win.w, win.h,
                   cell_size=max(2, int(cfg.get('cell_size', 6))))
        win.swap_buffers()
        win.pump_messages()
        frame += 1

        if frame % 300 == 0:
            fps = frame / (time.perf_counter() - t0)
            print(f"[{frame}] {fps:.0f} FPS, {n} particles", flush=True)

except KeyboardInterrupt:
    pass
finally:
    win.hide()
    win.destroy()
    gpu.destroy()
    print("Done.")
