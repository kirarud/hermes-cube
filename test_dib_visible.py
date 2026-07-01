#!/usr/bin/env python3
"""Test DIB overlay visibility — яркое окно 400x300 на 60 секунд."""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
os.environ['HERMES_LOCKED'] = '1'

from core.systems.gpu_window import GpuWindowSystem
from core.gpu import GpuRenderer

# Окно 400x300 в центре экрана (pos 400,200)
win = GpuWindowSystem(width=400, height=300, x=400, y=200, clickthrough=False)
print(f"hwnd={win._hwnd}", flush=True)

gpu = GpuRenderer()
gpu.init_from_context(win.ctx)

# Ярко-красные частицы в центре
n = 500
px = np.full(n, 200.0, dtype=np.float64)
py = np.full(n, 150.0, dtype=np.float64)
pz = np.zeros(n, dtype=np.float64)
rgb = np.full((n, 3), [255, 30, 30], dtype=np.uint8)  # ярко-красный
# добавим жёлтые и зелёные
rgb[100:200] = [30, 255, 30]
rgb[200:300] = [255, 255, 30]

gpu.upload(n)
win.show()
print("OK — window should be visible at (400,200) 400x300 with colored dots", flush=True)
print("Running for 60 seconds...", flush=True)

t_start = time.perf_counter()
frame = 0
while time.perf_counter() - t_start < 60:
    win.make_current()
    gpu.render(px, py, pz, rgb, win.w, win.h, cell_size=20)
    win.swap_buffers()
    win.pump_messages()
    frame += 1
    if frame % 100 == 0:
        print(f"Frame {frame}", flush=True)
    time.sleep(0.001)

win.hide()
gpu.destroy()
win.destroy()
print("DONE", flush=True)
