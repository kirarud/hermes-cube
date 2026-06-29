#!/usr/bin/env python3
"""core/gpu.py — GPU-рендерер для OpenGL overlay (moderngl v2).

Заменяет PointCloudRenderer при наличии GPU.

Ключевые решения:
  - Рисует через gl_PointSize с gl_PointCoord (круг/квадрат в шейдере)
  - Прямой рендер в OpenGL overlay окно — SwapBuffers, никаких read-back
  - Нуль PIL, нуль PhotoImage, нуль копий
  - CPU-fallback если GPU недоступен
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Any, Optional


# ═══════════════════════════════════════════════════════════════════════════
# Шейдеры
# ═══════════════════════════════════════════════════════════════════════════

VERTEX_SHADER = """
#version 330

in vec2 in_position;
in vec3 in_color;
in float in_depth;

uniform vec2 u_viewport;  // (width, height) в пикселях
uniform vec2 u_offset;    // смещение куба в пикселях
uniform float u_scale;    // масштаб куба
uniform float u_cell;     // размер точки в пикселях

out vec3 v_color;
out float v_depth;

void main() {
    // projected_x/y уже в пикселях: offset + scale уже применены
    // Конвертим пиксели → NDC [-1, 1]
    vec2 ndc = (in_position) / u_viewport * 2.0 - 1.0;
    // flip Y для Tk-совместимости
    ndc.y = -ndc.y;
    gl_Position = vec4(ndc, 0.0, 1.0);
    gl_PointSize = u_cell;

    v_color = in_color;
    v_depth = in_depth;
}
"""

FRAGMENT_SHADER = """
#version 330

in vec3 v_color;
in float v_depth;

out vec4 f_color;

void main() {
    // Круг через gl_PointCoord
    vec2 center = gl_PointCoord - vec2(0.5);
    float dist = length(center);
    if (dist > 0.5) {
        discard;  // прозрачные углы квадрата
    }

    // Depth shading
    float depth_factor = 0.6 + 0.4 * (v_depth + 1.0) / 2.0;

    // Цвет с глубиной, alpha=1 (цвет key #000001 невидим)
    f_color = vec4(v_color * depth_factor / 255.0, 1.0);
}
"""


# ═══════════════════════════════════════════════════════════════════════════
# GpuRenderer
# ═══════════════════════════════════════════════════════════════════════════

class GpuRenderer:
    """GPU-рендерер для облака частиц.

    Используется с GpuWindowSystem — рисует напрямую в OpenGL overlay.
    """

    # Максимальное количество частиц (VBO pre-alloc под это)
    MAX_PARTICLES: int = 32768

    def __init__(self) -> None:
        self._ctx: Any = None
        self._program: Any = None
        self._vbo: Any = None
        self._vao: Any = None

        self._initialized: bool = False
        self._fallback: bool = True
        self._n_capacity: int = 0

    @property
    def available(self) -> bool:
        return self._initialized and not self._fallback

    # ── Инициализация из существующего контекста ─────────────────────

    def init_from_context(self, ctx: Any) -> bool:
        """Создать шейдеры и VBO из moderngl контекста GpuWindowSystem.

        Args:
            ctx: moderngl.Context из GpuWindowSystem.ctx
        """
        try:
            self._ctx = ctx
            self._program = ctx.program(
                vertex_shader=VERTEX_SHADER,
                fragment_shader=FRAGMENT_SHADER,
            )
            self._initialized = True
            self._fallback = False
            return True
        except Exception as e:
            print(f"[GPU] Shader init failed: {e}")
            self._initialized = False
            self._fallback = True
            return False

    # ── VBO management ────────────────────────────────────────────────

    def upload(self, n_particles: int) -> bool:
        """Аллоцировать VBO на N частиц. Перевызвать при resize."""
        if self._fallback or not self._initialized:
            return False
        if n_particles == self._n_capacity and self._vbo is not None:
            return True

        n = min(n_particles, self.MAX_PARTICLES)
        # Interleaved: pos(2f) + color(3u1) + depth(1f) = 16 байт/частицу
        dtype = np.dtype([
            ('pos', 'f4', 2),
            ('col', 'u1', 3),
            ('depth', 'f4'),
        ])
        data = np.zeros(n, dtype=dtype)
        self._vbo = self._ctx.buffer(data.tobytes())
        self._vao = self._ctx.vertex_array(
            self._program,
            [(self._vbo, '2f 3u1 f', 'in_position', 'in_color', 'in_depth')],
        )
        self._n_capacity = n
        return True

    def update_vbo(self, data: bytes) -> None:
        if self._vbo is not None:
            self._vbo.write(data)

    # ── Рендер одного кадра ──────────────────────────────────────────

    def render(
        self,
        projected_x: NDArray[np.float64],
        projected_y: NDArray[np.float64],
        depth: NDArray[np.float64],
        rgb: NDArray[np.uint8],
        view_width: int,
        view_height: int,
        cell_size: int = 6,
    ) -> None:
        """Нарисовать частицы в текущий OpenGL контекст (FBO или окно).

        Ничего не возвращает — рисует напрямую в OpenGL буфер кадра.
        После render() нужно вызвать swap_buffers().
        """
        if self._fallback or not self._initialized or self._ctx is None:
            return

        n = len(projected_x)
        if n == 0:
            return

        # Resize VBO если нужно
        if n > self._n_capacity:
            self.upload(n)

        # Pack interleaved data
        dtype = np.dtype([
            ('pos', 'f4', 2),
            ('col', 'u1', 3),
            ('depth', 'f4'),
        ])
        data = np.empty(n, dtype=dtype)
        data['pos'][:, 0] = projected_x[:n].astype(np.float32)
        data['pos'][:, 1] = projected_y[:n].astype(np.float32)
        data['col'] = rgb[:n]
        data['depth'] = depth[:n].astype(np.float32)

        self.update_vbo(data.tobytes())

        # Uniforms
        self._program['u_viewport'] = (float(view_width), float(view_height))
        self._program['u_cell'] = float(cell_size)

        # Draw
        self._vao.render(mode=self._ctx.POINTS, vertices=n)

    # ── Освобождение ─────────────────────────────────────────────────

    def destroy(self) -> None:
        for obj in ('_vao', '_vbo', '_program'):
            try:
                o = getattr(self, obj, None)
                if o is not None:
                    o.release()
            except Exception:
                pass
        self._initialized = False
        self._fallback = True


# ═══════════════════════════════════════════════════════════════════════════
# Самотестирование
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import time
    import sys
    sys.path.insert(0, '.')
    from core.systems.gpu_window import GpuWindowSystem

    # Создаём окно
    win = GpuWindowSystem()
    win.make_current()

    # Создаём рендерер
    gpu = GpuRenderer()
    assert gpu.init_from_context(win.ctx), "GpuRenderer init failed"

    # Тестовые частицы
    n = 864
    rng = np.random.default_rng(0)
    px = rng.uniform(100, 1100, n).astype(np.float64)
    py = rng.uniform(100, 700, n).astype(np.float64)
    pz = rng.uniform(-1, 1, n).astype(np.float64)
    rgb = rng.integers(0, 256, (n, 3), dtype=np.uint8)

    gpu.upload(n)

    win.show()
    print(f"Window: {win.w}x{win.h}")
    print(f"Rendering {n} particles at ~60 FPS for 3 seconds...")

    t0 = time.perf_counter()
    frames = 0
    try:
        while time.perf_counter() - t0 < 3:
            win.make_current()
            win.clear()
            gpu.render(px, py, pz, rgb, win.w, win.h, cell_size=6)
            win.swap_buffers()
            win.pump_messages()
            frames += 1
    except KeyboardInterrupt:
        pass

    dt = time.perf_counter() - t0
    print(f"Rendered {frames} frames in {dt:.1f}s = {frames/dt:.0f} FPS")

    # Benchmark: 1000 frames
    t0 = time.perf_counter_ns()
    for _ in range(500):
        win.make_current()
        win.clear()
        gpu.render(px, py, pz, rgb, win.w, win.h, cell_size=6)
        win.swap_buffers()
    dt = (time.perf_counter_ns() - t0) / 500 / 1000
    print(f"Per frame: {dt:.1f} µs (GPU clear + render + swap)")

    win.destroy()
    gpu.destroy()
    print("DONE")
