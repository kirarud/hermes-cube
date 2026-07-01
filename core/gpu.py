#!/usr/bin/env python3
"""core/gpu.py — GPU-рендерер (moderngl, batch quads, векторизован).

Каждая частица = 6 вершин (2 треугольника), все в одном VBO.
Векторизованный batch — без Python-цикла по частицам.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Any, Optional

# fmt: off
_QUAD_UVS = np.array([
    [-0.5, -0.5],  [ 0.5, -0.5],  [-0.5,  0.5],
    [-0.5,  0.5],  [ 0.5, -0.5],  [ 0.5,  0.5],
], dtype=np.float32)  # (6, 2)
# fmt: on

VERTEX_SHADER = """#version 330
in vec2 in_position;
in vec3 in_color;
in vec2 in_uv;
out vec3 v_color;
out vec2 v_uv;
void main() {
    gl_Position = vec4(in_position, 0.0, 1.0);
    v_color = in_color;
    v_uv = in_uv;
}
"""

FRAGMENT_SHADER = """#version 330
in vec3 v_color;
in vec2 v_uv;
out vec4 f_color;
void main() {
    float d = length(v_uv);
    if (d > 0.5) discard;
    float edge = 1.0 - d * 0.3;
    f_color = vec4(v_color * edge, 1.0);
}
"""


class GpuRenderer:
    """GPU batch-рендерер. Векторизован — никаких Python-циклов по частицам."""

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

    def init_from_context(self, ctx: Any) -> bool:
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
            print(f"[GPU] Shader init failed: {e}", flush=True)
            self._initialized = False
            self._fallback = True
            return False

    def upload(self, n_particles: int) -> bool:
        if self._fallback or not self._initialized:
            return False
        if n_particles == self._n_capacity and self._vbo is not None:
            return True
        n = min(n_particles, self.MAX_PARTICLES)
        verts = np.zeros(n * 6, dtype=[
            ('pos', 'f4', 2), ('col', 'f4', 3), ('uv', 'f4', 2),
        ])
        self._vbo = self._ctx.buffer(verts.tobytes())
        self._vao = self._ctx.vertex_array(
            self._program,
            [(self._vbo, '2f 3f 2f', 'in_position', 'in_color', 'in_uv')],
        )
        self._n_capacity = n
        return True

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
        if self._fallback or not self._initialized or self._ctx is None:
            return
        n = len(projected_x)
        if n == 0:
            return
        if n > self._n_capacity:
            self.upload(n)
        if self._vbo is None or self._vao is None:
            return

        w_f, h_f = float(view_width), float(view_height)
        cell_f = float(cell_size)

        # Векторизованная сборка вершин — без Python-цикла
        # Принцип:
        #   Для каждой частицы генерируем 6 вершин с одинаковым цветом и разными UV.
        #   Расширяем массивы частиц в 6 раз → numpy broadcast.

        # Позиции частиц в NDC
        cx = projected_x[:n] / w_f * 2.0 - 1.0
        cy = -(projected_y[:n] / h_f * 2.0 - 1.0)

        dx = cell_f / w_f
        dy = cell_f / h_f

        # Смещения для 6 вершин: [b-l, b-r, t-l, t-l, b-r, t-r]
        dx_offsets = np.array([-dx, dx, -dx, -dx, dx, dx], dtype=np.float64)
        dy_offsets = np.array([-dy, -dy, dy, dy, -dy, dy], dtype=np.float64)

        # Расширяем позиции частиц в 6x
        cx_6 = np.repeat(cx, 6)  # (n*6,)
        cy_6 = np.repeat(cy, 6)
        dx_full = np.tile(dx_offsets, n)  # (n*6,)
        dy_full = np.tile(dy_offsets, n)

        # (n*6, 2)
        positions = np.column_stack((cx_6 + dx_full, cy_6 + dy_full))

        # Цвет: расширяем в 6x
        colors = np.repeat(rgb[:n] / 255.0, 6, axis=0)  # (n*6, 3)

        # UV: просто повторяем _QUAD_UVS n раз
        uvs = np.tile(_QUAD_UVS, (n, 1))  # (n*6, 2)

        # Pack в массив и шлём в VBO
        stride = n * 6
        data = np.empty(stride, dtype=[
            ('pos', 'f4', 2), ('col', 'f4', 3), ('uv', 'f4', 2),
        ])
        data['pos'] = positions
        data['col'] = colors
        data['uv'] = uvs

        self._vbo.write(data.tobytes())
        self._vao.render(mode=self._ctx.TRIANGLES, vertices=n * 6)

    def destroy(self) -> None:
        for attr in ('_vao', '_vbo', '_program'):
            try:
                obj = getattr(self, attr, None)
                if obj is not None:
                    obj.release()
            except Exception:
                pass
        self._initialized = False
        self._fallback = True
