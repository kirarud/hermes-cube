#!/usr/bin/env python3
"""core/gpu.py — GPU-рендерер (moderngl, batch quads, векторизован).

Каждая частица = 6 вершин (2 треугольника), все в одном VBO.
Векторизованный batch — без Python-цикла по частицам.

Режимы:
  - dots: шейдер выбирает square/circle/dot через uniform u_symbol
  - chars: font atlas текстура + per-instance char_idx
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Any, Dict, Optional

# fmt: off
_QUAD_UVS = np.array([
    [-0.5, -0.5],  [ 0.5, -0.5],  [-0.5,  0.5],
    [-0.5,  0.5],  [ 0.5, -0.5],  [ 0.5,  0.5],
], dtype=np.float32)  # (6, 2)
# fmt: on

VERTEX_SHADER_DOTS = """#version 330
in vec2 in_position;
in vec3 in_color;
in vec2 in_uv;
out vec3 v_color;
out vec2 v_uv;
flat out int v_symbol;
uniform int u_symbol;
void main() {
    gl_Position = vec4(in_position, 0.0, 1.0);
    v_color = in_color;
    v_uv = in_uv;
    v_symbol = u_symbol;
}
"""

FRAGMENT_SHADER_DOTS = """#version 330
in vec3 v_color;
in vec2 v_uv;
flat in int v_symbol;
out vec4 f_color;
void main() {
    float d = length(v_uv);
    if (v_symbol == 0) {  // square
        f_color = vec4(v_color, 1.0);
        return;
    }
    if (v_symbol == 1) {  // circle
        if (d > 0.5) discard;
        float edge = 1.0 - d * 0.3;
        f_color = vec4(v_color * edge, 1.0);
        return;
    }
    if (v_symbol == 2) {  // dot
        if (d > 0.15) discard;
        f_color = vec4(v_color, 1.0);
        return;
    }
    f_color = vec4(v_color, 1.0);
}
"""

# ── Шейдеры для font atlas ──────────────────────────────────────────

VERTEX_SHADER_CHARS = """#version 330
in vec2 in_position;
in vec3 in_color;
in vec2 in_uv;
in float in_char_idx;
out vec3 v_color;
out vec2 v_uv;
out float v_char_idx_f;
void main() {
    gl_Position = vec4(in_position, 0.0, 1.0);
    v_color = in_color;
    v_uv = in_uv;
    v_char_idx_f = in_char_idx;
}
"""

FRAGMENT_SHADER_CHARS = """#version 330
in vec3 v_color;
in vec2 v_uv;
in float v_char_idx_f;
out vec4 f_color;
uniform sampler2D u_font_tex;
void main() {
    float ch = v_char_idx_f;
    // 16x16 grid: cell (col, row)
    float tx = mod(ch, 16.0);
    float ty = floor(ch / 16.0);
    vec2 tc;
    tc.x = (tx + v_uv.x + 0.5) / 16.0;
    tc.y = (ty + v_uv.y + 0.5) / 16.0;

    vec4 texel = texture(u_font_tex, tc);
    if (texel.a < 0.01) discard;
    f_color = vec4(v_color, texel.a);
}
"""


class GpuRenderer:
    """GPU batch-рендерер. Векторизован — никаких Python-циклов по частицам.

    Два режима:
      - dots (default): square/circle/dot через `u_symbol` uniform
      - chars: font atlas текстура + per-instance char_idx
    """

    MAX_PARTICLES: int = 32768
    ATLAS_W: int = 128
    ATLAS_H: int = 128
    CELL_SIZE: int = 8
    ATLAS_GRID: int = 16  # 16×16 ячеек

    def __init__(self) -> None:
        self._ctx: Any = None
        self._program_dots: Any = None
        self._program_chars: Any = None
        self._vbo: Any = None  # (pos+col+uv) или (pos+col+uv+char_idx)
        self._vao_dots: Any = None
        self._vao_chars: Any = None
        self._font_tex: Any = None  # texture object
        self._initialized: bool = False
        self._fallback: bool = True
        self._n_capacity: int = 0
        self._symbol: str = 'circle'
        self._char_map: Dict[str, NDArray[np.int32]] = {}
        self._current_set: str = 'default'

    @property
    def available(self) -> bool:
        return self._initialized and not self._fallback

    def init_from_context(self, ctx: Any) -> bool:
        try:
            self._ctx = ctx
            # Dots program
            self._program_dots = ctx.program(
                vertex_shader=VERTEX_SHADER_DOTS,
                fragment_shader=FRAGMENT_SHADER_DOTS,
            )
            # Chars program
            self._program_chars = ctx.program(
                vertex_shader=VERTEX_SHADER_CHARS,
                fragment_shader=FRAGMENT_SHADER_CHARS,
            )
            self._initialized = True
            self._fallback = False
            return True
        except Exception as e:
            print(f"[GPU] Init failed: {e}", flush=True)
            self._initialized = False
            self._fallback = True
            return False

    def load_font_atlas(self, rgba_bytes: bytes,
                        char_maps: Dict[str, NDArray[np.int32]]) -> None:
        """Загрузить font atlas текстуру."""
        if self._ctx is None or self._fallback:
            return
        # Pillow возвращает RGBA, moderngl хранит как RGBA на non-Windows,
        # но font_atlas уже сконвертировал в BGRA для Windows совместимости
        self._font_tex = self._ctx.texture(
            (self.ATLAS_W, self.ATLAS_H), 4, rgba_bytes,
        )
        self._font_tex.filter = (self._ctx.NEAREST, self._ctx.NEAREST)
        self._font_tex.anisotropy = 0
        self._char_map = char_maps

        # Устанавливаем uniform атласа
        self._program_chars['u_font_tex'] = 0
        print(f"[GPU] Font atlas loaded: {len(self._char_map)} sets, {self.ATLAS_W}×{self.ATLAS_H}", flush=True)

    def upload(self, n_particles: int, use_chars: bool = False, force: bool = False) -> bool:
        if self._fallback or not self._initialized:
            return False
        if not force and n_particles == self._n_capacity and self._vbo is not None:
            return True
        n = min(n_particles, self.MAX_PARTICLES)

        # Единый формат VBO: pos(2f)+col(3f)+uv(2f)+cidx(1f) = 32 bytes
        # Dots VAO использует только первые 7f, chars VAO все 8f
        dtype = [('pos', 'f4', 2), ('col', 'f4', 3), ('uv', 'f4', 2), ('cidx', 'f4')]
        verts = np.zeros(n * 6, dtype=dtype)
        self._vbo = self._ctx.buffer(verts.tobytes())

        # VAO dots (берёт только 2f 3f 2f)
        self._vao_dots = self._ctx.vertex_array(
            self._program_dots,
            [(self._vbo, '2f 3f 2f', 'in_position', 'in_color', 'in_uv')],
        )
        # VAO chars (берёт все 2f 3f 2f 1f)
        self._vao_chars = self._ctx.vertex_array(
            self._program_chars,
            [(self._vbo, '2f 3f 2f 1f',
              'in_position', 'in_color', 'in_uv', 'in_char_idx')],
        )
        self._n_capacity = n
        self._vbo_has_cidx = True
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
        use_chars: bool = False,
        char_indices: Optional[NDArray[np.int32]] = None,
    ) -> None:
        if self._fallback or not self._initialized or self._ctx is None:
            return
        n = len(projected_x)
        if n == 0:
            return
        if n > self._n_capacity:
            self.upload(n, use_chars=use_chars, force=True)
        elif use_chars != getattr(self, '_last_use_chars', False):
            # Режим сменился — пересоздаём VBO
            self.upload(n, use_chars=use_chars, force=True)
        self._last_use_chars = use_chars
        if self._vbo is None:
            return

        w_f, h_f = float(view_width), float(view_height)
        cell_f = float(cell_size)

        # Позиции в NDC
        cx = projected_x[:n] / w_f * 2.0 - 1.0
        cy = -(projected_y[:n] / h_f * 2.0 - 1.0)

        dx = cell_f / w_f
        dy = cell_f / h_f

        dx_offsets = np.array([-dx, dx, -dx, -dx, dx, dx], dtype=np.float64)
        dy_offsets = np.array([-dy, -dy, dy, dy, -dy, dy], dtype=np.float64)

        cx_6 = np.repeat(cx, 6)
        cy_6 = np.repeat(cy, 6)
        dx_full = np.tile(dx_offsets, n)
        dy_full = np.tile(dy_offsets, n)

        positions = np.column_stack((cx_6 + dx_full, cy_6 + dy_full))
        colors = np.repeat(rgb[:n] / 255.0, 6, axis=0)
        uvs = np.tile(_QUAD_UVS, (n, 1))

        stride = n * 6

        if use_chars and char_indices is not None:
            # Extended vertex: pos+col+uv+char_idx
            cidx = np.repeat(char_indices[:n].astype(np.float32), 6)
            data = np.empty(stride, dtype=[
                ('pos', 'f4', 2), ('col', 'f4', 3), ('uv', 'f4', 2), ('cidx', 'f4'),
            ])
            data['pos'] = positions
            data['col'] = colors
            data['uv'] = uvs
            data['cidx'] = cidx

            self._vbo.write(data.tobytes())
            if self._font_tex is not None:
                self._font_tex.use(location=0)
            self._vao_chars.render(mode=self._ctx.TRIANGLES, vertices=n * 6)
        else:
            # Dots
            data = np.empty(stride, dtype=[
                ('pos', 'f4', 2), ('col', 'f4', 3), ('uv', 'f4', 2),
            ])
            data['pos'] = positions
            data['col'] = colors
            data['uv'] = uvs

            symbol_map = {'square': 0, 'circle': 1, 'dot': 2}
            symbol_name = getattr(self, '_symbol', 'circle')
            self._program_dots['u_symbol'] = int(symbol_map.get(symbol_name, 1))

            self._vbo.write(data.tobytes())
            self._vao_dots.render(mode=self._ctx.TRIANGLES, vertices=n * 6)

    def set_symbol_set(self, name: str) -> Optional[NDArray[np.int32]]:
        """Выбрать набор символов по имени. Вернуть массив индексов."""
        arr = self._char_map.get(name)
        if arr is not None:
            self._current_set = name
        return arr

    def destroy(self) -> None:
        for attr in ('_vao_dots', '_vao_chars', '_vbo',
                     '_program_dots', '_program_chars', '_font_tex'):
            try:
                obj = getattr(self, attr, None)
                if obj is not None:
                    obj.release()
            except Exception:
                pass
        self._initialized = False
        self._fallback = True
