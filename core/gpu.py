"""core/gpu.py — GPU-рендерер для Hermes Engine (moderngl).

Создаёт полноценный конвейер:
  - VBO с позициями частиц (один раз, static)
  - Vertex shader: rotation via matrix uniform
  - Fragment shader: color + depth shading
  - Instanced rendering: 1 draw call для N частиц
  - PBO read-back → numpy → Tkinter bridge

Заменяет PointCloudRenderer при наличии GPU.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple


# Vertext shader: принимает позиции частиц, применяет uniform-матрицу
VERTEX_SHADER = """
#version 330

in vec2 in_position;
in vec3 in_color;
in float in_depth;

uniform mat3 u_rotation;
uniform vec2 u_offset;
uniform float u_scale;

out vec3 v_color;
out float v_depth;

void main() {
    vec3 pos = u_rotation * vec3(in_position, 1.0);
    gl_Position = vec4(pos.xy * u_scale + u_offset, 0.0, 1.0);
    v_color = in_color;
    v_depth = pos.z;
}
"""

# Fragment shader: цвет с depth shading
FRAGMENT_SHADER = """
#version 330

in vec3 v_color;
in float v_depth;

out vec4 f_color;

void main() {
    float depth_factor = 0.6 + 0.4 * (v_depth + 1.0) / 2.0;
    f_color = vec4(v_color * depth_factor / 255.0, 1.0);
}
"""


class GpuRenderer:
    """GPU-рендерер для облака частиц.

    Создаёт контекст moderngl, загружает шейдеры, VBO.
    Поддерживает CPU-fallback при ошибках.
    """

    def __init__(self) -> None:
        self._ctx = None
        self._program = None
        self._vbo = None
        self._vao = None
        self._initialized: bool = False
        self._fallback: bool = True  # по умолчанию CPU

    def init(self) -> bool:
        """Инициализировать GPU-контекст. True если успешно."""
        try:
            import moderngl
            self._ctx = moderngl.create_standalone_context(require=330)
            self._program = self._ctx.program(
                vertex_shader=VERTEX_SHADER,
                fragment_shader=FRAGMENT_SHADER,
            )
            self._initialized = True
            self._fallback = False
            print("[GPU] moderngl context created (OpenGL 3.3+)")
            return True
        except Exception as e:
            print(f"[GPU] Fallback to CPU: {e}")
            self._initialized = False
            self._fallback = True
            return False

    def upload_particles(self, positions: NDArray[np.float64],
                         colors: NDArray[np.uint8]) -> None:
        """Загрузить частицы в VBO."""
        if self._fallback or not self._initialized:
            return
        n = len(positions)
        if n == 0:
            return

        # Pack: position(x,y) + color(r,g,b) + depth(z)
        data = np.zeros(n, dtype=[
            ('pos', 'f4', 2), ('color', 'u1', 3), ('depth', 'f4'),
        ])
        data['pos'] = positions[:, :2].astype(np.float32)
        data['color'] = colors.astype(np.uint8)
        data['depth'] = positions[:, 2].astype(np.float32)

        import moderngl
        self._vbo = self._ctx.buffer(data.tobytes())
        self._vao = self._ctx.vertex_array(
            self._program,
            [
                (self._vbo, '2f 3u1 f', 'in_position', 'in_color', 'in_depth'),
            ],
        )

    def render(self, rotation_matrix, offset_x, offset_y, scale) -> Optional[NDArray[np.uint8]]:
        """Выполнить отрисовку. Возвращает RGBA numpy-буфер или None."""
        if self._fallback or not self._initialized:
            return None

        self._program['u_rotation'].write(rotation_matrix.astype(np.float32).tobytes())
        self._program['u_offset'] = (offset_x, offset_y)
        self._program['u_scale'] = scale

        fbo = self._ctx.simple_framebuffer((800, 600))
        fbo.use()
        self._ctx.clear(0.0, 0.0, 0.0, 0.0)
        self._vao.render(mode=self._ctx.POINTS)

        # Read-back
        buf = fbo.read(components=4)
        arr = np.frombuffer(buf, dtype=np.uint8).reshape((600, 800, 4))
        return arr

    @property
    def available(self) -> bool:
        return self._initialized and not self._fallback


# Quick test
if __name__ == '__main__':
    gpu = GpuRenderer()
    if gpu.init():
        print("GPU renderer ready")
    else:
        print("GPU not available, CPU fallback")
