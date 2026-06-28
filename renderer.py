#!/usr/bin/env python3
"""renderer.py — Бэкенд отрисовки для Hermes Cube.

Переписан после рефакторинга (июнь 2026):
  - RGBA-буфер напрямую (без TRANSPARENT_RGB / drawn-маски)
  - Pre-clip невидимых частиц ДО expand
  - Убран _fix_transparent_collisions (мертвый код в RGBA-режиме)
  - Статические stamp-методы для Render Graph

АРХИТЕКТУРА
-----------
Единственное место, где пиксели попадают на canvas.
Render Graph (core/render_graph.py) оркестрирует пассы и штампует
их в RGBA-буфер, renderer ТОЛЬКО выводит готовый буфер на экран.

ПРОЗРАЧНОСТЬ
------------
Tk-окно использует color-key `-transparentcolor #000001` (RGB 0,0,1).
RGBA-буфер: alpha=0 фон, alpha=255 частицы. PIL/ImageTk уважает
alpha-канал, так что фон остаётся прозрачным.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Дисковые ядра (для символа 'circle') — кешируются по размеру
# ---------------------------------------------------------------------------

_circle_kernels: dict[int, NDArray[np.uint8]] = {}


def _circle_kernel(cell: int) -> NDArray[np.uint8]:
    """Маска диска cell×cell (1 = рисовать, 0 = пропуск). Кешируется."""
    cached = _circle_kernels.get(cell)
    if cached is not None:
        return cached
    half = (cell - 1) / 2.0
    offs = np.arange(cell) - half
    dx, dy = np.meshgrid(offs, offs)
    mask = (dx * dx + dy * dy) <= half * half + 0.5
    kernel = mask.astype(np.uint8)
    _circle_kernels[cell] = kernel
    return kernel


# ---------------------------------------------------------------------------
# Кеш глифов — растеризация символов ОДИН РАЗ через PIL ImageFont
# ---------------------------------------------------------------------------

_glyph_cache: dict[tuple[str, int], tuple] = {}

_FONT_PATHS: list[str] = [
    r'C:\Windows\Fonts\seguisym.ttf',
    r'C:\Windows\Fonts\seguiemj.ttf',
    r'C:\Windows\Fonts\segoeui.ttf',
    r'C:\Windows\Fonts\arial.ttf',
]
_font_tried: dict[str, bool] = {}
_font_pool: dict[int, tuple] = {}


def _load_font(size: int) -> Optional[Any]:
    """Вернуть ImageFont указанного размера из первого доступного шрифта."""
    cached = _font_pool.get(size)
    if cached is not None:
        return cached[0]
    from PIL import ImageFont
    import os
    for path in _FONT_PATHS:
        ok = _font_tried.get(path)
        if ok is None:
            _font_tried[path] = os.path.isfile(path)
            ok = _font_tried[path]
        if not ok:
            continue
        try:
            font = ImageFont.truetype(path, size)
            _font_pool[size] = (font, path)
            return font
        except Exception:
            continue
    try:
        font = ImageFont.load_default()
        _font_pool[size] = (font, None)
        return font
    except Exception:
        return None


def _rasterize_glyph(symbol: str, size: int) -> Optional[tuple]:
    """Растеризовать символ в маску. Возвращает (dx, dy, gw, gh, bx, by)."""
    key = (symbol, size)
    cached = _glyph_cache.get(key)
    if cached is not None:
        return cached
    from PIL import Image, ImageDraw
    pad = max(2, size // 3)
    box = size + pad * 2
    img = Image.new('L', (box, box), 0)
    draw = ImageDraw.Draw(img)
    font = _load_font(size)
    if font is not None:
        try:
            draw.text((pad, pad), symbol, fill=255, font=font)
        except Exception:
            return None
    else:
        draw.text((pad, pad), symbol, fill=255)
    arr = np.asarray(img)
    ys, xs = np.nonzero(arr > 127)
    if len(xs) == 0:
        dx = np.array([0], dtype=np.int64)
        dy = np.array([0], dtype=np.int64)
        result = (dx, dy, 1, 1, 0, 0)
        _glyph_cache[key] = result
        return result
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    sub = arr[y0:y1 + 1, x0:x1 + 1] > 127
    gh, gw = sub.shape
    sdy, sdx = np.nonzero(sub)
    bx = (x0 + x1) // 2 - pad
    by = (y0 + y1) // 2 - pad
    result = (sdx.astype(np.int64), sdy.astype(np.int64), gw, gh, bx, by)
    _glyph_cache[key] = result
    return result


# ---------------------------------------------------------------------------
# PointCloudRenderer — тонкий бэкенд вывода на canvas
# ---------------------------------------------------------------------------


class PointCloudRenderer:
    """Тонкий бэкенд: выводит готовый RGBA-буфер на canvas одной картинкой.

    Все операции штамповки выполняются Pass-ами Render Graph.
    """

    _MARGIN: int = 8
    _MAX_BOX: int = 4096

    def __init__(self) -> None:
        self._canvas: Optional[Any] = None
        self._image_item: Optional[int] = None
        self._photo: Optional[Any] = None  # анти-GC
        # Буферный пул — переиспользовать вместо аллокации каждый кадр
        self._pool_w: int = 0
        self._pool_h: int = 0
        self._pool_buf: Optional[NDArray[np.uint8]] = None
        self._pool_pil: Optional[Any] = None
        self._pool_photo: Optional[Any] = None

    def get_buffer(self, w: int, h: int) -> NDArray[np.uint8]:
        """Вернуть переиспользуемый RGBA-буфер. Аллоцирует только если
        новый размер больше кешированного."""
        if w > self._pool_w or h > self._pool_h:
            self._pool_w = w
            self._pool_h = h
            self._pool_buf = np.zeros((h, w, 4), dtype=np.uint8)
            from PIL import Image
            self._pool_pil = Image.frombuffer('RGBA', (w, h), self._pool_buf,
                                               'raw', 'RGBA', 0, 1)
            self._pool_photo = None  # будет создан при первом blit
        # Очистка — быстро memset в C
        self._pool_buf[:] = (0, 0, 0, 0)
        return self._pool_buf

    def attach(self, canvas: Any) -> None:
        """Создать один create_image-айтем на canvas."""
        self._canvas = canvas
        if self._image_item is None:
            self._image_item = canvas.create_image(0, 0, anchor='nw', image='')

    def hide(self) -> None:
        """Спрятать картинку (пустой кадр)."""
        if self._canvas is not None and self._image_item is not None:
            try:
                self._canvas.itemconfig(self._image_item, image='')
            except Exception:
                pass

    def blit(self, rgba: NDArray[np.uint8], x0: int, y0: int) -> None:
        """Вывести готовый RGBA-буфер на canvas.

        Если буфер совпадает по размеру с кешированным — переиспользует
        PIL Image и PhotoImage (PIL Image разделяет память с numpy,
        так что после очистки+штамповки данные уже обновлены).
        """
        if self._canvas is None or self._image_item is None or rgba.size == 0:
            self.hide()
            return
        try:
            h, w = rgba.shape[:2]

            # Если буфер не совпадает с пулом — используем как есть (временная аллокация)
            if w == self._pool_w and h == self._pool_h and self._pool_pil is not None:
                # PIL Image уже разделяет память с self._pool_buf,
                # и self._pool_buf уже содержит свежие пиксели (get_buffer был вызван)
                from PIL import ImageTk
                if self._pool_photo is None:
                    self._pool_photo = ImageTk.PhotoImage(self._pool_pil)
                else:
                    # Обновляем существующий PhotoImage — paste быстрее чем new
                    self._pool_photo.paste(self._pool_pil)
                self._photo = self._pool_photo
            else:
                # Fallback: свежий буфер, создаём PIL + PhotoImage заново
                from PIL import Image, ImageTk
                mode = 'RGBA' if rgba.shape[-1] == 4 else 'RGB'
                img = Image.fromarray(rgba, mode=mode)
                self._photo = ImageTk.PhotoImage(img)

            self._canvas.coords(self._image_item, x0, y0)
            self._canvas.itemconfig(self._image_item, image=self._photo)
            try:
                self._canvas.tag_raise(self._image_item, 'drag_handle')
            except Exception:
                pass
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Статические stamp-методы — используются Pass-ами Render Graph
    # ------------------------------------------------------------------

    @staticmethod
    def stamp_points(
        buf: NDArray[np.uint8],
        x0: int, y0: int,
        px: NDArray[np.float64],
        py: NDArray[np.float64],
        rgb: NDArray[np.uint8],
        cell: int,
        symbol: str = 'square',
    ) -> None:
        """Штамповать N точек в RGBA-буфер с pre-clip.

        buf: (H, W, 4) RGBA — модифицируется на месте.
        px, py, rgb: в экранных координатах.  x0, y0: смещение буфера.

        Vectorized expand — одна операция на все частицы.
        """
        n = len(px)
        if n == 0:
            return

        buf_h, buf_w = buf.shape[0], buf.shape[1]

        # 'dot' — уменьшенный квадрат
        if symbol == 'dot':
            cell = max(2, cell // 2)

        half = cell // 2

        # --- Pre-clip: отсеять частицы вне буфера ---
        in_view = (
            (px.astype(np.int64) + half >= x0)
            & (px.astype(np.int64) - half < x0 + buf_w)
            & (py.astype(np.int64) + half >= y0)
            & (py.astype(np.int64) - half < y0 + buf_h)
        )
        px, py, rgb = px[in_view], py[in_view], rgb[in_view]
        n = len(px)
        if n == 0:
            return

        # Ядро смещений
        if symbol == 'circle':
            kernel = _circle_kernel(cell)
            dy_grid, dx_grid = np.nonzero(kernel)
        else:
            offs = np.arange(-half, cell - half)
            dx_grid, dy_grid = np.meshgrid(offs, offs)
            dx_grid = dx_grid.ravel()
            dy_grid = dy_grid.ravel()

        k = len(dx_grid)

        # Vectorized expand: (N, k) → flat
        gx = (px[:, None] + dx_grid[None, :]).astype(np.int64).ravel()
        gy = (py[:, None] + dy_grid[None, :]).astype(np.int64).ravel()
        gcol = np.broadcast_to(rgb[:, None, :], (n, k, 3)).reshape(-1, 3)

        # В координаты буфера
        lx = gx - x0
        ly = gy - y0
        valid = (lx >= 0) & (lx < buf_w) & (ly >= 0) & (ly < buf_h)

        # Прямая запись RGBA (alpha=255 для частиц)
        buf[ly[valid], lx[valid], :3] = gcol[valid]
        buf[ly[valid], lx[valid], 3] = 255

    @staticmethod
    def stamp_chars(
        buf: NDArray[np.uint8],
        x0: int, y0: int,
        px: NDArray[np.float64],
        py: NDArray[np.float64],
        rgb: NDArray[np.uint8],
        size: int,
        chars: Optional[List[str]],
    ) -> None:
        """Штамповать N символов в RGBA-буфер с pre-clip и группировкой."""
        if chars is None:
            return
        n = len(px)
        if n == 0:
            return

        buf_h, buf_w = buf.shape[0], buf.shape[1]

        # Pre-clip
        half = max(2, size // 2)
        in_view = (
            (px.astype(np.int64) + half >= x0)
            & (px.astype(np.int64) - half < x0 + buf_w)
            & (py.astype(np.int64) + half >= y0)
            & (py.astype(np.int64) - half < y0 + buf_h)
        )
        px, py, rgb = px[in_view], py[in_view], rgb[in_view]
        n = len(px)
        if n == 0:
            return

        # Группировка по символу: {символ: [индексы]}
        groups: Dict[str, list[int]] = {}
        for i, ch in enumerate(chars):
            groups.setdefault(ch, []).append(i)

        for ch, idxs in groups.items():
            gpx = px[idxs]
            gpy = py[idxs]
            gcol = rgb[idxs]
            gn = len(idxs)

            glyph = _rasterize_glyph(ch, size)
            if glyph is None:
                continue
            sdx, sdy, _gw, _gh, bx, by = glyph
            dxs = sdx - bx
            dys = sdy - by
            k = len(dxs)

            gx = (gpx[:, None] + dxs[None, :]).astype(np.int64).ravel()
            gy = (gpy[:, None] + dys[None, :]).astype(np.int64).ravel()
            cols = np.broadcast_to(
                gcol[:, None, :], (gn, k, 3),
            ).reshape(-1, 3)

            lx = gx - x0
            ly = gy - y0
            valid = (lx >= 0) & (lx < buf_w) & (ly >= 0) & (ly < buf_h)

            buf[ly[valid], lx[valid], :3] = cols[valid]
            buf[ly[valid], lx[valid], 3] = 255

    @staticmethod
    def compute_bbox(
        layers: List[Dict[str, Any]],
        margin: int = 8,
        max_box: int = 4096,
    ) -> Optional[Tuple[int, int, int, int]]:
        """Вычислить bounding box для всех слоёв (экранные координаты).

        layers: список слоёв, каждый с {'px':..., 'py':..., 'cell':...}
        Возвращает (x0, y0, x1, y1) или None.
        """
        if not layers:
            return None
        x0 = y0 = 10**9
        x1 = y1 = -10**9
        for layer in layers:
            px = layer['px']
            py = layer['py']
            cell = layer.get('cell', 6)
            half = cell // 2
            if len(px) == 0:
                continue
            x0 = min(x0, int(px.min()) - half)
            y0 = min(y0, int(py.min()) - half)
            x1 = max(x1, int(px.max()) + (cell - half) + 1)
            y1 = max(y1, int(py.max()) + (cell - half) + 1)
        # Страховка от выбросов
        x0 = max(0, x0 - margin)
        y0 = max(0, y0 - margin)
        x1 = min(x1 + margin, max_box)
        y1 = min(y1 + margin, max_box)
        if x1 <= x0 or y1 <= y0:
            return None
        return (x0, y0, x1, y1)

    @staticmethod
    def allocate_rgba(x0: int, y0: int, x1: int, y1: int) -> NDArray[np.uint8]:
        """Создать RGBA-буфер (transparent background) под bbox."""
        w = max(1, x1 - x0)
        h = max(1, y1 - y0)
        return np.zeros((h, w, 4), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Самотестирование
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import time
    import tkinter as tk

    # Бенчмарк штамповки
    def bench(n: int, cell: int = 6) -> float:
        rng = np.random.default_rng(0)
        px = rng.uniform(100, 700, n)
        py = rng.uniform(100, 700, n)
        rgb = rng.integers(0, 256, (n, 3), dtype=np.uint8)
        buf = np.zeros((800, 800, 4), dtype=np.uint8)
        t0 = time.perf_counter()
        for _ in range(20):
            buf[:] = 0
            PointCloudRenderer.stamp_points(
                buf, 0, 0, px, py, rgb, cell, 'square')
        dt = (time.perf_counter() - t0) / 20
        return dt

    for n in (2400, 26000):
        ms = bench(n) * 1000
        print(f"  stamp {n:6d} points (cell=6): {ms:6.2f} ms/frame")

    # Визуальный smoke test
    root = tk.Tk()
    root.title('♢ renderer.py — smoke test')
    root.geometry('800x800+100+100')
    root.configure(bg='#000001')
    root.attributes('-transparentcolor', '#000001')
    canvas = tk.Canvas(root, bg='#000001', highlightthickness=0)
    canvas.pack(fill=tk.BOTH, expand=True)

    r = PointCloudRenderer()
    r.attach(canvas)

    rng = np.random.default_rng(1)
    frame = [0]

    def tick() -> None:
        t = frame[0] * 0.05
        n = 2400
        ang = np.linspace(0, 2 * np.pi, n) + t
        radii = np.linspace(0, 300, n)
        px = 400 + np.cos(ang * 3 + t) * radii
        py = 400 + np.sin(ang * 2) * radii
        z = np.sin(ang * 5)
        order = np.argsort(z)
        px, py = px[order], py[order]
        rgb = np.zeros((n, 3), dtype=np.uint8)
        rgb[:, 0] = (128 + 127 * np.sin(ang)).astype(np.uint8)
        rgb[:, 1] = (128 + 127 * np.cos(ang)).astype(np.uint8)
        rgb[:, 2] = 200

        # Render Graph style: allocate → stamp → blit
        bbox = PointCloudRenderer.compute_bbox([
            {'px': px, 'py': py, 'cell': 6},
        ], margin=8)
        if bbox:
            x0, y0, x1, y1 = bbox
            buf = PointCloudRenderer.allocate_rgba(x0, y0, x1, y1)
            PointCloudRenderer.stamp_points(
                buf, x0, y0, px, py, rgb, 6, 'square')
            r.blit(buf, x0, y0)

        frame[0] += 1
        root.after(33, tick)

    root.after(100, tick)
    print("\nОткрываю окно smoke-test. Закрой окно для выхода.")
    root.mainloop()
