#!/usr/bin/env python3
"""renderer.py — Слой отрисовки облака точек для Hermes Cube.

АРХИТЕКТУРА
-----------
Это ЕДИНСТВЕННОЕ место, где точки попадают на экран. Всё остальное
(CubeEngine, ai_module, etc.) считает данные, а renderer их рисует.

Текущий бэкенд: numpy-буфер + ОДИН create_image за кадр.
Раньше было: N вызовов canvas.coords()+itemconfig() (≈2400–26000 на кадр),
что и было главным тормозом. Теперь все точки штампуются в один
массив за векторный проход и выводятся одной картинкой.

Эта абстракция — точка роста: если завтра понадобится настоящий GPU,
меняется ТОЛЬКО тело finish_frame() / бэкенд, а все вызывающие модули
(CubeApp, трейлы) остаются нетронутыми.

ПРОЗРАЧНОСТЬ
------------
Tk-окно использует color-key `-transparentcolor #000001` (RGB 0,0,1).
Фон буфера = TRANSPARENT_RGB (0,0,1). Любой пиксель частицы, случайно
равный этому цвету, стал бы «дыркой». После заливки стоит дешёвый
векторный guard, который такие пиксели чуть сдвигает.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

# Должно совпадать с TRANSPARENT_COLOR в cube_app.py (#000001) и pixel_grid.py
TRANSPARENT_RGB: Tuple[int, int, int] = (0, 0, 1)

#: Максимальный размер bbox-буфера по стороне (страховка от выбросов)
_MAX_BOX: int = 4096

#: Запас вокруг bbox, чтобы частицы на краю не обрезались
_MARGIN: int = 8


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

#: Кеш: (symbol, size_px) → (dx_offsets, dy_offsets, gw, gh, baseline_x, baseline_y)
#: dx/dy — смещения активных пикселей маски относительно центра глифа.
_glyph_cache: dict[tuple[str, int], tuple] = {}

#: Шрифты Windows (пробуются по порядку), fallback на PIL default
_FONT_PATHS: list[str] = [
    r'C:\Windows\Fonts\seguisym.ttf',   # символы ◆◇●○▲△★☆♥♢
    r'C:\Windows\Fonts\seguiemj.ttf',   # эмодзи 😊😢🤖
    r'C:\Windows\Fonts\segoeui.ttf',    # кириллица + base
    r'C:\Windows\Fonts\arial.ttf',      # запасная кириллица
]
#: (путь → True) шрифтов, которые удалось загрузить — чтобы не долбить диск
_font_tried: dict[str, bool] = {}
#: Загруженные font-объекты по размеру: size → (ImageFont obj, путь)
_font_pool: dict[int, tuple] = {}


def _load_font(size: int) -> Optional[Any]:
    """Вернуть ImageFont указанного размера из первого доступного шрифта.
    Кеш: повторные запросы того же размера не дёргают диск."""
    cached = _font_pool.get(size)
    if cached is not None:
        return cached[0]
    from PIL import ImageFont
    for path in _FONT_PATHS:
        ok = _font_tried.get(path)
        if ok is None:
            import os
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
    # Полный fallback
    try:
        font = ImageFont.load_default()
        _font_pool[size] = (font, None)
        return font
    except Exception:
        return None


def _rasterize_glyph(symbol: str, size: int) -> Optional[tuple]:
    """Растеризовать символ в маску. Возвращает (dx, dy, gw, gh, bx, by):
    dx/dy — координаты активных пикселей относительно центра глифа;
    gw/gh — размеры растра; bx/by — смещение центра в пикселях.
    Кешируется по (symbol, size)."""
    key = (symbol, size)
    cached = _glyph_cache.get(key)
    if cached is not None:
        return cached
    from PIL import Image, ImageDraw
    # Рисуем с запасом, чтобы глиф не обрезался
    pad = max(2, size // 3)
    box = size + pad * 2
    img = Image.new('L', (box, box), 0)  # чёрный фон = 0
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
    # Обрезаем пустые края → компактная маска
    ys, xs = np.nonzero(arr > 127)
    if len(xs) == 0:
        # Пустой глиф (нет в шрифте) — ставим крошечную точку
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
    # Центр растра (примерно где базовая линия/центр символа)
    bx = (x0 + x1) // 2 - pad
    by = (y0 + y1) // 2 - pad
    result = (sdx.astype(np.int64), sdy.astype(np.int64), gw, gh, bx, by)
    _glyph_cache[key] = result
    return result


# ---------------------------------------------------------------------------
# PointCloudRenderer
# ---------------------------------------------------------------------------

# Слой точек: (px, py, rgb, cell_size, symbol, chars)
# chars — Optional[List[str]]: символ на частицу (для символьного режима),
#                                 либо None (геометрический режим).
_Layer = Tuple[NDArray[np.float64], NDArray[np.float64],
               NDArray[np.uint8], int, str, Optional[List[str]]]


class PointCloudRenderer:
    """Рисует облако точек одной картинкой.

    Жизненный цикл кадра:
        r.begin_frame()
        r.add_points(trail_px, trail_py, trail_rgb, cell, 'square')  # фон
        r.add_points(cube_px,  cube_py,  cube_rgb,  cell, 'square')  # перед
        r.finish_frame(canvas)
    Слои штампуются в порядке добавления; позже добавленный — сверху
    (важно для трейлов под кубом и для painter's algorithm внутри слоя,
    когда массив уже отсортирован по глубине back→front).
    """

    def __init__(self) -> None:
        self._canvas: Optional[any] = None
        self._image_item: Optional[int] = None
        self._photo: Optional[any] = None  # анти-GC ссылка на PhotoImage
        self._layers: List[_Layer] = []
        self._bbox: Optional[Tuple[int, int, int, int]] = None
        self._last_offset: Tuple[int, int] = (0, 0)

    # ── Привязка к canvas ────────────────────────────────────────────────

    def attach(self, canvas: any) -> None:
        """Создать один create_image-айтем на canvas и запомнить его."""
        self._canvas = canvas
        if self._image_item is None:
            self._image_item = canvas.create_image(0, 0, anchor='nw', image='')

    def hide(self) -> None:
        """Спрятать картинку (например, когда кадр пустой)."""
        if self._canvas is not None and self._image_item is not None:
            try:
                self._canvas.itemconfig(self._image_item, image='')
            except Exception:
                pass

    # ── Жизненный цикл кадра ─────────────────────────────────────────────

    def begin_frame(self) -> None:
        """Начать новый кадр: сбросить слои и bbox."""
        self._layers = []
        self._bbox = None

    def add_points(
        self,
        px: NDArray[np.float64],
        py: NDArray[np.float64],
        rgb: NDArray[np.uint8],
        cell_size: int,
        symbol: str = 'square',
    ) -> None:
        """Добавить слой из N точек.

        Аргументы:
            px, py: (N,) экранные координаты (float64).
            rgb:    (N, 3) uint8 — цвет каждой точки (уже с depth-shading
                    и HSV-сдвигом). Для painter's algorithm массив должен
                    быть отсортирован back→front: тогда при коллизиях
                    «последний выигрывает», т.е. ближние точки сверху.
            cell_size: размер частицы в пикселях.
            symbol: 'square' | 'circle' | 'dot'.
        """
        n = len(px)
        if n == 0:
            return
        # 'dot' — уменьшенный квадрат (поведение как в старом коде)
        if symbol == 'dot':
            cell_size = max(2, cell_size // 2)
        self._layers.append((px, py, rgb, cell_size, symbol, None))
        half = cell_size // 2
        x_lo = int(px.min()) - half
        y_lo = int(py.min()) - half
        x_hi = int(px.max()) + (cell_size - half) + 1
        y_hi = int(py.max()) + (cell_size - half) + 1
        if self._bbox is None:
            self._bbox = (x_lo, y_lo, x_hi, y_hi)
        else:
            bx0, by0, bx1, by1 = self._bbox
            self._bbox = (min(bx0, x_lo), min(by0, y_lo),
                          max(bx1, x_hi), max(by1, y_hi))

    def add_chars(
        self,
        px: NDArray[np.float64],
        py: NDArray[np.float64],
        rgb: NDArray[np.uint8],
        chars: List[str],
        size: int,
    ) -> None:
        """Добавить символьный слой: N частиц, каждая — свой символ из chars.

        Символы растеризуются ОДИН РАЗ (кешируются), затем штампуются
        в буфер векторно — как квадраты, но с маской глифа вместо цельного
        ядра. Группировка по символу (их в наборе 8-32) даёт горсть numpy-
        операций вместо N Canvas-вызовов.

        Аргументы:
            px, py: (N,) экранные координаты.
            rgb:    (N, 3) uint8 — цвет каждой частицы.
            chars:  List[str] длиной N — символ на частицу.
            size:   целевой размер глифа в пикселях.
        """
        n = len(px)
        if n == 0 or len(chars) == 0:
            return
        # Согласуем длины: chars добиваем/обрезаем под N
        if len(chars) < n:
            chars = [chars[i % len(chars)] for i in range(n)]
        elif len(chars) > n:
            chars = chars[:n]
        self._layers.append((px, py, rgb, size, 'char', list(chars)))
        # bbox: оценка с запасом под размер глифа
        half = max(2, size // 2)
        x_lo = int(px.min()) - half
        y_lo = int(py.min()) - half
        x_hi = int(px.max()) + size + half
        y_hi = int(py.max()) + size + half
        if self._bbox is None:
            self._bbox = (x_lo, y_lo, x_hi, y_hi)
        else:
            bx0, by0, bx1, by1 = self._bbox
            self._bbox = (min(bx0, x_lo), min(by0, y_lo),
                          max(bx1, x_hi), max(by1, y_hi))

    def finish_frame(self) -> Tuple[int, int]:
        """Штамповать все слои в одну картинку и вывести на canvas.

        Возвращает (x0, y0) — экранные координаты левого-верхнего угла
        картинки (может пригодиться вызывающему коду).
        """
        if self._canvas is None or self._image_item is None:
            return (0, 0)
        if not self._layers or self._bbox is None:
            self.hide()
            return (0, 0)

        x0, y0, x1, y1 = self._bbox
        # Запас и страховка от ухода за экран / гигантских bbox
        x0 -= _MARGIN
        y0 -= _MARGIN
        x1 += _MARGIN
        y1 += _MARGIN
        w = max(1, x1 - x0)
        h = max(1, y1 - y0)
        if w > _MAX_BOX or h > _MAX_BOX:
            # Выброс (например, куб уехал за экран) — клампим
            w = min(w, _MAX_BOX)
            h = min(h, _MAX_BOX)

        # Буфер фона = прозрачный цвет окна
        buf: NDArray[np.uint8] = np.full(
            (h, w, 3), TRANSPARENT_RGB, dtype=np.uint8,
        )
        # Маска реально нарисованных пикселей (чтобы фиксить коллизии
        # ТОЛЬКО на частицах, не на всём фоне).
        drawn: NDArray[np.bool_] = np.zeros((h, w), dtype=np.bool_)

        for px, py, rgb, cell, symbol, chars in self._layers:
            if symbol == 'char':
                self._stamp_char_layer(buf, drawn, x0, y0, px, py, rgb, cell, chars)
            else:
                self._stamp_layer(buf, drawn, x0, y0, px, py, rgb, cell, symbol)

        self._fix_transparent_collisions(buf, drawn)

        # Конвертируем RGB→RGBA: нарисованные пиксели непрозрачны,
        # фон — полностью прозрачен. Так PIL учитывает настоящую альфу
        # при blit, и drag handle (Canvas-полигон) не перекрывается
        # грязным прямоугольником.
        rgba: NDArray[np.uint8] = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[..., :3] = buf
        rgba[drawn, 3] = 255  # нарисованные пиксели → непрозрачные
        # Фон (0,0,0,0) — полностью прозрачный, PIL его пропускает.

        self._blit(rgba, x0, y0)
        self._last_offset = (x0, y0)
        return (x0, y0)

    # ── Векторная заливка одного слоя ────────────────────────────────────

    @staticmethod
    def _stamp_layer(
        buf: NDArray[np.uint8],
        drawn: NDArray[np.bool_],
        x0: int, y0: int,
        px: NDArray[np.float64], py: NDArray[np.float64],
        rgb: NDArray[np.uint8], cell: int, symbol: str,
    ) -> None:
        """Штамповать N квадратов/дисков в buf за векторный проход."""
        n = len(px)
        h, w, _ = buf.shape

        # Сетка смещений одного ядра
        if symbol == 'circle':
            kernel = _circle_kernel(cell)  # cell×cell, 0/1
            dy_grid, dx_grid = np.nonzero(kernel)
            # nonzeros в C-order: (dy, dx) отсортированы по строкам
        else:
            half = cell // 2
            offs = np.arange(-half, cell - half)  # cell значений
            dx_grid, dy_grid = np.meshgrid(offs, offs)  # (cell, cell)
            dx_grid = dx_grid.ravel()
            dy_grid = dy_grid.ravel()

        k = len(dx_grid)

        # (N, k) экранные координаты пикселей ядра
        gx = (px[:, None] + dx_grid[None, :]).astype(np.int64).ravel()
        gy = (py[:, None] + dy_grid[None, :]).astype(np.int64).ravel()

        # Цвет: (N,3) → размножить на k → (N*k, 3)
        gcol = np.broadcast_to(
            rgb[:, None, :], (n, k, 3),
        ).reshape(-1, 3)

        # В координаты буфера + клипинг
        lx = gx - x0
        ly = gy - y0
        valid = (lx >= 0) & (lx < w) & (ly >= 0) & (ly < h)

        # Присваивание с возможными повторами индексов:
        # CPython «последний выигрывает», а порядок — back→front,
        # поэтому ближние точки корректно перекрывают дальние.
        buf[ly[valid], lx[valid]] = gcol[valid]
        # Помечаем эти пиксели как нарисованные (для защиты от коллизий
        # с color-key: фиксим ТОЛЬКО их, фон не трогаем).
        drawn[ly[valid], lx[valid]] = True

    # ── Векторная заливка символьного слоя (по группам символов) ─────────

    @staticmethod
    def _stamp_char_layer(
        buf: NDArray[np.uint8],
        drawn: NDArray[np.bool_],
        x0: int, y0: int,
        px: NDArray[np.float64], py: NDArray[np.float64],
        rgb: NDArray[np.uint8], size: int,
        chars: Optional[List[str]],
    ) -> None:
        """Штамповать N символов в buf. Группировка по символу: для каждой
        уникальной буквы — одна векторная операция с её маской."""
        if chars is None:
            return
        n = len(px)
        if n == 0:
            return
        h, w, _ = buf.shape

        # Группируем индексы по символу → {символ: [индексы]}
        groups: dict[str, list[int]] = {}
        for i, ch in enumerate(chars):
            groups.setdefault(ch, []).append(i)

        for ch, idxs in groups.items():
            gpx = px[idxs]
            gpy = py[idxs]
            gcol = rgb[idxs]
            gn = len(idxs)
            # Маска глифа из кеша (растеризуется один раз)
            glyph = _rasterize_glyph(ch, size)
            if glyph is None:
                continue
            sdx, sdy, gw, gh, bx, by = glyph
            # bx/by — смещение центра растра; приводим смещения к центру глифа
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
            valid = (lx >= 0) & (lx < w) & (ly >= 0) & (ly < h)
            buf[ly[valid], lx[valid]] = cols[valid]
            drawn[ly[valid], lx[valid]] = True

    # ── Защита от случайных прозрачных пикселей ──────────────────────────

    @staticmethod
    def _fix_transparent_collisions(
        buf: NDArray[np.uint8],
        drawn: NDArray[np.bool_],
    ) -> None:
        """Пиксели частиц, совпавшие с color-key (0,0,1), чуть сдвигаем.
        Фон НЕ трогаем — он должен остаться прозрачным."""
        # Коллизия только среди реально нарисованных пикселей
        collision = drawn & (buf[..., 0] == 0) & (buf[..., 1] == 0) & (buf[..., 2] == 1)
        if collision.any():
            buf[collision, 2] = 2

    # ── Вывод одной картинкой ────────────────────────────────────────────

    def _blit(self, buf: NDArray[np.uint8], x0: int, y0: int) -> None:
        try:
            from PIL import Image, ImageTk
        except ImportError:
            return
        mode = 'RGBA' if buf.shape[-1] == 4 else 'RGB'
        img = Image.fromarray(buf, mode=mode)
        photo = ImageTk.PhotoImage(img)
        self._photo = photo  # удержать от сборки мусора
        try:
            self._canvas.coords(self._image_item, x0, y0)
            self._canvas.itemconfig(self._image_item, image=photo)
            # Z-порядок: поднимаем картинку ПОД drag_handle НЕ нужен
            # (drag_handle — tag_lower'd). Поднимаем её над drag_handle
            # чтобы частицы были поверх чёрного полигона:
            try:
                self._canvas.tag_raise(self._image_item, 'drag_handle')
            except Exception:
                pass  # drag_handle может не существовать
        except Exception:
            # canvas мог быть уничтожен
            pass


# ---------------------------------------------------------------------------
# Самотестирование: python renderer.py
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import time
    import tkinter as tk

    # ── Тест производительности заливки (без GUI) ────────────────────────
    def bench(n: int, cell: int = 6) -> float:
        rng = np.random.default_rng(0)
        px = rng.uniform(100, 700, n)
        py = rng.uniform(100, 700, n)
        rgb = rng.integers(0, 256, (n, 3), dtype=np.uint8)
        buf = np.full((800, 800, 3), TRANSPARENT_RGB, dtype=np.uint8)
        drawn = np.zeros((800, 800), dtype=np.bool_)
        t0 = time.perf_counter()
        for _ in range(20):
            PointCloudRenderer._stamp_layer(
                buf, drawn, 0, 0, px, py, rgb, cell, 'square')
        dt = (time.perf_counter() - t0) / 20
        return dt

    for n in (2400, 26000):
        ms = bench(n) * 1000
        print(f"  stamp {n:6d} points (cell=6): {ms:6.2f} ms/frame")

    # ── Визуальный тест (Tk окно) ────────────────────────────────────────
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
        # depth-sort (для порядка перекрытия)
        z = np.sin(ang * 5)
        order = np.argsort(z)
        px, py = px[order], py[order]
        rgb = np.zeros((n, 3), dtype=np.uint8)
        rgb[:, 0] = (128 + 127 * np.sin(ang)).astype(np.uint8)
        rgb[:, 1] = (128 + 127 * np.cos(ang)).astype(np.uint8)
        rgb[:, 2] = 200

        r.begin_frame()
        r.add_points(px, py, rgb, 6, 'square')
        r.finish_frame()
        frame[0] += 1
        root.after(33, tick)

    root.after(100, tick)
    print("\nОткрываю окно smoke-test. Закрой окно для выхода.")
    root.mainloop()
