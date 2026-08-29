"""text_layout.py — Раскладка текста в маску из частиц.

Генерирует bitmap текста через PIL, сэмплирует позиции пикселей
для морфинга частиц. Каждая частица = один пиксель буквы.

Возвращает позиции в [-1, 1] и bounding box для overlay текста.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont


def layout_text_mask(
    text: str,
    n_particles: int,
    font_name: str = 'Segoe UI',
    font_size: int = 42,
    oversample: float = 2.0,
    fill: float = 0.92,
    grid_cols: Optional[int] = None,
) -> Tuple[NDArray[np.float64], int, Tuple[float, float, float, float]]:
    """Сгенерировать маску текста из частиц.

    Аспект сохраняется: обе оси нормализуются ОДНИМ делителем, иначе буквы
    растягиваются по вертикали (было: X/512, Y/128 → растяжение 4×).

    Args:
        fill: какую долю полуразмера [-1,1] занимает текст по длинной оси.
        grid_cols: сколько ГЛИФОВ влезает в ширину надписи на экране.
            Если задано, bitmap рендерится ровно такой ширины — один белый
            пиксель = одна частица = один глиф. Без этого маска сэмплируется
            плотнее, чем размер глифа, и буквы наслаиваются друг на друга.

    Returns:
        positions: (n_particles, 3) — -1..1, Z=0 для текста
        n_used: сколько частиц занято текстом
        bbox: (center_x, center_y, width_norm, height_norm) для overlay текста
    """
    import os
    windir = os.environ.get('WINDIR', 'C:\\Windows')

    # Поиск шрифта
    font_path = os.path.join(windir, 'Fonts', 'segoeui.ttf')
    if not os.path.isfile(font_path):
        font_path = os.path.join(windir, 'Fonts', 'arial.ttf')

    if grid_cols is not None and grid_cols > 0:
        arr = _render_grid_bitmap(text, font_path, int(grid_cols), n_particles)
    else:
        arr = _render_bitmap(text, font_path, int(font_size))

    ys, xs = np.where(arr > 128)
    n_white = len(xs)

    if n_white == 0:
        # Пустой текст
        positions = np.zeros((n_particles, 3), dtype=np.float64)
        positions[:, 2] = 5.0  # скрыты
        return positions, 0, (0.0, 0.0, 0.1, 0.1)

    h, w = arr.shape

    # ── Аспект-корректная нормализация ───────────────────────────────────
    # Центрируем по середине bitmap и делим ОБЕ оси на один масштаб.
    # long_side = max(w, h) → текст впишется в [-fill, fill] по длинной оси.
    long_side = float(max(w, h))
    sx = (xs.astype(np.float64) - w / 2.0) / (long_side / 2.0) * fill
    sy = -(ys.astype(np.float64) - h / 2.0) / (long_side / 2.0) * fill

    # Субсэмплируем если пикселей больше, чем частиц
    n_text = min(n_white, n_particles)
    if n_text < n_white:
        # Равномерный отбор без потери формы (linspace, не целочисленный step)
        indices = np.linspace(0, n_white - 1, n_text).astype(np.int64)
    else:
        indices = np.arange(n_white)

    positions = np.zeros((n_particles, 3), dtype=np.float64)
    n_used = len(indices)

    # Текст — позиции с Z=0 (плоскость)
    positions[:n_used, 0] = sx[indices]
    positions[:n_used, 1] = sy[indices]

    # Лишние частицы — прячем за камерой, чтобы не засорять надпись «рамкой».
    if n_used < n_particles:
        positions[n_used:, 0] = 0.0
        positions[n_used:, 1] = 0.0
        positions[n_used:, 2] = 5.0

    # Bounding box для overlay текста (в нормализованных координатах)
    tw_norm = w / long_side * 2.0 * fill
    th_norm = h / long_side * 2.0 * fill

    return positions, n_used, (0.0, 0.0, tw_norm, th_norm)


def _render_bitmap(text: str, font_path: str, font_size: int) -> NDArray[np.uint8]:
    """Bitmap ровно по размеру текста (+ поля) — аспект не зависит от длины."""
    font = ImageFont.truetype(font_path, max(4, int(font_size)))
    probe = Image.new('L', (8, 8), 0)
    tb = ImageDraw.Draw(probe).textbbox((0, 0), text, font=font)
    tw = max(1, tb[2] - tb[0])
    th = max(1, tb[3] - tb[1])

    pad = max(1, font_size // 8)
    img = Image.new('L', (tw + pad * 2, th + pad * 2), 0)
    ImageDraw.Draw(img).text((pad - tb[0], pad - tb[1]), text, font=font, fill=255)
    return np.array(img, dtype=np.uint8)


def _render_grid_bitmap(
    text: str,
    font_path: str,
    grid_cols: int,
    n_particles: int,
    min_font: int = 10,
) -> NDArray[np.uint8]:
    """Bitmap, где 1 пиксель маски = 1 глиф на экране.

    Ограничение жёсткое: надпись должна уместиться в квадрат
    grid_cols × grid_cols пикселей маски (нормализация делит обе оси на
    длинную сторону). Одна строка из 11 символов при grid_cols=38 даёт ~3 px
    на букву — форма теряется, поэтому текст переносится по словам и
    использует вертикаль.

    min_font: ниже этого кегля буква в маске перестаёт быть узнаваемой.
    """
    box = max(8, min(grid_cols, 400))

    for size in range(min(box, 64), min_font - 1, -1):
        arr = _render_wrapped(text, font_path, size, box)
        if arr is None:
            continue
        h, w = arr.shape
        if w > box or h > box:
            continue
        if int((arr > 128).sum()) > n_particles:
            continue
        return arr

    # Не влезло даже минимальным кеглем — берём его и отдаём субсэмплингу
    arr = _render_wrapped(text, font_path, min_font, box, force=True)
    return arr if arr is not None else _render_bitmap(text, font_path, min_font)


def _wrap_lines(text: str, font: 'ImageFont.FreeTypeFont',
                max_px: int) -> Optional[List[str]]:
    """Разбить текст по словам так, чтобы каждая строка влезала в max_px."""
    probe = ImageDraw.Draw(Image.new('L', (8, 8), 0))

    def width(s: str) -> int:
        tb = probe.textbbox((0, 0), s, font=font)
        return tb[2] - tb[0]

    lines: List[str] = []
    current = ''
    for word in text.split():
        candidate = word if not current else f'{current} {word}'
        if width(candidate) <= max_px:
            current = candidate
            continue
        if current:
            lines.append(current)
        if width(word) > max_px:
            return None  # одно слово не влезает — нужен меньший кегль
        current = word
    if current:
        lines.append(current)
    return lines or None


def _render_wrapped(
    text: str,
    font_path: str,
    font_size: int,
    box: int,
    force: bool = False,
) -> Optional[NDArray[np.uint8]]:
    """Многострочный bitmap с переносом по словам, вписанный в box × box."""
    font = ImageFont.truetype(font_path, max(4, int(font_size)))
    lines = _wrap_lines(text, font, box)
    if lines is None:
        if not force:
            return None
        lines = [text]

    probe = ImageDraw.Draw(Image.new('L', (8, 8), 0))
    line_gap = max(1, font_size // 6)
    metrics = []
    for line in lines:
        tb = probe.textbbox((0, 0), line, font=font)
        metrics.append((line, tb))

    width = max(1, max(tb[2] - tb[0] for _, tb in metrics))
    heights = [tb[3] - tb[1] for _, tb in metrics]
    height = max(1, sum(heights) + line_gap * (len(metrics) - 1))

    pad = 1
    img = Image.new('L', (width + pad * 2, height + pad * 2), 0)
    draw = ImageDraw.Draw(img)
    y = pad
    for (line, tb), lh in zip(metrics, heights):
        lw = tb[2] - tb[0]
        x = pad + (width - lw) // 2  # центрируем строку
        draw.text((x - tb[0], y - tb[1]), line, font=font, fill=255)
        y += lh + line_gap

    return np.array(img, dtype=np.uint8)


def glyph_grid_cols(
    window_min_side: int,
    cube_scale: float,
    cell_size: int,
    fill: float,
) -> int:
    """Сколько глифов размера cell_size влезает в ширину надписи.

    projection.py: пик масштаба = min(w, h) * cube_scale, надпись занимает
    2 * fill в нормализованных координатах.
    """
    peak_scale = float(window_min_side) * float(cube_scale)
    px_span = 2.0 * float(fill) * peak_scale
    cols = int(px_span / max(1.0, float(cell_size)))
    return max(6, cols)
