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

# ── Стили отрисовки текста ─────────────────────────────────────────────
# Стиль НЕ меняет геометрию букв — он меняет то, КАКИЕ пиксели маски
# становятся частицами и с какой яркостью каждая частица светится.
# Переключается из трея (config['text_style']).
TEXT_STYLES: Tuple[str, ...] = (
    'solid',     # все пиксели буквы
    'outline',   # только контур
    'glow',      # все пиксели + мягкие точки (через char_mode='glow')
    'grid',      # разреженная сетка — «пиксель-арт»
    'wave',      # яркость гаснет слева направо
    'halftone',  # яркий контур + приглушённая заливка
)

TEXT_STYLE_LABELS: Dict[str, str] = {
    'solid':    'Заливка',
    'outline':  'Контур',
    'glow':     'Неон',
    'grid':     'Пиксель-сетка',
    'wave':     'Волна',
    'halftone': 'Контур + полутон',
}

DEFAULT_TEXT_STYLE: str = 'solid'


def layout_text_mask(
    text: str,
    n_particles: int,
    font_name: str = 'Segoe UI',
    font_size: int = 42,
    oversample: float = 2.0,
    fill: float = 0.92,
    grid_cols: Optional[int] = None,
) -> Tuple[NDArray[np.float64], int, Tuple[float, float, float, float]]:
    """Старая сигнатура без стиля: solid, веса отбрасываются."""
    positions, n_used, bbox, _ = layout_text_mask_styled(
        text, n_particles, font_name=font_name, font_size=font_size,
        oversample=oversample, fill=fill, grid_cols=grid_cols,
        style=DEFAULT_TEXT_STYLE,
    )
    return positions, n_used, bbox


def _edge_mask(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """True там, где пиксель буквы граничит с пустотой (4-связность)."""
    h, w = mask.shape
    padded = np.pad(mask, 1, mode='constant', constant_values=False)
    # Снаружи паддинг = False, поэтому AND даёт False на границе буквы:
    # «не все соседи закрашены» == «пиксель на контуре».
    all_solid = (
        padded[:h, 1:w + 1]        # сверху
        & padded[2:, 1:w + 1]      # снизу
        & padded[1:h + 1, :w]      # слева
        & padded[1:h + 1, 2:]      # справа
    )
    return mask & ~all_solid


def _style_mask(
    arr: NDArray[np.uint8],
    style: str,
    n_particles: int,
) -> Tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """Отбор пикселей под стиль + карта яркости.

    Возвращает (mask, weight_map) формы arr:
      mask       — какие пиксели становятся частицами,
      weight_map — множитель яркости 0..1 для каждой частицы.
    """
    solid = arr > 128
    wm = np.ones(arr.shape, dtype=np.float64)

    if style == 'outline':
        return solid & _edge_mask(solid), wm

    if style == 'grid':
        # Чем крупнее шаг, тем «пиксельнее» и дешевле по частицам.
        step = 1
        for s in (2, 3, 4, 6, 8):
            if int(solid[::s, ::s].sum()) <= n_particles:
                step = s
                break
        grid = np.zeros(arr.shape, dtype=bool)
        grid[::step, ::step] = solid[::step, ::step]
        return grid, wm

    if style == 'wave':
        cols = np.where(np.any(solid, axis=0))[0]
        if cols.size == 0:
            return solid, wm
        x0 = float(cols[0])
        span = max(1.0, float(cols[-1]) - x0)
        xs = np.arange(arr.shape[1], dtype=np.float64)
        t = np.clip((xs - x0) / span, 0.0, 1.0)
        # 1.0 у левого края → 0.35 у правого: фраза «выплывает» из куба
        return solid, np.ones(arr.shape, dtype=np.float64) * (1.0 - 0.65 * t)[None, :]

    if style == 'halftone':
        return solid, np.where(_edge_mask(solid), 1.0, 0.3)

    # 'solid' и 'glow' — все пиксели; мягкость glow даёт рендер (char_mode)
    return solid, wm


def layout_text_mask_styled(
    text: str,
    n_particles: int,
    font_name: str = 'Segoe UI',
    font_size: int = 42,
    oversample: float = 2.0,
    fill: float = 0.92,
    grid_cols: Optional[int] = None,
    style: str = DEFAULT_TEXT_STYLE,
) -> Tuple[NDArray[np.float64], int, Tuple[float, float, float, float],
           NDArray[np.float64]]:
    """Сгенерировать маску текста из частиц.

    Аспект сохраняется: обе оси нормализуются ОДНИМ делителем, иначе буквы
    растягиваются по вертикали (было: X/512, Y/128 → растяжение 4×).

    Args:
        fill: какую долю полуразмера [-1,1] занимает текст по длинной оси.
        grid_cols: сколько ГЛИФОВ влезает в ширину надписи на экране.
            Если задано, bitmap рендерится ровно такой ширины — один белый
            пиксель = одна частица = один глиф. Без этого маска сэмплируется
            плотнее, чем размер глифа, и буквы наслаиваются друг на друга.
        style: одно из TEXT_STYLES. Неизвестное значение → 'solid'.

    Returns:
        positions: (n_particles, 3) — -1..1, Z=0 для текста
        n_used: сколько частиц занято текстом
        bbox: (center_x, center_y, width_norm, height_norm) для overlay текста
        weights: (n_particles,) — яркость 0..1, множитель к цвету частицы
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

    if style not in TEXT_STYLES:
        style = DEFAULT_TEXT_STYLE
    mask, wm = _style_mask(arr, style, n_particles)

    ys, xs = np.where(mask)
    n_white = len(xs)

    if n_white == 0:
        # Пустой текст
        positions = np.zeros((n_particles, 3), dtype=np.float64)
        positions[:, 2] = 5.0  # скрыты
        return (positions, 0, (0.0, 0.0, 0.1, 0.1),
                np.ones(n_particles, dtype=np.float64))

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

    # Вес = яркость частицы. Незадействованные частицы спрятаны за камерой,
    # им яркость не важна — 1.0, чтобы цвет куба вернулся как был.
    weights = np.ones(n_particles, dtype=np.float64)
    weights[:n_used] = wm[ys[indices], xs[indices]]

    return positions, n_used, (0.0, 0.0, tw_norm, th_norm), weights


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
