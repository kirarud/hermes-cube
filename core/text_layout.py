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
) -> Tuple[NDArray[np.float64], int, Tuple[float, float, float, float]]:
    """Сгенерировать маску текста из частиц.

    Returns:
        positions: (n_particles, 3) — -1..1, Z=0 для текста, Z=±0.5 для рамки
        n_used: сколько частиц занято текстом
        bbox: (center_x, center_y, width_norm, height_norm) для overlay текста
    """
    import sys, os
    windir = os.environ.get('WINDIR', 'C:\\Windows')

    # Поиск шрифта
    font_path = os.path.join(windir, 'Fonts', 'segoeui.ttf')
    if not os.path.isfile(font_path):
        font_path = os.path.join(windir, 'Fonts', 'arial.ttf')

    font = ImageFont.truetype(font_path, font_size)

    # Рендерим текст в bitmap
    img_size = (512, 128)
    img = Image.new('L', img_size, 0)
    draw = ImageDraw.Draw(img)

    # Вычисляем реальный bounding box текста
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    # Центрируем
    ox = (img_size[0] - tw) // 2 - bbox[0]
    oy = (img_size[1] - th) // 2 - bbox[1]
    draw.text((ox, oy), text, font=font, fill=255)

    # Сэмплируем позиции белых пикселей
    arr = np.array(img, dtype=np.uint8)
    ys, xs = np.where(arr > 128)
    n_white = len(xs)

    if n_white == 0:
        # Пустой текст
        positions = np.zeros((n_particles, 3), dtype=np.float64)
        positions[:, 2] = 5.0  # скрыты
        return positions, 0, (0.0, 0.0, 0.1, 0.1)

    # Нормализуем в [-1, 1]
    # Y инвертируем (PIL Y сверху вниз)
    h, w = arr.shape
    px = xs.astype(np.float64) / w * 2.0 - 1.0
    py = 1.0 - ys.astype(np.float64) / h * 2.0

    # Субсэмплируем если слишком много пикселей
    n_text = min(n_white, n_particles - 20)  # оставляем минимум 20 на рамку
    if n_text < n_white:
        step = n_white // n_text
        indices = np.arange(0, n_white, step)[:n_text]
    else:
        indices = np.arange(n_white)

    positions = np.zeros((n_particles, 3), dtype=np.float64)
    n_used = len(indices)

    # Текст — позиции с Z=0
    positions[:n_used, 0] = px[indices]
    positions[:n_used, 1] = py[indices]

    # Рамка: оставшиеся частицы равномерно вокруг
    if n_used < n_particles:
        n_frame = n_particles - n_used
        rng = np.random.default_rng(seed=42)
        # Рамка за пределами текста
        attempts = 0
        frame_x_list = []
        frame_y_list = []
        while len(frame_x_list) < n_frame and attempts < n_frame * 10:
            fx = rng.uniform(-1.2, 1.2)
            fy = rng.uniform(-1.2, 1.2)
            # Проверяем что не внутри текста
            if n_used > 0:
                dists = np.sqrt(
                    (fx - positions[:n_used, 0]) ** 2 +
                    (fy - positions[:n_used, 1]) ** 2
                )
                if dists.min() < 0.06:
                    attempts += 1
                    continue
            frame_x_list.append(fx)
            frame_y_list.append(fy)
            attempts += 1

        n_frame_valid = min(len(frame_x_list), n_frame)
        if n_frame_valid > 0:
            positions[n_used:n_used + n_frame_valid, 0] = frame_x_list[:n_frame_valid]
            positions[n_used:n_used + n_frame_valid, 1] = frame_y_list[:n_frame_valid]
            positions[n_used:n_used + n_frame_valid, 2] = 0.3

    # Bounding box для overlay текста (в нормализованных координатах)
    tw_norm = tw / w * 2.0
    th_norm = th / h * 2.0
    cx_norm = (ox + bbox[0] + tw // 2) / w * 2.0 - 1.0
    cy_norm = 1.0 - (oy + bbox[1] + th // 2) / h * 2.0

    return positions, n_used, (cx_norm, cy_norm, tw_norm, th_norm)
