"""core/effects.py — Post-processing эффекты для Render Graph.

Содержит функции-эффекты, которые можно использовать как Pass.
Каждая функция принимает RGBA-буфер и возвращает модифицированный буфер.

Доступные эффекты:
  - bloom: свечение ярких участков
  - gaussian_blur: размытие (box approximation)
  - depth_fog: дымка по глубине (требует depth-карту)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def bloom(rgba: NDArray[np.uint8], threshold: int = 200,
          radius: int = 3, intensity: float = 0.5) -> NDArray[np.uint8]:
    """Bloom-эффект: выделить яркие пиксели, размыть, наложить.

    Args:
        rgba: (H, W, 4) входной буфер
        threshold: порог яркости для выделения (0-255)
        radius: радиус размытия (box kernel size)
        intensity: сила наложения (0-1)

    Returns:
        (H, W, 4) буфер с bloom
    """
    if rgba.size == 0:
        return rgba

    # Выделить яркие пиксели
    brightness = rgba[..., :3].max(axis=2)
    mask = brightness > threshold

    if not mask.any():
        return rgba.copy()

    # Копия ярких пикселей
    bright = np.zeros_like(rgba)
    bright[mask] = rgba[mask]

    # Размытие (box blur, 1 проход)
    if radius > 1:
        kernel = np.ones((radius, radius), dtype=np.float32) / (radius * radius)
        from scipy import ndimage
        for c in range(3):
            bright[..., c] = ndimage.convolve(
                bright[..., c].astype(np.float32), kernel, mode='constant')

    # Наложение
    result = rgba.astype(np.float32)
    result[..., :3] += bright[..., :3].astype(np.float32) * intensity
    result = np.clip(result, 0, 255).astype(np.uint8)
    result[..., 3] = rgba[..., 3]  # сохранить alpha
    return result


def quick_blur(rgba: NDArray[np.uint8], radius: int = 3) -> NDArray[np.uint8]:
    """Быстрое размытие через box filter (numpy-only, без scipy)."""
    if rgba.size == 0 or radius < 2:
        return rgba.copy()

    result = rgba.copy().astype(np.float32)
    h, w = result.shape[:2]

    # Горизонтальный проход
    for c in range(3):
        temp = result[..., c].copy()
        for i in range(1, radius + 1):
            temp[:, :-i] += result[:, i:, c]
            temp[:, i:] += result[:, :-i, c]
        temp /= (2 * radius + 1)
        result[..., c] = temp

    return np.clip(result, 0, 255).astype(np.uint8)
