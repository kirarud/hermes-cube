"""probe_glyph_density.py — проверка: 1 частица = 1 глиф, без наслоения.

Главный баг «буквы слипаются, за ними другие поменьше»: маска сэмплировалась
плотнее, чем размер глифа на экране (cell_size), поэтому соседние частицы
рисовали свои символы поверх друг друга.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from core.text_layout import layout_text_mask, glyph_grid_cols


def probe(text: str, win: int, scale: float, cell: int, fill: float,
          n_particles: int = 2400) -> None:
    peak = win * scale
    print(f'--- "{text}"  окно={win} scale={scale} cell={cell} fill={fill}')

    # Старый путь: фиксированный кегль, без учёта cell_size
    pos_old, n_old, _ = layout_text_mask(text, n_particles, font_size=64, fill=fill)
    step_old = np.diff(np.unique(np.round(pos_old[:n_old, 0], 6)))
    px_old = step_old.min() * peak if len(step_old) else 0.0

    # Новый путь: grid_cols привязан к cell_size
    gc = glyph_grid_cols(win, scale, cell, fill)
    pos_new, n_new, _ = layout_text_mask(text, n_particles, fill=fill, grid_cols=gc)
    step_new = np.diff(np.unique(np.round(pos_new[:n_new, 0], 6)))
    px_new = step_new.min() * peak if len(step_new) else 0.0

    print(f'  grid_cols={gc}')
    print(f'  БЫЛО: частиц={n_old:5} шаг={px_old:6.2f}px '
          f'перекрытие={cell / px_old if px_old else 0:5.2f}x')
    print(f'  СТАЛО: частиц={n_new:5} шаг={px_new:6.2f}px '
          f'перекрытие={cell / px_new if px_new else 0:5.2f}x')

    width_px = (pos_new[:n_new, 0].max() - pos_new[:n_new, 0].min()) * peak
    print(f'  ширина надписи={width_px:.0f}px, влезает в {win}px: '
          f'{"да" if width_px <= win else "НЕТ"}')
    ok = px_new >= cell * 0.95
    print(f'  => {"OK — глифы не наслаиваются" if ok else "ПЛОХО — всё ещё плотно"}')
    print()


if __name__ == '__main__':
    probe('ПРИВЕТ КИРА', 550, 0.5, 12, 0.85)
    probe('ПРИВЕТ КИРА', 550, 0.5, 6, 0.92)
    probe('ДА', 550, 0.13, 6, 0.92)
    probe('Я ЖИВОЙ И ГОТОВ ГОВОРИТЬ С ТОБОЙ', 720, 0.45, 12, 0.9)
