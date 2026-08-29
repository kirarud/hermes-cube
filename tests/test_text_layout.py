"""test_text_layout.py — аспект и покрытие маски текста.

Проверяет главный баг «текст = набор пикселей»:
  1. аспект букв не растянут (обе оси делятся одним масштабом)
  2. лишние частицы спрятаны (Z=5), а не разбросаны «рамкой» по надписи
  3. символы под частицами идут слева-направо в порядке строки
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from core.text_layout import layout_text_mask


class TestTextLayoutAspect(unittest.TestCase):

    def test_aspect_not_stretched(self):
        """Один символ 'O' должен быть примерно круглым, не растянутым 4×."""
        pos, n_used, _ = layout_text_mask('O', 2400, font_size=42)
        self.assertGreater(n_used, 50)
        xs = pos[:n_used, 0]
        ys = pos[:n_used, 1]
        span_x = xs.max() - xs.min()
        span_y = ys.max() - ys.min()
        ratio = span_y / span_x
        # Буква O у Segoe UI выше, чем шире, но не в разы
        self.assertGreater(ratio, 0.8, f"O сплющена: ratio={ratio:.2f}")
        self.assertLess(ratio, 2.0, f"O растянута: ratio={ratio:.2f}")

    def test_wide_text_stays_wide(self):
        """Длинная строка должна быть шире, чем выше — аспект сохранён."""
        pos, n_used, _ = layout_text_mask('ПРИВЕТ МИР', 2400, font_size=42)
        xs = pos[:n_used, 0]
        ys = pos[:n_used, 1]
        span_x = xs.max() - xs.min()
        span_y = ys.max() - ys.min()
        self.assertGreater(
            span_x, span_y * 2.0,
            f"строка не шире, чем выше: {span_x:.2f} vs {span_y:.2f}",
        )

    def test_fits_in_bounds(self):
        """Текст не выходит за [-1, 1] — иначе обрежется краем окна."""
        pos, n_used, _ = layout_text_mask('ЖЖЖЖЖЖЖЖЖЖ', 2400, font_size=42)
        xs = pos[:n_used, 0]
        ys = pos[:n_used, 1]
        self.assertLessEqual(float(np.abs(xs).max()), 1.0)
        self.assertLessEqual(float(np.abs(ys).max()), 1.0)

    def test_unused_particles_hidden(self):
        """Лишние частицы уходят за камеру (Z=5), не сорят рамкой."""
        pos, n_used, _ = layout_text_mask('И', 2400, font_size=42)
        self.assertLess(n_used, 2400)
        hidden = pos[n_used:]
        self.assertTrue(np.all(hidden[:, 2] > 4.0), "лишние частицы не спрятаны")

    def test_no_stray_frame_inside(self):
        """В радиусе надписи не должно быть частиц с Z≈0.3 (старая «рамка»)."""
        pos, n_used, _ = layout_text_mask('ТЕСТ', 2400, font_size=42)
        z_used = pos[:n_used, 2]
        self.assertTrue(np.allclose(z_used, 0.0), "текст должен быть плоским Z=0")

    def test_empty_text(self):
        pos, n_used, _ = layout_text_mask('', 100, font_size=42)
        self.assertEqual(n_used, 0)
        self.assertTrue(np.all(pos[:, 2] > 4.0))

    def test_subsample_when_few_particles(self):
        """Мало частиц — форма сохраняется, ошибок нет."""
        pos, n_used, _ = layout_text_mask('ПРИВЕТ', 200, font_size=42)
        self.assertLessEqual(n_used, 200)
        self.assertGreater(n_used, 100)
        xs = pos[:n_used, 0]
        self.assertGreater(xs.max() - xs.min(), 1.0, "надпись сжалась в точку")


class TestSymbolMapping(unittest.TestCase):

    def test_symbol_follows_x_position(self):
        """Символ под частицей соответствует её горизонтальной позиции."""
        from core.systems.avatar_text import AvatarTextSystem, _build_char_map
        from core.world import World

        text = 'АБВГ'
        world = World.create({}, n_particles=2400)
        world.meta.config = {}
        pos, n_used, _ = layout_text_mask(text, world.sim.active_count, font_size=42)
        world.sim.shape_cache['text'] = pos.copy()

        sysm = AvatarTextSystem()
        sysm._map_text_to_symbols(world, text, n_used)

        char_map = _build_char_map()
        xs = pos[:n_used, 0]
        order = np.argsort(xs)
        # Самая левая частица → первый символ, самая правая → последний
        left_idx = int(world.sim.symbol_idx[order[0]])
        right_idx = int(world.sim.symbol_idx[order[-1]])
        self.assertEqual(left_idx, char_map['А'], "левая частица не 'А'")
        self.assertEqual(right_idx, char_map['Г'], "правая частица не 'Г'")


if __name__ == '__main__':
    unittest.main(verbosity=2)
