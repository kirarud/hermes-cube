"""test_projection_fit.py — куб не должен обрезаться «невидимой рамкой».

projection.py масштабирует по доле окна (cube_scale). При большом значении
частицы уезжали за границы FBO и просто исчезали. fit_to_window зажимает
масштаб так, чтобы всё осталось в кадре.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from core.world import World
from core.systems import projection


def _make_world(scale: float, w: int = 600, h: int = 600,
                fit: bool = True, cell: int = 6) -> World:
    world = World.create({}, n_particles=8)
    # Куб с вершинами ±1 — максимально далёкие частицы
    corners = np.array([
        [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1],
    ], dtype=np.float64)
    world.sim.world_position[:8] = corners
    world.sim.active_count = 8
    world.meta.w = w
    world.meta.h = h
    world.meta.t = 0.0
    world.meta.config = {
        'cube_scale': scale,
        'pulse_rate': 1.8,
        'pulse_amplitude': 0.12,
        'cell_size': cell,
        'fit_to_window': fit,
    }
    return world


class TestProjectionFit(unittest.TestCase):

    def test_large_scale_stays_in_window(self):
        """cube_scale=0.9 не должен выкидывать частицы за края окна."""
        world = _make_world(0.9, w=600, h=600, fit=True)
        projection.update(world, 0.016)
        px = world.render.projected_x
        py = world.render.projected_y
        self.assertGreaterEqual(float(px.min()), 0.0, "частица уехала влево за кадр")
        self.assertLessEqual(float(px.max()), 600.0, "частица уехала вправо за кадр")
        self.assertGreaterEqual(float(py.min()), 0.0, "частица уехала вверх за кадр")
        self.assertLessEqual(float(py.max()), 600.0, "частица уехала вниз за кадр")

    def test_extreme_scale_clamped(self):
        """Даже cube_scale=5.0 остаётся внутри кадра."""
        world = _make_world(5.0, w=800, h=600, fit=True)
        projection.update(world, 0.016)
        px = world.render.projected_x
        py = world.render.projected_y
        self.assertGreaterEqual(float(px.min()), 0.0)
        self.assertLessEqual(float(px.max()), 800.0)
        self.assertGreaterEqual(float(py.min()), 0.0)
        self.assertLessEqual(float(py.max()), 600.0)

    def test_small_scale_untouched(self):
        """Маленький масштаб не должен трогаться clamp-ом."""
        world_fit = _make_world(0.15, fit=True)
        world_raw = _make_world(0.15, fit=False)
        projection.update(world_fit, 0.016)
        projection.update(world_raw, 0.016)
        np.testing.assert_allclose(
            world_fit.render.projected_x,
            world_raw.render.projected_x,
            rtol=1e-9,
            err_msg="clamp сработал там, где не нужно",
        )

    def test_fit_disabled_allows_overflow(self):
        """fit_to_window=False → старое поведение (для совместимости)."""
        world = _make_world(0.9, w=600, h=600, fit=False)
        projection.update(world, 0.016)
        px = world.render.projected_x
        self.assertTrue(
            float(px.min()) < 0.0 or float(px.max()) > 600.0,
            "без fit_to_window частицы должны выходить за кадр",
        )

    def test_centered(self):
        """Центр масс остаётся в центре окна."""
        world = _make_world(0.9, w=600, h=400, fit=True)
        projection.update(world, 0.016)
        cx = float(world.render.projected_x.mean())
        cy = float(world.render.projected_y.mean())
        self.assertAlmostEqual(cx, 300.0, delta=1.0)
        self.assertAlmostEqual(cy, 200.0, delta=1.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
