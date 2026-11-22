"""Friendly Hermes Face preserves the archived experimental form and particle IDs."""
import unittest

import numpy as np

from core.face_avatar import generate_face, regions
from core.friendly_face import generate_friendly_face, friendly_face_colors
from core.systems.grid_generator import _generate_cube_grid
from core.pipeline import build_default_pipeline
from core.world import World


class FriendlyFaceTests(unittest.TestCase):
    def setUp(self):
        self.points = _generate_cube_grid(12)
        self.face = generate_friendly_face(self.points)

    def test_compact_without_antenna_and_with_filled_eyes(self):
        a, b, c, d, n = regions(len(self.points))
        self.assertEqual(self.face.shape, self.points.shape)
        np.testing.assert_array_equal(self.face, generate_friendly_face(self.points))
        self.assertTrue(np.isfinite(self.face).all())
        self.assertLess(self.face[:, 1].max(), 1.28)
        self.assertGreater(self.face[:, 1].min(), -1.32)
        self.assertGreater(self.face[b:c, 2].min(), .7)
        self.assertAlmostEqual(np.ptp(self.face[c:d, 1]), 0, places=7)
        # Former antennae form small accents on each cheek, not long stalks.
        self.assertLess(self.face[d:n, 1].max(), .14)

    def test_soft_material_has_visible_pupils(self):
        a, b, c, d, n = regions(len(self.face))
        color = friendly_face_colors(self.face, 'Soft Glow')
        self.assertEqual(color.shape, (n, 3))
        self.assertEqual(color.dtype, np.uint8)
        self.assertGreater(np.max(color[b:c, :]), 240)
        self.assertLess(np.min(np.mean(color[b:c], axis=1)), 120)
        self.assertGreater(np.mean(color[a:b]), 90)

    def test_original_is_preserved_in_cache(self):
        cfg = {
            'particle_density': 12, 'shape_preset': 'face',
            'face_theme': 'Soft Glow', 'morph_progress': 1.,
            'particle_mode': 'off', 'rotation_speed': 0.12,
            'pulse_rate': 1.8, 'pulse_amplitude': .035,
            'cell_size': 5, 'color_mode': 'default',
        }
        world = World.create(cfg)
        world.meta.w = world.meta.h = 700
        pipeline = build_default_pipeline()
        pipeline.run(world, .016)
        np.testing.assert_array_equal(world.sim.shape_cache['face'], self.face)
        np.testing.assert_array_equal(world.sim.shape_cache['face_original'],
                                      generate_face(self.points))
        first_position = world.sim.world_position[:len(self.points)].copy()
        cfg['shape_preset'] = 'face_original'
        pipeline.run(world, .016)
        self.assertGreater(np.max(np.abs(first_position-
                            world.sim.world_position[:len(self.points)])), .05)
        self.assertTrue(np.isfinite(world.render.final_rgb).all())

    def test_smile_only_when_happy(self):
        cfg = {
            'particle_density': 12, 'shape_preset': 'face',
            'face_theme': 'Soft Glow', 'morph_progress': 1.,
            'particle_mode': 'off', 'rotation_speed': .12,
            'pulse_rate': 1.8, 'pulse_amplitude': .035,
            'cell_size': 5, 'color_mode': 'default',
        }
        world = World.create(cfg)
        world.meta.w = world.meta.h = 700
        pipeline = build_default_pipeline()
        pipeline.run(world, .016)
        _, _, c, d, _ = regions(len(self.points))
        idle_mouth = world.sim.morphed[c:d, 1].copy()
        world.meta.mood = 'happy'
        pipeline.run(world, .016)
        self.assertLess(world.sim.morphed[c:d, 1].mean(), idle_mouth.mean())
        world.meta.mood = 'idle'
        pipeline.run(world, .016)
        np.testing.assert_allclose(world.sim.morphed[c:d, 1], idle_mouth)


if __name__ == '__main__':
    unittest.main()
