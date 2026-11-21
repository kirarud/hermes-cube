"""Run with python -m unittest discover -s tests -p test_render_resolution.py."""
import unittest

from core.render_resolution import limit_dimensions, resolve_max_render_side


class RenderResolutionTests(unittest.TestCase):
    def test_819px_not_upscaled(self):
        self.assertEqual(limit_dimensions(819, 819), (819, 819))

    def test_square_is_capped(self):
        self.assertEqual(limit_dimensions(1600, 1600), (1024, 1024))

    def test_aspect_ratio_preserved(self):
        self.assertEqual(limit_dimensions(2560, 1600), (1024, 640))

    def test_ceiling_and_lower_cap(self):
        self.assertEqual(resolve_max_render_side(4096), 1024)
        self.assertEqual(resolve_max_render_side(768), 768)
        self.assertEqual(resolve_max_render_side('invalid'), 1024)


if __name__ == '__main__':
    unittest.main()
