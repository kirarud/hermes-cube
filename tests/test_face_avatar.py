"""Hermes Face regression checks, using built-in unittest."""
import unittest
import numpy as np
from core.face_avatar import generate_face, regions
from core.face_themes import face_colors, THEMES
from core.systems.grid_generator import _generate_cube_grid
from core.pipeline import build_default_pipeline
from core.world import World

class FaceTests(unittest.TestCase):
    def test_deterministic(self):
        src=_generate_cube_grid(12)
        face=generate_face(src)
        self.assertEqual(face.shape,src.shape)
        self.assertTrue(np.isfinite(face).all())
        np.testing.assert_array_equal(face,generate_face(src))
        a,b,c,d,n=regions(len(face))
        self.assertGreater(a,0)
        self.assertGreater(c-b,20)
        self.assertTrue((face[b:c,2] > 0.58).all())
        self.assertGreater(face[:,1].max(),1.2)
        self.assertLess(face[:,1].min(),-0.9)

    def test_themes_and_eye_brightness(self):
        face=generate_face(_generate_cube_grid(12))
        a,b,c,d,n=regions(len(face))
        for name in THEMES:
            color=face_colors(face,name)
            self.assertEqual(color.shape,(n,3))
            self.assertEqual(color.dtype,np.uint8)
            self.assertGreater(float(color[b:c].mean()),float(color[:a].mean()),name)
        self.assertGreater(face_colors(face,'Bioform')[:,0].mean(),face_colors(face,'Holographic')[:,0].mean())

    def test_pipeline_and_existing_forms(self):
        cfg=dict(window_width=700,window_height=700,particle_density=12,
            shape_preset='face',face_theme='Glass',morph_progress=1.0,
            particle_mode='breathe',wave_speed=1.5,wave_amp=.04,
            rotation_speed=.12,pulse_rate=1.8,pulse_amplitude=.08,
            cell_size=5,fit_to_window=True,color_mode='default')
        world=World.create(cfg)
        world.meta.w=700;world.meta.h=700;world.meta.t=12
        pipeline=build_default_pipeline()
        pipeline.run(world,.016)
        self.assertEqual(world.sim.active_count,864)
        self.assertIn('face',world.sim.shape_cache)
        self.assertEqual(world.render.final_rgb.shape,(864,3))
        self.assertTrue(np.isfinite(world.render.projected_x).all())
        self.assertTrue(np.isfinite(world.render.projected_y).all())
        cfg['shape_preset']='cube';cfg['morph_progress']=0.0
        pipeline.run(world,.016)
        np.testing.assert_array_equal(world.sim.morphed[:864],world.sim.base_position[:864])

if __name__=='__main__':unittest.main()
