"""Headless cloud-avatar checks for the existing Hermes Cube pipeline."""

from __future__ import annotations

import numpy as np

from core.cloud_skin import generate_cloud
from core.systems.grid_generator import _generate_cube_grid
from core.world import World
from core.pipeline import build_default_pipeline
from cube_app import DEFAULT_CONFIG


def test_cloud_is_deterministic_and_lobed():
    source = _generate_cube_grid(12)
    cloud = generate_cloud(source)
    assert cloud.shape == source.shape
    assert np.isfinite(cloud).all()
    np.testing.assert_array_equal(cloud, generate_cloud(source))
    assert cloud[:, 0].min() < -0.7
    assert cloud[:, 0].max() > 0.7
    assert np.ptp(cloud[:, 1]) > 0.7


def test_cloud_pipeline_and_return_to_cube():
    cfg = dict(DEFAULT_CONFIG, shape_preset='cloud', morph_progress=1.0,
               color_mode='default', particle_mode='breathe', particle_density=12)
    world = World.create(cfg)
    world.meta.w = 700
    world.meta.h = 550
    world.meta.t = 0.7
    pipeline = build_default_pipeline()
    pipeline.run(world, 0.042)
    n = world.sim.active_count
    assert n == 12 * 12 * 6
    assert 'cloud' in world.sim.shape_cache
    assert world.sim.shape_cache['cloud'].shape == (n, 3)
    assert np.isfinite(world.render.projected_x).all()
    assert np.isfinite(world.render.projected_y).all()
    assert world.render.final_rgb.shape == (n, 3)
    # Cloud mode uses a cool-white palette rather than the existing rainbow.
    assert np.mean(world.render.final_rgb[:, 2]) > np.mean(world.render.final_rgb[:, 0])

    world.meta.config['shape_preset'] = 'cube'
    world.meta.config['morph_progress'] = 0.0
    pipeline.run(world, 0.042)
    np.testing.assert_allclose(world.sim.morphed[:n],
                               world.sim.base_position[:n], atol=1e-12)
