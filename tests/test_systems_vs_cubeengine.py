"""tests/test_systems_vs_cubeengine.py — Верификация Systems.

Сравнивает output новых Systems и старого CubeEngine поэлементно.
Убеждаемся, что Rotation → Morph → Animation → Color → Projection
дают идентичный результат.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from core.world import World
from core.systems import grid_generator, rotation, morph, animation, color, projection
from cube_app import CubeEngine, DEFAULT_CONFIG


def _setup_world() -> World:
    cfg = dict(DEFAULT_CONFIG)
    cfg['particle_density'] = 12
    cfg['shape_preset'] = 'sphere'
    cfg['morph_progress'] = 0.5
    cfg['particle_mode'] = 'wave'
    cfg['cube_scale'] = 0.27
    cfg['rotation_speed'] = 0.28
    cfg['pulse_rate'] = 1.8
    cfg['pulse_amplitude'] = 0.12
    cfg['wave_speed'] = 1.5
    cfg['wave_amp'] = 0.12
    return World.create(cfg, n_particles=864, pool_size=4096)


def test_identity() -> None:
    """Systems output должен совпадать с CubeEngine.get_frame()."""
    cfg = dict(DEFAULT_CONFIG)
    cfg['particle_density'] = 12
    cfg['shape_preset'] = 'sphere'
    cfg['morph_progress'] = 0.5
    cfg['particle_mode'] = 'wave'
    cfg['cube_scale'] = 0.27
    cfg['rotation_speed'] = 0.28
    cfg['pulse_rate'] = 1.8
    cfg['pulse_amplitude'] = 0.12
    cfg['wave_speed'] = 1.5
    cfg['wave_amp'] = 0.12

    # --- CubeEngine reference ---
    engine = CubeEngine(cfg['particle_density'])
    t = 1.23
    pts3d_ref, pulse_ref = engine.get_frame(t, cfg)

    # Rebuild shape cache with current density
    engine.recalc(cfg)

    # --- Systems ---
    world = _setup_world()
    world.meta.t = t
    world.meta.dt = 0.042
    world.meta.w = 600
    world.meta.h = 600

    # Run systems (order must match CubeEngine.get_frame: morph → anim → rot)
    grid_generator.update(world, 0.042)
    morph.update(world, 0.042)
    animation.update(world, 0.042)
    rotation.update(world, 0.042)
    color.update(world, 0.042)
    projection.update(world, 0.042)

    # Compare position (most important)
    n = world.sim.active_count
    pos = world.sim.position[:n]
    ref_pos = pts3d_ref[:n]

    max_diff = np.max(np.abs(pos - ref_pos))
    print(f"Max position difference: {max_diff:.10f}")
    assert max_diff < 1e-10, f"Position mismatch: {max_diff}"

    # Compare depth
    depth = world.render.depth[:n]
    ref_depth = pts3d_ref[:n, 2]
    max_depth_diff = np.max(np.abs(depth - ref_depth))
    print(f"Max depth difference: {max_depth_diff:.10f}")
    assert max_depth_diff < 1e-10, f"Depth mismatch: {max_depth_diff}"

    # Compare projected (screen) positions
    ref_px = ref_pos[:, 0] * (min(600, 600) * 0.27 / (1.0 + 0.12) * pulse_ref) + 600/2
    ref_py = ref_pos[:, 1] * (min(600, 600) * 0.27 / (1.0 + 0.12) * pulse_ref) + 600/2
    max_px_diff = np.max(np.abs(world.render.projected_x[:n] - ref_px))
    max_py_diff = np.max(np.abs(world.render.projected_y[:n] - ref_py))
    print(f"Max projected_x difference: {max_px_diff:.10f}")
    print(f"Max projected_y difference: {max_py_diff:.10f}")
    assert max_px_diff < 1e-10, f"Projected X mismatch: {max_px_diff}"
    assert max_py_diff < 1e-10, f"Projected Y mismatch: {max_py_diff}"

    print("\n✅ ALL IDENTITY TESTS PASSED")


def test_all_presets() -> None:
    """Проверить все 5 пресетов форм на корректную генерацию."""
    for preset in ['cube', 'sphere', 'torus', 'dna', 'metaball']:
        cfg = dict(DEFAULT_CONFIG)
        cfg['particle_density'] = 12
        cfg['shape_preset'] = preset
        cfg['morph_progress'] = 1.0
        world = World.create(cfg, n_particles=864)
        grid_generator.update(world, 0.0)
        n = world.sim.active_count
        assert n == 864, f"{preset}: expected 864 particles, got {n}"
        pos = world.sim.position[:n]
        assert not np.any(np.isnan(pos)), f"{preset}: NaN in position"
        assert not np.any(np.isinf(pos)), f"{preset}: Inf in position"
        print(f"  ✅ {preset}: {n} particles, no NaN/Inf")


def test_animation_does_not_break() -> None:
    """Анимации не должны давать NaN/Inf."""
    for mode in ['off', 'wave', 'breathe', 'orbit', 'geyser']:
        cfg = dict(DEFAULT_CONFIG)
        cfg['particle_density'] = 12
        cfg['particle_mode'] = mode
        cfg['wave_amp'] = 0.3
        world = World.create(cfg, n_particles=864)
        world.meta.t = 5.0
        grid_generator.update(world, 0.042)
        animation.update(world, 0.042)
        n = world.sim.active_count
        pos = world.sim.position[:n]
        assert not np.any(np.isnan(pos)), f"{mode}: NaN in position"
        assert not np.any(np.isinf(pos)), f"{mode}: Inf in position"
        print(f"  ✅ {mode}: no NaN/Inf")


if __name__ == '__main__':
    print("=== Identity test (v2 Systems vs CubeEngine) ===\n")
    test_identity()

    print("\n=== Shape presets ===\n")
    test_all_presets()

    print("\n=== Animation modes ===\n")
    test_animation_does_not_break()

    print("\n🎉 ALL TESTS PASSED")
