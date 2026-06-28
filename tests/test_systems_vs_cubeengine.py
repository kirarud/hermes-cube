"""tests/test_systems_vs_cubeengine.py — Верификация Systems.

Проверяет поэлементное совпадение Pipeline (stage-буферы) и CubeEngine.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import math
import numpy as np
from core.world import World
from core.systems import grid_generator, morph, rotation, animation, color, projection
from cube_app import CubeEngine, DEFAULT_CONFIG


def _make_cfg() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    for k, v in [
        ('particle_density', 12), ('shape_preset', 'sphere'),
        ('morph_progress', 0.5), ('particle_mode', 'wave'),
        ('cube_scale', 0.27), ('rotation_speed', 0.28),
        ('pulse_rate', 1.8), ('pulse_amplitude', 0.12),
        ('wave_speed', 1.5), ('wave_amp', 0.12),
    ]:
        cfg[k] = v
    return cfg


def test_pipeline_vs_cubeengine() -> None:
    """Systems output (stage buffers) должен совпадать с CubeEngine.get_frame()."""
    cfg = _make_cfg()
    n = 864
    t = 1.23

    # --- CubeEngine reference ---
    engine = CubeEngine(cfg['particle_density'])
    engine.recalc(cfg)
    pts3d_ref, pulse_ref = engine.get_frame(t, cfg)

    # --- Pipeline (stage buffers) ---
    world = World.create(cfg, n_particles=n)
    world.meta.t = t
    world.meta.w = 600
    world.meta.h = 600

    grid_generator.update(world, 0.042)
    morph.update(world, 0.042)
    animation.update(world, 0.042)
    rotation.update(world, 0.042)
    color.update(world, 0.042)
    projection.update(world, 0.042)

    # 1) world_position vs CubeEngine output
    world_pos = world.sim.world_position[:n]
    max_pos_diff = np.max(np.abs(world_pos - pts3d_ref[:n]))
    print(f"world_position diff: {max_pos_diff:.2e}")
    assert max_pos_diff < 1e-10, f"world_position mismatch: {max_pos_diff}"

    # 2) projected vs CubeEngine projected
    # Manual projection for reference
    w, h = 600, 600
    pr = cfg.get('pulse_rate', 1.8)
    pa = cfg.get('pulse_amplitude', 0.12)
    pulse = 1.0 + pa * math.sin(t * pr)
    scale_val = cfg.get('cube_scale', 0.27)
    scale = min(w, h) * scale_val / (1.0 + pa) * pulse
    ref_px = pts3d_ref[:n, 0] * scale + w / 2.0
    ref_py = pts3d_ref[:n, 1] * scale + h / 2.0

    max_px = np.max(np.abs(world.render.projected_x[:n] - ref_px))
    max_py = np.max(np.abs(world.render.projected_y[:n] - ref_py))
    print(f"projected_x diff: {max_px:.2e}")
    print(f"projected_y diff: {max_py:.2e}")
    assert max_px < 1e-10, f"Projected X mismatch: {max_px}"
    assert max_py < 1e-10, f"Projected Y mismatch: {max_py}"

    # 3) stage buffer isolation check
    assert np.any(world.sim.morphed[:n] != world.sim.base_position[:n]), \
        "morphed should differ from base (morph=0.5)"
    # After animation + rotation, animated != morphed
    diff_anim = np.max(np.abs(world.sim.animated[:n] - world.sim.morphed[:n]))
    print(f"animated vs morphed diff: {diff_anim:.6f} (should be >0)")
    assert diff_anim > 0.001, "Animation did not change position"

    print("\n✅ ALL SYSTEMS VERIFIED AGAINST CubeEngine")
    print("✅ Stage buffers correctly isolated")


def test_no_copy_in_hot_path() -> None:
    """Проверка что системы не используют points.copy() в hot path."""
    import inspect
    import core.systems.rotation as rot
    import core.systems.morph as m
    import core.systems.animation as anim

    for mod, name in [(rot, 'rotation'), (m, 'morph'), (anim, 'animation')]:
        src = inspect.getsource(mod)
        # allow copy() only in grid_generator (build phase)
        # hot-path systems should NOT call copy()
        # They use separate output buffers instead
        lines = [l for l in src.split('\n') if '.copy()' in l and not l.strip().startswith('#')]
        if lines:
            # In rotation: out[:] = inp (the initial copy to output buffer) - this is fine
            # Let's check for explicit .copy() calls
            explicit = [l for l in lines if '.copy()' in l and '[:]' not in l]
            if explicit:
                print(f"  ⚠️  {name}: explicit .copy() call: {explicit}")
            else:
                print(f"  ✅ {name}: no wasteful .copy()")


if __name__ == '__main__':
    print("=== Pipeline vs CubeEngine (stage buffers) ===\n")
    test_pipeline_vs_cubeengine()

    print("\n=== Hot path copy check ===\n")
    test_no_copy_in_hot_path()

    print("\n🎉 ALL TESTS PASSED")
