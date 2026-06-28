"""tests/test_pipeline.py — Тестирование Pipeline.

1. Identity: Pipeline output идентичен ручному вызову тех же систем
2. Skipping: выключенный Stage не влияет на output
3. Timing: замер ms на каждый Stage
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import numpy as np

from core.world import World
from core.pipeline import Pipeline, Stage, Schedule, build_default_pipeline
import core.systems.grid_generator as gg
import core.systems.morph as morph
import core.systems.animation as anim
import core.systems.rotation as rot
import core.systems.trail as trail
import core.systems.color as color
import core.systems.projection as proj
from cube_app import CubeEngine, DEFAULT_CONFIG


def _make_world() -> World:
    cfg = dict(DEFAULT_CONFIG)
    for k, v in [
        ('particle_density', 12), ('shape_preset', 'sphere'),
        ('morph_progress', 0.5), ('particle_mode', 'wave'),
        ('cube_scale', 0.27), ('rotation_speed', 0.28),
        ('wave_amp', 0.12),
    ]:
        cfg[k] = v
    world = World.create(cfg, n_particles=864)
    world.meta.t = 1.23
    world.meta.w = 600
    world.meta.h = 600
    world.meta.config['trail_enabled'] = False
    return world


def _manual_run(world: World) -> None:
    """Ручной вызов систем в правильном порядке."""
    gg.update(world, 0.042)
    morph.update(world, 0.042)
    anim.update(world, 0.042)
    rot.update(world, 0.042)
    color.update(world, 0.042)
    proj.update(world, 0.042)


def test_pipeline_identity() -> None:
    """Pipeline output идентичен ручному вызову."""
    w1 = _make_world()
    w2 = _make_world()

    _manual_run(w1)

    pipeline = build_default_pipeline()
    pipeline.run(w2, 0.042)

    n = w1.sim.active_count
    diff_pos = np.max(np.abs(w1.sim.position[:n] - w2.sim.position[:n]))
    diff_rgb = np.max(np.abs(
        w1.render.final_rgb[:n].astype(np.float64)
        - w2.render.final_rgb[:n].astype(np.float64)
    ))
    diff_px = np.max(np.abs(w1.render.projected_x[:n] - w2.render.projected_x[:n]))

    print(f"  Position diff: {diff_pos:.2e}")
    print(f"  RGB diff:      {diff_rgb:.2e}")
    print(f"  Projected X:   {diff_px:.2e}")

    assert diff_pos < 1e-10, f"Position mismatch: {diff_pos}"
    assert diff_rgb < 1.0, f"RGB mismatch: {diff_rgb}"
    assert diff_px < 1e-10, f"Projected X mismatch: {diff_px}"
    print("  ✅ Pipeline identity: PASS")


def test_stage_skipping() -> None:
    """Выключенный View Stage не должен обновлять render.projected_x."""
    w = _make_world()
    gg.update(w, 0.042)  # set up initial state

    pipeline = build_default_pipeline()
    pipeline.enable_stage('view', False)  # disable view
    pipeline.run(w, 0.042)

    # Without view stage, projected_x should still be zeros (initial)
    assert np.all(w.render.projected_x == 0.0), "View Stage was not skipped!"
    print("  ✅ Stage skipping: PASS")


def test_pipeline_timing() -> None:
    """Замерить ms на каждый Stage и на весь Pipeline."""
    w = _make_world()
    pipeline = build_default_pipeline()

    # Warmup
    for _ in range(10):
        pipeline.run(w, 0.042)

    # Measure each stage individually
    for stage in pipeline.stages:
        times = []
        for _ in range(100):
            w2 = _make_world()
            gg.update(w2, 0.042)
            t0 = time.perf_counter()
            stage.run(w2, 0.042)
            times.append((time.perf_counter() - t0) * 1000)
        avg = np.mean(times)
        print(f"  Stage '{stage.name}': {avg:.3f} ms")

    # Total pipeline
    times = []
    for _ in range(100):
        w2 = _make_world()
        t0 = time.perf_counter()
        pipeline.run(w2, 0.042)
        times.append((time.perf_counter() - t0) * 1000)
    avg_total = np.mean(times)
    print(f"  Total pipeline: {avg_total:.3f} ms")
    assert avg_total < 100, f"Pipeline too slow: {avg_total:.1f} ms"
    print("  ✅ Pipeline timing: PASS")


if __name__ == '__main__':
    print("=== Pipeline Identity ===\n")
    test_pipeline_identity()

    print("\n=== Stage Skipping ===\n")
    test_stage_skipping()

    print("\n=== Pipeline Timing ===\n")
    test_pipeline_timing()

    print("\n🎉 ALL PIPELINE TESTS PASSED")
