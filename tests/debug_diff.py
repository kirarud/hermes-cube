#!/usr/bin/env python3
"""Debug: compare Steps systems vs CubeEngine step by step."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from cube_app import CubeEngine, DEFAULT_CONFIG
from cube_app import _animate_wave, _rotate_x, _rotate_y, _rotate_z
from core.world import World
from core.systems import grid_generator as gg
from core.systems import morph, animation, rotation

cfg = dict(DEFAULT_CONFIG)
for k, v in [
    ('particle_density', 12), ('shape_preset', 'sphere'),
    ('morph_progress', 0.5), ('particle_mode', 'wave'),
    ('cube_scale', 0.27), ('rotation_speed', 0.28),
    ('pulse_rate', 1.8), ('pulse_amplitude', 0.12),
    ('wave_speed', 1.5), ('wave_amp', 0.12),
]:
    cfg[k] = v

n = 864
t = 1.23
engine = CubeEngine(cfg['particle_density'])
engine.recalc(cfg)

# --- Step 1: Cube grid ---
print("Comparing cube grids...")
pts1 = engine.pts[:n]

world = World.create(cfg, n_particles=n)
world.meta.t = t
gg.update(world, 0.042)
pts2 = world.sim.position[:n].copy()

diff = np.max(np.abs(pts1 - pts2))
print(f"  Grid diff: {diff}")
if diff > 1e-10:
    print(f"  pts1[:3]: {pts1[:3]}")
    print(f"  pts2[:3]: {pts2[:3]}")

# --- Step 2: Morph ---
print("Comparing morph...")
target1 = engine.shape_cache['sphere'][:n]
target2 = world.sim.shape_cache['sphere'][:n]
diff_t = np.max(np.abs(target1 - target2))
print(f"  Sphere target diff: {diff_t}")

morphed1 = engine.pts[:n] * (1.0 - 0.5) + target1 * 0.5
morph.update(world, 0.042)
morphed2 = world.sim.position[:n].copy()
diff_m = np.max(np.abs(morphed1 - morphed2))
print(f"  Morph diff: {diff_m}")

if diff_m > 1e-10:
    # Check if morph system is working at all
    print(f"  morphed1[:3]: {morphed1[:3]}")
    print(f"  morphed2[:3]: {morphed2[:3]}")

# --- Step 3: Animation ---
print("Comparing animation...")
anim1 = _animate_wave(morphed1, t, 1.5, 0.12)
animation.update(world, 0.042)
anim2 = world.sim.position[:n].copy()
diff_a = np.max(np.abs(anim1 - anim2))
print(f"  Animation diff: {diff_a}")

# --- Step 4: Rotation ---
print("Comparing rotation...")
speed = 0.28
ang_x = t * 0.20 * (speed / 0.28)
ang_y = t * speed
ang_z = t * 0.08 * (speed / 0.28)
rot1 = _rotate_x(anim1, ang_x)
rot1 = _rotate_y(rot1, ang_y)
rot1 = _rotate_z(rot1, ang_z)
rotation.update(world, 0.042)
rot2 = world.sim.position[:n].copy()
diff_r = np.max(np.abs(rot1 - rot2))
print(f"  Rotation diff: {diff_r}")

# Full reference
pts3d_ref, pulse_ref = engine.get_frame(t, cfg)
full_diff = np.max(np.abs(world.sim.position[:n] - pts3d_ref[:n]))
print(f"\n  FULL diff: {full_diff}")
