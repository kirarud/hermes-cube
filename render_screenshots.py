#!/usr/bin/env python3
"""Headless screenshot generator for Hermes Cube.

Uses CubeEngine directly (no tkinter, no window, no desktop capture).
Cycles all shapes × animation modes, creates clean composite grids.
"""

import os, sys, math, numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Any, List, Tuple

# Clean import — cube_app runs _check_single_instance() but that's fine here
os.environ['DISPLAY'] = ''  # ensure no display needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Remove lock so import doesn't exit
lock = os.path.join(os.environ.get('TEMP', '/tmp'), 'hermes_cube.lock')
if os.path.exists(lock):
    os.remove(lock)

from cube_app import (
    CubeEngine, MIN_DENSITY, MAX_DENSITY, DEFAULT_CONFIG,
    SHAPE_GENERATORS, PARTICLE_ANIMATORS,
)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'screenshots')
os.makedirs(OUT, exist_ok=True)

# ── Theme ─────────────────────────────────────────────────────────────
BG = (10, 10, 26)
LABEL = (200, 200, 210)
ACCENT = (233, 69, 96)
WHITE = (255, 255, 255)

def _get_font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def render_frame(
    engine: CubeEngine,
    config: Dict[str, Any],
    w: int = 800,
    h: int = 600,
    t: float = 0.0,
    cell_size: int = 5,
) -> Image.Image:
    """Render one cube frame → PIL image. Mirrors cube_app._render_frame()."""
    pts3d, pulse = engine.get_frame(t, config)

    scale = (min(w, h) * config.get('cube_scale', 0.27)
             / (1.0 + config.get('pulse_amplitude', 0.12))
             * pulse)
    cx = w / 2.0
    cy = h / 2.0

    px = pts3d[:, 0] * scale + cx
    py = pts3d[:, 1] * scale + cy
    pz = pts3d[:, 2]

    order = np.argsort(pz)
    px, py, pz = px[order], py[order], pz[order]

    depth = 0.6 + 0.4 * (pz + 1.0) / 2.0
    r = np.clip(engine.r0[order] * depth, 0, 255).astype(int)
    g = np.clip(engine.g0[order] * depth, 0, 255).astype(int)
    b = np.clip(engine.b0[order] * depth, 0, 255).astype(int)

    cs = max(2, cell_size)
    img = Image.new('RGBA', (w, h), BG + (255,))
    draw = ImageDraw.Draw(img)
    for i in range(len(px)):
        x, y = int(px[i]), int(py[i])
        if 0 <= x < w and 0 <= y < h:
            draw.rectangle([x - cs//2, y - cs//2, x + cs//2, y + cs//2],
                          fill=(int(r[i]), int(g[i]), int(b[i]), 255))
    return img


def label(img: Image.Image, text: str, xy=(20, 20), color=ACCENT, size=28):
    draw = ImageDraw.Draw(img)
    draw.text(xy, text, fill=color + (255,), font=_get_font(size))


def grid_render(
    engine: CubeEngine,
    shapes: List[str],
    anims: List[str],
    config: Dict[str, Any],
    cell: int = 300,
) -> Image.Image:
    """Grid of shape rows × animation columns."""
    rows, cols = len(shapes), len(anims)
    img = Image.new('RGBA', (cols * cell, rows * cell), BG + (255,))
    for ri, shp in enumerate(shapes):
        for ci, anm in enumerate(anims):
            cfg = dict(config)
            cfg['shape_preset'] = shp
            cfg['particle_mode'] = anm
            sub = render_frame(engine, cfg, cell, cell, t=0.0)
            sx, sy = ci * cell, ri * cell
            img.paste(sub, (sx, sy), sub)
            d = ImageDraw.Draw(img)
            if ci == 0:
                d.text((sx + 6, sy + 6), shp, fill=LABEL + (255,), font=_get_font(12))
            if ri == 0:
                d.text((sx + 6, sy + cell - 20), anm, fill=LABEL + (255,), font=_get_font(12))
    return img


# ═══════════════════════════════════════════════════════════════════════
# GENERATE
# ═══════════════════════════════════════════════════════════════════════

SHAPES = ['cube', 'sphere', 'torus', 'dna', 'metaball']
ANIMS = ['off', 'wave', 'breathe', 'orbit', 'geyser']
CHARS = ['dots', 'symbols', 'words', 'glow']
MORTH_TARGETS = ['sphere', 'torus', 'dna', 'metaball']
MORPH_STEPS = [0.0, 0.25, 0.5, 0.75, 1.0]
DENSITIES = [8, 12, 16, 20]

engine = CubeEngine(12)
print("Rendering screenshots...\n")

# ── 1. Hero — large standalone ───────────────────────────────────────
img = render_frame(engine, DEFAULT_CONFIG, 1200, 900, t=1.5, cell_size=6)
label(img, '♢ Hermes Cube', xy=(20, 20), size=36)
label(img, 'Desktop Particle Avatar', xy=(20, 62), color=LABEL, size=16)
img.save(os.path.join(OUT, 'hero.png'))
print("  hero.png")

# ── 2. All 5 shapes ──────────────────────────────────────────────────
sg = grid_render(engine, SHAPES, ['off'], DEFAULT_CONFIG, cell=400)
sg.save(os.path.join(OUT, 'shapes.png'))
print("  shapes.png  — 5 shape presets")

# ── 3. All 5 animation modes ─────────────────────────────────────────
ag = grid_render(engine, ['cube'], ANIMS, DEFAULT_CONFIG, cell=400)
ag.save(os.path.join(OUT, 'animations.png'))
print("  animations.png  — 5 animation modes")

# ── 4. Full 5×5 matrix ───────────────────────────────────────────────
mx = grid_render(engine, SHAPES, ANIMS, DEFAULT_CONFIG, cell=250)
label(mx, 'Формы × Анимации', xy=(20, 20), size=24)
mx.save(os.path.join(OUT, 'matrix.png'))
print("  matrix.png  — 5×5 shape × animation grid")

# ── 5. Morph ─────────────────────────────────────────────────────────
mc = len(MORPH_STEPS)
mr = len(MORTH_TARGETS)
morph = Image.new('RGBA', (mc * 260, mr * 260 + 40), BG + (255,))
for ti, target in enumerate(MORTH_TARGETS):
    for mi, m in enumerate(MORPH_STEPS):
        cfg = dict(DEFAULT_CONFIG)
        cfg['shape_preset'] = target
        cfg['morph_progress'] = m
        sub = render_frame(engine, cfg, 260, 260, t=0.5)
        sx, sy = mi * 260, ti * 260 + 40
        morph.paste(sub, (sx, sy), sub)
        d = ImageDraw.Draw(morph)
        if ti == 0:
            d.text((sx + 6, sy + 260 - 20), f'{int(m*100)}%', fill=LABEL + (255,), font=_get_font(11))
        if mi == 0:
            d.text((sx + 6, sy + 6), target, fill=LABEL + (255,), font=_get_font(11))
label(morph, 'Морфинг: куб → форма', xy=(20, 20), size=24)
morph.save(os.path.join(OUT, 'morph.png'))
print("  morph.png  — morph 0-100%")

# ── 6. AI mode ───────────────────────────────────────────────────────
cfg = dict(DEFAULT_CONFIG)
cfg['pulse_rate'] = 2.5
cfg['rotation_speed'] = 0.4
ai = render_frame(engine, cfg, 800, 600, t=2.0, cell_size=5)
label(ai, 'AI-режим', size=28)
label(ai, 'Напиши C — куб ответит и изменит настроение', xy=(20, 56), color=LABEL, size=14)
ai.save(os.path.join(OUT, 'ai.png'))
print("  ai.png  — AI chat concept")

# ── 7. Density comparison ────────────────────────────────────────────
dc = len(DENSITIES)
density = Image.new('RGBA', (dc * 350, 350), BG + (255,))
for di, dens in enumerate(DENSITIES):
    eng = CubeEngine(dens)
    sub = render_frame(eng, DEFAULT_CONFIG, 350, 350, t=0.5)
    sx = di * 350
    density.paste(sub, (sx, 0), sub)
    d = ImageDraw.Draw(density)
    d.text((sx + 6, 350 - 22), f'density={dens}', fill=LABEL + (255,), font=_get_font(12))
label(density, 'Плотность частиц', size=24)
density.save(os.path.join(OUT, 'density.png'))
print("  density.png  — density variation")

# ── 8. Each shape × wave animation ───────────────────────────────────
for shape in SHAPES:
    cfg = dict(DEFAULT_CONFIG)
    cfg['shape_preset'] = shape
    cfg['particle_mode'] = 'wave'
    img = render_frame(engine, cfg, 800, 600, t=0.0, cell_size=5)
    label(img, f'♢ {shape}', size=32)
    img.save(os.path.join(OUT, f'{shape}.png'))
    print(f"  {shape}.png")

# ── 9. Char modes ────────────────────────────────────────────────────
for cm in CHARS:
    cfg = dict(DEFAULT_CONFIG)
    cfg['char_mode'] = cm
    cfg['shape_preset'] = 'sphere'
    img = render_frame(engine, cfg, 800, 600, t=1.0)
    label(img, f'Char: {cm}', size=24)
    img.save(os.path.join(OUT, f'char_{cm}.png'))
    print(f"  char_{cm}.png")

# ── Clean up old real screenshots ─────────────────────────────────────
old_real = ['idle.png', 'settings.png', 'draggable.png', 'ai_input.png',
            'pixelgrid.png', 'agent.png', 'overlay.png', 'take_screenshots.py']
for f in old_real:
    p = os.path.join(OUT, f) if f.endswith('.png') else os.path.join(os.path.dirname(OUT), f)
    if os.path.exists(p):
        os.remove(p)
        print(f"  cleaned: {f}")

n = len([f for f in os.listdir(OUT) if f.endswith('.png')])
print(f"\nDone! {n} images in screenshots/")
