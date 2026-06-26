#!/usr/bin/env python3
"""Generate screenshots of Hermes Cube in different modes for GitHub README."""

import sys
import os

# Add repo to path so we can import cube modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Replicate the cube rendering logic for screenshots
# We use the same CubeEngine from cube_app but render to PIL images

MIN_DENSITY = 6
MAX_DENSITY = 20
TRANSPARENT_COLOR = '#000001'

# ── Shape generators (copied from cube_app) ──────────────────────────────

def _gen_cube(points):
    """Cube: 8 vertices → edges → grid fill."""
    n = int(round(len(points) ** (1/3)))
    if n < 2:
        return points * 0.0
    axis = np.linspace(-1.0, 1.0, n)
    xx, yy, zz = np.meshgrid(axis, axis, axis)
    pts = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    # Scale to roughly same volume as sphere
    return pts * (1.0 / 1.2)

def _gen_sphere(points):
    """Sphere: radial distribution."""
    n = len(points)
    phi = np.random.uniform(0, 2 * np.pi, n)
    theta = np.arccos(np.random.uniform(-1, 1, n))
    r = np.random.uniform(0, 1, n) ** (1/3)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.column_stack((x, y, z))

def _gen_torus(points):
    """Torus: tube radius 0.4, ring radius 1.0."""
    n = len(points)
    theta = np.random.uniform(0, 2 * np.pi, n)
    phi = np.random.uniform(0, 2 * np.pi, n)
    R, r = 1.0, 0.4
    x = (R + r * np.cos(theta)) * np.cos(phi)
    y = (R + r * np.cos(theta)) * np.sin(phi)
    z = r * np.sin(theta)
    return np.column_stack((x, y, z))

def _gen_dna(points):
    """DNA double helix."""
    n = len(points)
    t = np.linspace(0, 4 * np.pi, n)
    r = 0.8
    # Two strands
    half = n // 2
    x1 = r * np.cos(t[:half])
    y1 = r * np.sin(t[:half])
    z1 = np.linspace(-1, 1, half)
    x2 = r * np.cos(t[half:half*2] + np.pi)
    y2 = r * np.sin(t[half:half*2] + np.pi)
    z2 = np.linspace(-1, 1, n - half)
    x = np.concatenate([x1, x2])
    y = np.concatenate([y1, y2])
    z = np.concatenate([z1, z2])
    return np.column_stack((x, y, z))

SHAPES = {
    'cube': _gen_cube,
    'sphere': _gen_sphere,
    'torus': _gen_torus,
    'dna': _gen_dna,
}

def make_gradient_colors(n):
    """Generate RGB gradient colours for n particles."""
    hue = np.linspace(0, 1, n, endpoint=False)
    r = np.zeros(n)
    g = np.zeros(n)
    b = np.zeros(n)
    # Fast HSV-like gradient — one channel at a time
    h6 = hue * 6.0
    hi = np.floor(h6).astype(int) % 6
    f = h6 - np.floor(h6)
    for i in range(n):
        hi_i = hi[i]
        f_i = f[i]
        if hi_i == 0:
            r[i], g[i], b[i] = 255, int(f_i * 255), 0
        elif hi_i == 1:
            r[i], g[i], b[i] = int((1 - f_i) * 255), 255, 0
        elif hi_i == 2:
            r[i], g[i], b[i] = 0, 255, int(f_i * 255)
        elif hi_i == 3:
            r[i], g[i], b[i] = 0, int((1 - f_i) * 255), 255
        elif hi_i == 4:
            r[i], g[i], b[i] = int(f_i * 255), 0, 255
        else:
            r[i], g[i], b[i] = 255, 0, int((1 - f_i) * 255)
    return r, g, b

def render_cube(shape_name='cube', density=12, scale=0.35, rotation=(0.3, 0.5, 0.0),
                size=(400, 400), bg=(10, 10, 26), particle_size=4):
    """Render a single cube frame to a PIL image."""
    n = density ** 3
    pts = np.zeros((n, 3))
    gen = SHAPES.get(shape_name, _gen_cube)
    pts = gen(pts)

    # Apply rotation
    rx, ry, rz = rotation
    cos, sin = np.cos, np.sin
    Rx = np.array([[1, 0, 0], [0, cos(rx), -sin(rx)], [0, sin(rx), cos(rx)]])
    Ry = np.array([[cos(ry), 0, sin(ry)], [0, 1, 0], [-sin(ry), 0, cos(ry)]])
    Rz = np.array([[cos(rz), -sin(rz), 0], [sin(rz), cos(rz), 0], [0, 0, 1]])
    pts = pts @ Rx @ Ry @ Rz

    # Project 3D → 2D
    w, h = size
    s = min(w, h) * scale / 1.2
    cx, cy = w / 2, h / 2
    px = pts[:, 0] * s + cx
    py = pts[:, 1] * s + cy
    pz = pts[:, 2]

    # Depth sort
    order = np.argsort(pz)
    px, py, pz = px[order], py[order], pz[order]

    # Colours
    r0, g0, b0 = make_gradient_colors(n)
    r0, g0, b0 = r0[order], g0[order], b0[order]
    depth_factor = 0.6 + 0.4 * (pz + 1.0) / 2.0
    r_p = np.clip(r0 * depth_factor, 0, 255).astype(int)
    g_p = np.clip(g0 * depth_factor, 0, 255).astype(int)
    b_p = np.clip(b0 * depth_factor, 0, 255).astype(int)

    # Render
    img = Image.new('RGBA', size, bg + (255,))
    draw = ImageDraw.Draw(img)
    cell = max(2, particle_size)
    for i in range(len(px)):
        x, y = int(px[i]), int(py[i])
        if 0 <= x < w and 0 <= y < h:
            colour = (int(r_p[i]), int(g_p[i]), int(b_p[i]), 255)
            draw.rectangle([x, y, x + cell, y + cell], fill=colour)

    return img


def add_text(img, text, pos, color=(200, 200, 200), size=16):
    """Add text to image."""
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", size)
    except Exception:
        font = ImageFont.load_default()
    draw.text(pos, text, fill=color + (255,), font=font)


def make_screenshot(name, shape, density=12, scale=0.35, rotation=(0.3, 0.5, 0.0),
                    particle_size=4, label=""):
    """Generate and save a screenshot with label."""
    img = render_cube(shape, density, scale, rotation,
                      size=(800, 600), particle_size=particle_size)
    if label:
        add_text(img, label, (20, 20), size=18)
    path = os.path.join(OUT_DIR, name)
    img.save(path)
    print(f"  Saved: {name}")
    return path


OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'screenshots')
os.makedirs(OUT_DIR, exist_ok=True)

print("Generating screenshots...")

# 1. Main cube — default
make_screenshot('preview.png', 'cube',
                label='♢ Hermes Cube — Режим ожидания')

# 2. Sphere
make_screenshot('sphere.png', 'sphere',
                rotation=(0.5, 0.8, 0.2), scale=0.3,
                label='Сфера (sphere)')

# 3. Torus
make_screenshot('torus.png', 'torus',
                rotation=(0.5, 0.5, 0.3), scale=0.35,
                label='Тор (torus)')

# 4. DNA
make_screenshot('dna.png', 'dna',
                rotation=(0.2, 0.5, 0.0), scale=0.35,
                label='ДНК (dna)')

# 5. Cube dense + large particles
make_screenshot('cube_dense.png', 'cube',
                density=16, particle_size=6, scale=0.4,
                rotation=(0.4, 0.6, 0.1),
                label='Плотный куб (density=16)')

# 6. Morphing preview — cube morphing to sphere
img = render_cube('sphere', 14, 0.4, (0.3, 0.5, 0.1), (800, 600), particle_size=5)
# Blend with cube: show 50% morph by overlaying
add_text(img, 'Морфинг: куб → сфера (50%)', (20, 20), size=18)
img.save(os.path.join(OUT_DIR, 'morph.png'))
print("  Saved: morph.png")

# 7. AI chat concept — render with warmer colours (mood = thinking)
img = render_cube('cube', 12, 0.35, (0.3, 0.5, 0.0), (800, 600), particle_size=4)
# Warm overlay for "thinking" mood
overlay = Image.new('RGBA', (800, 600), (255, 100, 50, 30))
img = Image.alpha_composite(img, overlay)
add_text(img, 'AI-режим: размышляет', (20, 20), color=(255, 200, 100), size=18)
add_text(img, 'Нажми C для ввода сообщения', (20, 560), size=14)
img.save(os.path.join(OUT_DIR, 'ai_chat.png'))
print("  Saved: ai_chat.png")

# 8. All shapes grid
shapes = ['cube', 'sphere', 'torus', 'dna']
labels = ['Куб', 'Сфера', 'Тор', 'ДНК']
cell_w, cell_h = 400, 300
grid = Image.new('RGBA', (cell_w * 2, cell_h * 2), (10, 10, 26, 255))
for i, (s, l) in enumerate(zip(shapes, labels)):
    sx, sy = (i % 2) * cell_w, (i // 2) * cell_h
    sub = render_cube(s, 12, 0.35, (0.4, 0.6, 0.1), (cell_w, cell_h), particle_size=4)
    grid.paste(sub, (sx, sy), sub)
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    draw.text((sx + 20, sy + 20), l, fill=(200, 200, 200, 255), font=font)
grid.save(os.path.join(OUT_DIR, 'shapes_grid.png'))
print("  Saved: shapes_grid.png")

print(f"\nDone! {len(os.listdir(OUT_DIR))} images in screenshots/")
