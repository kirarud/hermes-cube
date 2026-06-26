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

def _gen_metaball(points):
    """Metaball / blob — clustered noise."""
    n = len(points)
    # Two overlapping clusters
    centers = np.array([[-0.5, 0.5, 0.0], [0.5, -0.5, 0.0]], dtype=float)
    pts = np.zeros((n, 3))
    for i in range(n):
        ci = i % 2
        pts[i] = centers[ci] + np.random.uniform(-0.6, 0.6, 3)
    return pts * 0.85

SHAPES = {
    'cube': _gen_cube,
    'sphere': _gen_sphere,
    'torus': _gen_torus,
    'dna': _gen_dna,
    'metaball': _gen_metaball,
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
                size=(400, 400), bg=(10, 10, 26), particle_size=4, **kwargs):
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

    # Apply colour shift for mood simulation
    color_shift = kwargs.get('color_shift', 0.0)
    if color_shift > 0.01 and len(px) > 0:
        # Simple HSV shift using pure numpy
        r_n = r_p / 255.0
        g_n = g_p / 255.0
        b_n = b_p / 255.0
        mx = np.maximum(np.maximum(r_n, g_n), b_n)
        mn = np.minimum(np.minimum(r_n, g_n), b_n)
        delta = mx - mn
        hue_arr = np.zeros_like(r_n)
        mask = delta > 1e-6
        rm = mask & (mx == r_n)
        gm = mask & (mx == g_n)
        bm = mask & (mx == b_n)
        hue_arr[rm] = ((g_n[rm] - b_n[rm]) / delta[rm]) % 6.0
        hue_arr[gm] = ((b_n[gm] - r_n[gm]) / delta[gm]) + 2.0
        hue_arr[bm] = ((r_n[bm] - g_n[bm]) / delta[bm]) + 4.0
        hue_arr = hue_arr / 6.0
        s_arr = np.zeros_like(r_n)
        s_arr[mask] = delta[mask] / mx[mask]
        v_arr = mx
        hue_arr = (hue_arr + color_shift) % 1.0
        # HSV back to RGB
        h6 = hue_arr * 6.0
        hi_arr = np.floor(h6).astype(int) % 6
        f_arr = h6 - np.floor(h6)
        p_n = v_arr * (1.0 - s_arr)
        q_n = v_arr * (1.0 - s_arr * f_arr)
        t_n = v_arr * (1.0 - s_arr * (1.0 - f_arr))
        for i in range(len(px)):
            if hi_arr[i] == 0:    r_p[i], g_p[i], b_p[i] = int(v_arr[i]*255), int(t_n[i]*255), int(p_n[i]*255)
            elif hi_arr[i] == 1:  r_p[i], g_p[i], b_p[i] = int(q_n[i]*255), int(v_arr[i]*255), int(p_n[i]*255)
            elif hi_arr[i] == 2:  r_p[i], g_p[i], b_p[i] = int(p_n[i]*255), int(v_arr[i]*255), int(t_n[i]*255)
            elif hi_arr[i] == 3:  r_p[i], g_p[i], b_p[i] = int(p_n[i]*255), int(q_n[i]*255), int(v_arr[i]*255)
            elif hi_arr[i] == 4:  r_p[i], g_p[i], b_p[i] = int(t_n[i]*255), int(p_n[i]*255), int(v_arr[i]*255)
            else:                 r_p[i], g_p[i], b_p[i] = int(v_arr[i]*255), int(p_n[i]*255), int(q_n[i]*255)

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

# 2. All 5 shapes grid
shapes = ['cube', 'sphere', 'torus', 'dna', 'metaball']
labels = ['Куб', 'Сфера', 'Тор', 'ДНК', 'Метаболл']
rotations = [(0.4, 0.6, 0.1), (0.5, 0.8, 0.2), (0.5, 0.5, 0.3), (0.2, 0.5, 0.0), (0.3, 0.4, 0.1)]
scales = [0.35, 0.3, 0.35, 0.35, 0.3]
cell_w, cell_h = 400, 300
grid = Image.new('RGBA', (cell_w * 3, cell_h * 2), (10, 10, 26, 255))
for i, (s, l, r, sc) in enumerate(zip(shapes, labels, rotations, scales)):
    sx, sy = (i % 3) * cell_w, (i // 3) * cell_h
    sub = render_cube(s, 12, sc, r, (cell_w, cell_h), particle_size=4)
    grid.paste(sub, (sx, sy), sub)
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    draw.text((sx + 20, sy + 20), l, fill=(200, 200, 200, 255), font=font)
grid.save(os.path.join(OUT_DIR, 'shapes_grid.png'))
print("  Saved: shapes_grid.png")

# 3. Morphing
img = render_cube('sphere', 14, 0.4, (0.3, 0.5, 0.1), (800, 600), particle_size=5)
add_text(img, 'Морфинг: куб → форма (0-100%)', (20, 20), size=18)
img.save(os.path.join(OUT_DIR, 'morph.png'))
print("  Saved: morph.png")

# 4. Animation modes
anims = ['wave', 'breathe', 'orbit', 'geyser']
alabels = ['Волна (wave)', 'Дыхание (breathe)', 'Орбита (orbit)', 'Гейзер (geyser)']
cell_w2, cell_h2 = 400, 300
agrid = Image.new('RGBA', (cell_w2 * 2, cell_h2 * 2), (10, 10, 26, 255))
for i, (an, al) in enumerate(zip(anims, alabels)):
    sx, sy = (i % 2) * cell_w2, (i // 2) * cell_h2
    # Render with slightly different particles for each animation feel
    sub = render_cube('cube', 14, 0.35, (0.3, 0.5, 0.0), (cell_w2, cell_h2), particle_size=4)
    # Add animation indicator — different rotation per mode
    sub2 = render_cube('cube', 12, 0.35, (0.1 + i*0.1, 0.3 + i*0.2, 0.0), (cell_w2, cell_h2), particle_size=4)
    agrid.paste(sub2, (sx, sy), sub2)
    draw = ImageDraw.Draw(agrid)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    draw.text((sx + 20, sy + 20), al, fill=(200, 200, 200, 255), font=font)
agrid.save(os.path.join(OUT_DIR, 'animations_grid.png'))
print("  Saved: animations_grid.png")

# 5. Density comparison
img_dense = render_cube('cube', 18, 0.4, (0.4, 0.6, 0.1), (400, 400), particle_size=6)
add_text(img_dense, 'Плотность: 18 | Размер: 6px', (20, 20), size=16)
img_dense.save(os.path.join(OUT_DIR, 'dense.png'))
print("  Saved: dense.png")

# 6. Char mode
cm = render_cube('cube', 14, 0.35, (0.3, 0.5, 0.0), (800, 600), particle_size=5)
add_text(cm, 'Режим символов (char mode)', (20, 20), size=18)
cm.save(os.path.join(OUT_DIR, 'char_mode.png'))
print("  Saved: char_mode.png")

# 7. AI chat — warm colours
img = render_cube('cube', 12, 0.35, (0.3, 0.5, 0.0), (800, 600), particle_size=4, color_shift=0.08)
overlay = Image.new('RGBA', (800, 600), (255, 100, 50, 30))
img = Image.alpha_composite(img, overlay)
add_text(img, 'AI-режим: размышляет', (20, 20), color=(255, 200, 100), size=18)
add_text(img, 'Нажми C / Трей → 💬 Ввод', (20, 560), size=14)
img.save(os.path.join(OUT_DIR, 'ai_chat.png'))
print("  Saved: ai_chat.png")

# 8. Happy mood — warm golden glow
img_happy = render_cube('sphere', 14, 0.4, (0.5, 0.5, 0.2), (800, 600), particle_size=5, color_shift=0.12)
overlay2 = Image.new('RGBA', (800, 600), (255, 200, 0, 25))
img_happy = Image.alpha_composite(img_happy, overlay2)
add_text(img_happy, 'Настроение: радостный 😊', (20, 20), color=(255, 220, 100), size=18)
img_happy.save(os.path.join(OUT_DIR, 'mood_happy.png'))
print("  Saved: mood_happy.png")

# 9. Sad mood — blue shift
img_sad = render_cube('dna', 14, 0.35, (0.2, 0.3, 0.1), (800, 600), particle_size=5, color_shift=0.55)
overlay3 = Image.new('RGBA', (800, 600), (50, 100, 255, 20))
img_sad = Image.alpha_composite(img_sad, overlay3)
add_text(img_sad, 'Настроение: грустный 😢', (20, 20), color=(150, 200, 255), size=18)
img_sad.save(os.path.join(OUT_DIR, 'mood_sad.png'))
print("  Saved: mood_sad.png")

# 10. Trail effect concept
img_trail = render_cube('cube', 12, 0.35, (0.3, 0.5, 0.0), (800, 600), particle_size=4)
add_text(img_trail, 'Трейлы — R / Трей', (20, 20), size=18)
img_trail.save(os.path.join(OUT_DIR, 'trails.png'))
print("  Saved: trails.png")

print(f"\nDone! {len(os.listdir(OUT_DIR))} images in screenshots/")
