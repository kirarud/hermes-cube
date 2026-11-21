"""Render a headless cloud-skin comparison from the real pipeline (no GUI)."""
from __future__ import annotations

import os
import sys
from PIL import Image, ImageDraw, ImageFont
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from core.world import World
from core.pipeline import build_default_pipeline
from cube_app import DEFAULT_CONFIG

W, H = 600, 530
BACKGROUND = (8, 14, 31)
OUTPUT = os.path.join(ROOT, 'screenshots', 'cloud-preview.png')

def render(shape: str) -> Image.Image:
    cfg = dict(DEFAULT_CONFIG, shape_preset=shape, morph_progress=1.0,
               particle_mode='breathe', particle_density=18,
               rotation_speed=0.20, wave_amp=0.08, color_mode='default',
               cube_scale=0.41, cell_size=4)
    world = World.create(cfg, n_particles=18*18*6)
    world.meta.w, world.meta.h = W, H
    world.meta.t = 2.2
    build_default_pipeline().run(world, 0.042)
    img = Image.new('RGB', (W, H), BACKGROUND)
    draw = ImageDraw.Draw(img)
    n = world.sim.active_count
    order = np.argsort(world.render.depth[:n])
    xx = world.render.projected_x[:n]
    yy = world.render.projected_y[:n]
    rgb = world.render.final_rgb[:n]
    for i in order:
        x, y = float(xx[i]), float(yy[i])
        if 2 <= x < W-2 and 2 <= y < H-2:
            c = tuple(int(k) for k in rgb[i])
            draw.ellipse((x-2.1, y-2.1, x+2.1, y+2.1), fill=c)
    return img

def main():
    target = Image.new('RGB', (W*2 + 44, H+85), (7, 12, 26))
    d = ImageDraw.Draw(target)
    target.paste(render('cube'), (16, 55))
    target.paste(render('cloud'), (W+28, 55))
    d.text((23, 22), 'HERMES CUBE   /   ORIGINAL', fill=(167, 185, 225))
    d.text((W+36, 22), 'HERMES CUBE   /   CLOUD', fill=(194, 226, 252))
    d.text((20, H+62), 'HEADLESS PREVIEW  |  One particle engine, two forms', fill=(99, 132, 173))
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    target.save(OUTPUT)
    print(OUTPUT)
    print('size',target.size)

if __name__ == '__main__':
    main()
