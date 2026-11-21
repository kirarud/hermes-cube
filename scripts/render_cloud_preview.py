"""Render a faithful headless cloud-vs-cube image using the real Hermes pipeline."""
from pathlib import Path
import sys
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.world import World
from core.pipeline import build_default_pipeline
from cube_app import DEFAULT_CONFIG

W,H = 560,560
OUT=Path(__file__).resolve().parents[1] / 'screenshots' / 'cloud-preview.png'
def frame(shape,morph):
    cfg = dict(DEFAULT_CONFIG, shape_preset=shape, morph_progress=morph,
        color_mode='default', particle_mode='breathe',
        symbol='circle', cell_size=7, particle_density=20,
        cube_scale=0.37, pulse_amplitude=0.03, wave_amp=0.045)
    world=World.create(cfg,n_particles=2400)
    world.meta.w=W; world.meta.h=H; world.meta.t=1.7
    build_default_pipeline().run(world,0.042)
    im=Image.new('RGB',(W,H),(9,15,30))
    dr=ImageDraw.Draw(im,'RGBA')
    x=world.render.projected_x; y=world.render.projected_y
    z=world.render.depth; col=world.render.final_rgb
    order=np.argsort(z)
    for i in order:
        c=tuple(map(int,col[i]))
        ix=float(x[i]);iy=float(y[i])
        radius=3.2 if shape=='cloud' else 2.5
        if 4<ix<W-4 and 4<iy<H-4:
            dr.ellipse((ix-radius,iy-radius,ix+radius,iy+radius),fill=(*c,190 if shape=='cloud' else 235))
    dr.text((24,20),'HERMES CUBE' if shape=='cube' else 'HERMES CLOUD',fill=(230,240,255,255))
    dr.text((24,H-35),'shape_preset = '+shape,fill=(138,163,190,255))
    return im
out=Image.new('RGB',(W*2+20,H),(6,10,20))
out.paste(frame('cube',0),(0,0))
out.paste(frame('cloud',1),(W+20,0))
OUT.parent.mkdir(parents=True,exist_ok=True)
out.save(OUT)
print('Preview written:',OUT, 'pixels:',out.size)
