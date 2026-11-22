"""Friendly, non-humanoid Hermes Face shape.

The original face is preserved in core.face_avatar.generate_face.
Particle slots stay compatible with the existing morphing and color pipeline.
No external assets or procedural randomness per frame.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from core.face_avatar import regions


def generate_friendly_face(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Make a rounded companion with soft eyes, neutral mouth, and cheek lights."""
    n = len(points)
    out = np.zeros((n, 3), dtype=np.float64)
    if n == 0:
        return out

    a, b, c, d, _ = regions(n)
    golden = np.pi * (3.0 - np.sqrt(5.0))

    # One soft orb, rather than a human bust: no neck, skull or antennae.
    # Existing body slots become the subtle outer rim of the same silhouette.
    if b:
        i = np.arange(b, dtype=np.float64) + .5
        radius = np.sqrt(i/b)
        angle = golden*i
        out[:b, 0] = .79*radius*np.cos(angle)
        out[:b, 1] = .88*radius*np.sin(angle)
        out[:b, 2] = .49 + .24*(1.0-radius**2)

    # Eyes are filled soft discs, not empty glowing rings.
    if c > b:
        i = np.arange(c-b, dtype=np.int64)
        left_count = max(1, (c-b)//2)
        left = i < left_count
        local = np.where(left, i, i-left_count)
        total = np.where(left, left_count, max(1, c-b-left_count))
        radius = .095*np.sqrt((local+.5)/total)
        angle = golden*local
        out[b:c, 0] = np.where(left, -.22, .22) + radius*np.cos(angle)
        out[b:c, 1] = .20 + radius*.9*np.sin(angle)
        out[b:c, 2] = .78

    # A discreet neutral mouth. A smile is only added when the mood is happy.
    if d > c:
        t = np.linspace(-1.0, 1.0, d-c)
        out[c:d, 0] = .105*t
        out[c:d, 1] = -.20
        out[c:d, 2] = .79

    # The former antenna particle slots become small round cheek accents.
    if n > d:
        i = np.arange(n-d, dtype=np.int64)
        left_count = max(1, (n-d)//2)
        left = i < left_count
        local = np.where(left, i, i-left_count)
        total = np.where(left, left_count, max(1,n-d-left_count))
        radius = .066*np.sqrt((local+.5)/total)
        angle = golden*local
        out[d:, 0] = np.where(left, -.40, .40) + radius*np.cos(angle)
        out[d:, 1] = -.08 + radius*np.sin(angle)
        out[d:, 2] = .77
    return out


def friendly_face_colors(shape: NDArray[np.float64], theme: str,
                         mood: str = 'idle') -> NDArray[np.uint8]:
    """Pastel body, visible gentle pupils and small, non-staring eye glints."""
    from core.face_themes import THEMES

    n = len(shape)
    if not n:
        return np.empty((0, 3), dtype=np.uint8)
    base, glow, eye = THEMES.get(theme, THEMES['Holographic'])
    base = np.asarray(base, dtype=np.float64)
    glow = np.asarray(glow, dtype=np.float64)
    eye = np.asarray(eye, dtype=np.float64)
    a, b, c, d, _ = regions(n)

    out = np.zeros((n, 3), dtype=np.float64)
    y = np.clip((shape[:a, 1]+.88)/1.76, 0, 1)
    z = np.clip((shape[:a, 2]+.49)/1.22, 0, 1)
    # Lighter head, with a warm/soft rim instead of a hollow hologram.
    shade = (.42 + .22*y + .23*z)[:, None]
    out[:a] = base[None, :] * (1-shade) + glow[None, :]*shade
    out[a:b] = base*.45 + glow*.55

    # Small dark pupils within pale, filled eye discs.
    eyes = shape[b:c]
    if len(eyes):
        left = eyes[:, 0] < 0
        dx = eyes[:, 0] - np.where(left, -.22, .22)
        dy = eyes[:, 1] - .20
        radial = np.sqrt(dx*dx + (dy/.9)**2)
        iris = (base*.68 + glow*.18)
        out[b:c] = eye
        out[b:c][radial < .052] = iris
        # A tiny highlight makes the gaze more readable and less blank.
        highlight = (dx < -.018) & (dy > .022) & (radial < .08)
        out[b:c][highlight] = eye

    out[c:d] = base*.62 + glow*.27
    out[d:n] = base*.38 + glow*.62

    if mood == 'happy':
        out[d:n] = np.minimum(255.0, out[d:n]*1.13)
    return np.clip(out, 0, 255).astype(np.uint8)
