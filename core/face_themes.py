"""Safe, declarative Hermes Face theme presets (no executable mod code)."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from core.face_avatar import regions

THEMES = {
    'Soft Glow': ((96, 92, 136), (220, 183, 178), (255, 244, 228)),
    'Holographic': ((30, 106, 166), (103, 230, 251), (220, 255, 255)),
    'Bioform': ((76, 29, 14), (246, 151, 28), (255, 235, 141)),
    'Retro Hacker': ((24, 80, 42), (78, 232, 113), (207, 255, 195)),
    'Glass': ((93, 77, 157), (205, 190, 250), (252, 250, 255)),
}


def face_colors(shape: NDArray[np.float64], theme: str, mood: str = 'idle') -> NDArray[np.uint8]:
    """Map particle position and anatomy to a theme with bright readable eyes."""
    n = len(shape)
    if not n:
        return np.zeros((0, 3), dtype=np.uint8)
    base, glow, eye = THEMES.get(theme, THEMES['Holographic'])
    base = np.asarray(base, dtype=np.float64)
    glow = np.asarray(glow, dtype=np.float64)
    eye = np.asarray(eye, dtype=np.float64)
    y = np.clip((shape[:, 1]+1.2)/2.6, 0, 1)
    z = np.clip((shape[:, 2]+.56)/1.2, 0, 1)
    factor = np.clip(.24+.30*y+.40*z, 0, 1)
    result = base[None,:]+factor[:,None]*(glow-base)[None,:]
    a,b,c,d,_ = regions(n)
    result[a:b] *= .88
    result[b:c] = eye
    result[c:d] = glow * .90 + eye * .10
    result[d:] = eye * .70 + glow * .30
    if mood in ('happy','speaking','thinking'):
        result = result*.88 + eye[None,:]*.12
    return np.clip(result,0,255).astype(np.uint8)
