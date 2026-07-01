#!/usr/bin/env python3
"""font_atlas.py — Font atlas для GPU-рендера символов.

Атлас: квадратный, 128×128, 16×16 ячеек по 8×8 px.
Это обход бага glReadPixels на текстурах > 64×64.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from char_cube import SYMBOL_SETS

CELL: int = 8
GRID: int = 16  # 16×16 = 256 слотов
ATLAS_W: int = CELL * GRID  # 128
ATLAS_H: int = CELL * GRID  # 128


def _find_font() -> Optional[str]:
    import os as _os
    windir = _os.environ.get('WINDIR', 'C:\\Windows')
    fonts = [
        _os.path.join(windir, 'Fonts', 'segoeui.ttf'),
        _os.path.join(windir, 'Fonts', 'arial.ttf'),
        _os.path.join(windir, 'Fonts', 'consola.ttf'),
        _os.path.join(windir, 'Fonts', 'seguiemj.ttf'),
    ]
    for fp in fonts:
        if _os.path.isfile(fp):
            return fp
    return None


def build_atlas(font_size: int = 7) -> Tuple[bytes, Dict[str, NDArray[np.int32]]]:
    font_path = _find_font()
    try:
        font = ImageFont.truetype(font_path, font_size) if font_path else None
    except Exception:
        font = None

    all_symbols: List[str] = []
    symbol_to_idx: Dict[str, int] = {}

    for name, syms in SYMBOL_SETS.items():
        for ch in syms:
            if ch not in symbol_to_idx and len(all_symbols) < GRID * GRID:
                symbol_to_idx[ch] = len(all_symbols)
                all_symbols.append(ch)

    n_symbols = len(all_symbols)

    img = Image.new('RGBA', (ATLAS_W, ATLAS_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    for idx, ch in enumerate(all_symbols):
        gx = idx % GRID
        gy = idx // GRID
        x = gx * CELL
        y = gy * CELL
        try:
            bbox = draw.textbbox((0, 0), ch, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            ox = (CELL - tw) // 2 - bbox[0]
            oy = (CELL - th) // 2 - bbox[1]
        except Exception:
            ox, oy = 1, 1
        draw.text((x + ox, y + oy), ch, font=font, fill=(255, 255, 255, 255))

    rgba_bytes = img.tobytes()

    symbol_maps: Dict[str, NDArray[np.int32]] = {}
    for name, syms in SYMBOL_SETS.items():
        indices = [symbol_to_idx.get(ch, 0) for ch in syms]
        symbol_maps[name] = np.array(indices, dtype=np.int32)

    print(f"[FontAtlas] {n_symbols} glyphs in {ATLAS_W}×{ATLAS_H}", flush=True)
    return rgba_bytes, symbol_maps


def get_atlas_dims() -> Tuple[int, int]:
    return ATLAS_W, ATLAS_H
