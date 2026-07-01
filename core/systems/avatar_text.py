"""systems/avatar_text.py — Минимальный текст из частиц.

При поступлении speak_buffer: morph в текст через pipeline,
показ 3.5с, morph обратно.
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np

from core.world import World
from core.text_layout import layout_text_mask

CHAR_MAP: dict[str, int] | None = None


def _get_char_map() -> dict[str, int]:
    global CHAR_MAP
    if CHAR_MAP is not None:
        return CHAR_MAP
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from char_cube import SYMBOL_SETS
    m: dict[str, int] = {}
    for name, syms in SYMBOL_SETS.items():
        for ch in syms:
            if ch not in m and len(m) < 256:
                m[ch] = len(m)
    extra = (
        list('abcdefghijklmnopqrstuvwxyz') +
        list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') +
        list('0123456789') +
        list(".,!?—…:;'\"()[]{}@#$%^&*+=<>/~`|\\- ") +
        list('абвгдеёжзийклмнопрстуфхцчшщъыьэюя')
    )
    for ch in extra:
        if ch not in m and len(m) < 256:
            m[ch] = len(m)
    CHAR_MAP = m
    return m

MORPH_IN_TIME: float = 0.8
HOLD_TIME: float = 3.5
MORPH_OUT_TIME: float = 0.8


class AvatarTextSystem:

    def __init__(self) -> None:
        self.state: str = 'idle'
        self._timer: float = 0.0
        self._last_response: str = ''
        self._original_shape: str = 'cube'
        self._original_speed: float = 0.28
        self._text_cell: int = 12
        self._text_scale: float = 0.5

    def update(self, world: World, dt: float) -> None:
        # speak_buffer = прямой ввод из HTTP API
        buf = world.meta.speak_buffer
        if buf and buf != self._last_response:
            self._last_response = buf
            world.meta.speak_buffer = ''
            self._start_text(world, buf)
            return

        # ai_response = ответ от LM Studio
        response = world.meta.ai_response
        if response and response != self._last_response and not world.meta.ai_thinking:
            self._last_response = response
            text = self._extract_text(response)
            if text:
                self._start_text(world, text)
            world.meta.ai_response = ''

        if self.state == 'idle':
            return

        cfg = world.meta.config
        if self.state == 'morph_in':
            self._timer += dt
            t = min(1.0, self._timer / MORPH_IN_TIME)
            cfg['morph_progress'] = t
            cfg['cube_scale'] = self._original_scale * (1.0 - t) + self._text_scale * t
            cfg['cell_size'] = 6 + int((self._text_cell - 6) * t)
            if t >= 1.0:
                self.state = 'display'
                self._timer = 0.0
        elif self.state == 'display':
            self._timer += dt
            if self._timer >= HOLD_TIME:
                self.state = 'morph_out'
                self._timer = 0.0
        elif self.state == 'morph_out':
            self._timer += dt
            t = max(0.0, 1.0 - self._timer / MORPH_OUT_TIME)
            # morph_progress 1→0 — чисто pipeline lerp text→base
            cfg['morph_progress'] = t
            cfg['cube_scale'] = self._original_scale * t + self._text_scale * (1.0 - t)
            cfg['cell_size'] = self._text_cell + int((self._original_cell - self._text_cell) * (1.0 - t))
            # rotation_speed плавно возвращается
            cfg['rotation_speed'] = self._original_speed * (1.0 - t)
            if t <= 0.02:
                self._finish(world)

    def _start_text(self, world: World, text: str) -> None:
        n = world.sim.active_count
        if n == 0 or not text:
            return
        print(f"[AvatarText] {text[:40]}", flush=True)
        cfg = world.meta.config
        self._original_shape = cfg.get('shape_preset', 'cube')
        self._original_scale = float(cfg.get('cube_scale', 0.27))
        self._original_speed = float(cfg.get('rotation_speed', 0.28))
        self._original_cell = int(cfg.get('cell_size', 6))

        positions, n_used, _ = layout_text_mask(text, n, font_size=48)
        world.sim.shape_cache['text'] = positions.copy()
        self._saved_positions = world.sim.world_position[:n].copy()

        # Per-particle symbol indices
        char_map = _get_char_map()
        for i in range(min(n_used, len(text))):
            world.sim.symbol_idx[i] = char_map.get(text[i], 0)
        if n_used < n:
            world.sim.symbol_idx[n_used:] = 0

        cfg['char_mode'] = 'symbols'
        cfg['shape_preset'] = 'text'
        cfg['morph_progress'] = 0.0
        cfg['rotation_speed'] = 0.0
        cfg['cube_scale'] = self._original_scale
        cfg['cell_size'] = 6
        world.meta.text_mode = True

        if world.render.trail_enabled:
            world.render.trail_enabled = False
            self._trails_was_enabled = True
        else:
            self._trails_was_enabled = False

        self.state = 'morph_in'
        self._timer = 0.0

    def _finish(self, world: World) -> None:
        cfg = world.meta.config
        cfg['shape_preset'] = self._original_shape
        cfg['morph_progress'] = 0.0
        cfg['cube_scale'] = self._original_scale
        cfg['cell_size'] = self._original_cell
        cfg['rotation_speed'] = self._original_speed
        cfg['char_mode'] = 'dots'
        world.meta.text_mode = False

        if getattr(self, '_trails_was_enabled', False):
            world.render.trail_enabled = True

        self.state = 'idle'
        self._last_response = ''
        self._saved_positions = None
        print("[AvatarText] done", flush=True)

    @staticmethod
    def _extract_text(raw):
        if not raw: return ''
        cleaned = raw.strip()
        if cleaned.startswith('```'):
            cleaned = cleaned.strip('` \n')
            if cleaned.startswith('json'): cleaned = cleaned[4:].strip()
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                text = parsed.get('text', '') or str(parsed.get('display_text', ''))
                if text: return text
        except: pass
        return cleaned
