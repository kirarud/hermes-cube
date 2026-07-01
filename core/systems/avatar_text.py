"""systems/avatar_text.py — Минимальный аватар: текст из частиц.

Ничего не трогает в основном цикле куба.
Только пишет в shape_cache и управляет morph_progress.
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np

from core.world import World
from core.text_layout import layout_text_mask

MORPH_IN_TIME: float = 0.8
HOLD_TIME: float = 3.5
MORPH_OUT_TIME: float = 0.8


class AvatarTextSystem:

    def __init__(self) -> None:
        self.state: str = 'idle'
        self._timer: float = 0.0
        self._last_response: str = ''

    def update(self, world: World, dt: float) -> None:
        buf = world.meta.speak_buffer
        if buf and buf != self._last_response:
            self._last_response = buf
            world.meta.speak_buffer = ''
            self._start_text(world, buf)
            return

        if self.state == 'idle':
            return

        cfg = world.meta.config
        if self.state == 'morph_in':
            self._timer += dt
            t = min(1.0, self._timer / MORPH_IN_TIME)
            cfg['morph_progress'] = t
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
            cfg['morph_progress'] = t
            if t <= 0.02:
                self._finish(world)

    def _start_text(self, world: World, text: str) -> None:
        n = world.sim.active_count
        if n == 0 or not text:
            return
        cfg = world.meta.config
        positions, n_used, _ = layout_text_mask(text, n, font_size=42)
        world.sim.shape_cache['text'] = positions.copy()
        cfg['morph_progress'] = 0.0
        cfg['shape_preset'] = 'text'
        cfg['rotation_speed'] = 0.0
        world.meta.text_mode = True
        self.state = 'morph_in'
        self._timer = 0.0

    def _finish(self, world: World) -> None:
        cfg = world.meta.config
        cfg['morph_progress'] = 0.0
        cfg['shape_preset'] = 'cube'
        cfg['rotation_speed'] = 0.28
        world.meta.text_mode = False
        self.state = 'idle'
        self._last_response = ''

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
