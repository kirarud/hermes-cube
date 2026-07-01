"""systems/avatar_text.py — Аватар: текст через font atlas с per-particle символами.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

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
        self._original_shape: str = 'cube'
        self._original_scale: float = 0.27
        self._original_cell: int = 6
        self._original_speed: float = 0.28
        self._text_scale: float = 0.5
        self._text_cell: int = 16
        self._display_text: str = ''
        self._saved_positions: NDArray[np.float64] | None = None  # позиции куба на момент старта

    def update(self, world: World, dt: float) -> None:
        response = world.meta.ai_response
        if response and response != self._last_response and not world.meta.ai_thinking:
            self._last_response = response
            self._start_text_display(world)
            world.meta.ai_response = ''

        # Прямой ввод из speak_buffer (HTTP API)
        buf = world.meta.speak_buffer
        if buf and buf != self._last_response:
            self._last_response = buf
            world.meta.speak_buffer = ''
            text = self._extract_text(json.dumps({"text": buf}))
            if text:
                self._start_text_display(world)
                return
        if self.state == 'idle':
            return
        cfg = world.meta.config
        if self.state == 'morph_in':
            self._timer += dt
            t = min(1.0, self._timer / MORPH_IN_TIME)
            cfg['morph_progress'] = t
            cfg['cube_scale'] = self._original_scale * (1.0 - t) + self._text_scale * t
            cfg['cell_size'] = self._original_cell + int((self._text_cell - self._original_cell) * t)
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
            cfg['cube_scale'] = self._original_scale * t + self._text_scale * (1.0 - t)
            cfg['cell_size'] = self._text_cell + int((self._original_cell - self._text_cell) * (1.0 - t))
            cfg['rotation_speed'] = self._original_speed * (1.0 - t)
            if t <= 0.02:
                self._finish(world)

    def _start_text_display(self, world: World) -> None:
        n = world.sim.active_count
        if n == 0:
            return
        text = self._extract_text(world.meta.ai_response)
        if not text:
            return
        print(f"[AvatarText] → \"{text[:60]}...\" ({len(text)} chars)", flush=True)
        cfg = world.meta.config
        self._original_scale = float(cfg.get('cube_scale', 0.27))
        self._original_cell = int(cfg.get('cell_size', 6))
        self._original_speed = float(cfg.get('rotation_speed', 0.28))
        # Сохраняем текущие позиции куба (уже повёрнутые) для плавного возврата
        self._saved_positions = world.sim.world_position[:n].copy()
        positions, n_used, _ = layout_text_mask(text, n, font_size=48)
        world.sim.shape_cache['text'] = positions.copy()
        self._display_text = text
        cfg['char_mode'] = 'symbols'
        cfg['color_mode'] = 'z_layers'
        cfg['shape_preset'] = 'text'
        cfg['morph_progress'] = 0.0
        cfg['rotation_speed'] = 0.0
        cfg['cube_scale'] = self._original_scale
        cfg['cell_size'] = self._original_cell
        world.meta.text_mode = True
        self._map_text_to_symbols(world, text, n_used)
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
        cfg['color_mode'] = 'default'
        world.meta.text_mode = False
        if getattr(self, '_trails_was_enabled', False):
            world.render.trail_enabled = True

        self.state = 'idle'
        self._last_response = ''
        self._display_text = ''
        self._saved_positions = None
        print("[AvatarText] ← restored", flush=True)

    def _write_lerp_out(self, world, t=None):
        if self._saved_positions is None:
            return
        if t is None:
            t = max(0.0, 1.0 - self._timer / MORPH_OUT_TIME) if self._timer > 0 else 0.0
        n = world.sim.active_count
        text = world.sim.shape_cache.get('text')
        if text is None:
            return
        n_used = min(n, len(text), len(self._saved_positions))
        if n_used == 0:
            return
        text_pos = text[:n_used]
        orig = self._saved_positions[:n_used]
        world.sim.world_position[:n_used] = orig * t + text_pos * (1.0 - t)
        if n_used < n:
            world.sim.world_position[n_used:] = self._saved_positions[n_used:]

    def _map_text_to_symbols(self, world: World, text: str, n_used: int) -> None:
        """Назначить каждой частице индекс символа из font atlas."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from char_cube import SYMBOL_SETS
        char_map = {}
        for name, syms in SYMBOL_SETS.items():
            for ch in syms:
                if ch not in char_map and len(char_map) < 256:
                    char_map[ch] = len(char_map)
        extra = (
            list('abcdefghijklmnopqrstuvwxyz') +
            list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') +
            list('0123456789') +
            list(".,!?—…:;'\"()[]{}@#$%^&*+=<>/~`|\\- ") +
            list('абвгдеёжзийклмнопрстуфхцчшщъыьэюя')
        )
        for ch in extra:
            if ch not in char_map and len(char_map) < 256:
                char_map[ch] = len(char_map)
        text_chars = list(text)
        for i in range(min(n_used, len(text_chars))):
            world.sim.symbol_idx[i] = char_map.get(text_chars[i], 0)
        if n_used < world.sim.active_count:
            world.sim.symbol_idx[n_used:] = 0
        print(f"[AvatarText] mapped {min(n_used, len(text_chars))} symbols", flush=True)

    @staticmethod
    def _extract_text(raw: str) -> str:
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
