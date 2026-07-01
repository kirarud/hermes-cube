"""systems/avatar_text.py — Аватар: текст из частиц куба.

Прямой контроль world_position. 
lerp от base_position (не world_position) и обратно — чтобы избежать рывка.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import numpy as np
from numpy.typing import NDArray

from core.world import World
from core.text_layout import layout_text, get_text_scale_override

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
        self._original_char_mode: str = 'dots'
        self._original_rotation_speed: float = 0.28
        self._text_scale: float = 0.42  # крупнее куба, чтобы текст был читаем
        self._float_time: float = 0.0
        self._text_pos: NDArray[np.float64] | None = None
        self._start_colors: NDArray[np.float64] | None = None
        self._start_base: NDArray[np.float64] | None = None  # base_position
        self._n_used: int = 0

    def update(self, world: World, dt: float) -> None:
        response = world.meta.ai_response
        if response and response != self._last_response and not world.meta.ai_thinking:
            self._last_response = response
            self._start_text_display(world)
            world.meta.ai_response = ''

        if self.state == 'idle':
            return

        if self.state == 'morph_in':
            self._timer += dt
            t = min(1.0, self._timer / MORPH_IN_TIME)
            self._set_morph(world, t)
            if t >= 1.0:
                self.state = 'display'
                self._timer = 0.0

        elif self.state == 'display':
            self._timer += dt
            self._float_time += dt
            self._set_morph(world, 1.0)
            self._apply_float(world)
            if self._timer >= HOLD_TIME:
                self.state = 'morph_out'
                self._timer = 0.0

        elif self.state == 'morph_out':
            self._timer += dt
            t = max(0.0, 1.0 - self._timer / MORPH_OUT_TIME)
            self._set_morph(world, t)
            if t <= 0.02:
                self._finish(world)

    def _set_morph(self, world: World, t: float) -> None:
        """Записать lerp base → text в world_position."""
        n = world.sim.active_count
        if n == 0 or self._text_pos is None or self._start_base is None:
            return

        n_used = min(n, len(self._text_pos), len(self._start_base))
        if n_used == 0:
            return

        # Векторизованный lerp — без Python-цикла
        world.sim.world_position[:n_used] = (
            self._start_base[:n_used] * (1.0 - t) + self._text_pos[:n_used] * t
        )
        if n_used < n:
            world.sim.world_position[n_used:] = self._start_base[n_used:]

        # cube_scale
        cfg = world.meta.config
        cfg['cube_scale'] = self._original_scale * (1.0 - t) + self._text_scale * t

        # Цвета — векторизованно
        if self._start_colors is not None:
            s = 1.0 - t
            target = np.array([220.0, 230.0, 255.0], dtype=np.float64)
            world.sim.color[:n_used] = self._start_colors[:n_used] * s + target * t

    def _apply_float(self, world: World) -> None:
        """Лёгкое покачивание букв."""
        n = world.sim.active_count
        if n == 0 or self._text_pos is None:
            return
        n_used = min(n, len(self._text_pos))
        wave = np.sin(self._float_time * 1.5 + np.arange(n_used) * 0.7) * 0.015
        world.sim.world_position[:n_used, 1] += wave

    def _start_text_display(self, world: World) -> None:
        n = world.sim.active_count
        if n == 0:
            return

        text = self._extract_text(world.meta.ai_response)
        if not text:
            return

        print(f"[AvatarText] → \"{text[:60]}...\" ({len(text)} chars)", flush=True)

        cfg = world.meta.config
        self._original_shape = cfg.get('shape_preset', 'cube')
        self._original_scale = float(cfg.get('cube_scale', 0.27))
        self._original_char_mode = cfg.get('char_mode', 'dots')
        self._original_rotation_speed = float(cfg.get('rotation_speed', 0.28))

        # Старт = base_position (неповёрнутая) — чтобы lerp шёл от неё
        self._start_base = world.sim.base_position[:n].copy()

        # Генерация раскладки текста (не растягиваем на весь экран)
        positions, char_indices, n_used = layout_text(text, n)
        self._text_pos = positions.copy()
        self._n_used = n_used

        # Цвета
        self._start_colors = world.sim.color[:n].copy()
        world.sim.color[:n_used] = [220, 230, 255]  # светло-голубой

        # Индексы символов
        world.sim.symbol_idx[:] = char_indices

        # Отключаем pipeline морф/вращение
        cfg['morph_progress'] = 0.0
        cfg['cube_scale'] = self._original_scale
        cfg['char_mode'] = 'symbols'
        cfg['rotation_speed'] = 0.0
        cfg['shape_preset'] = 'text'
        world.meta.text_mode = True

        if world.render.trail_enabled:
            world.render.trail_enabled = False
            self._trails_was_enabled = True
        else:
            self._trails_was_enabled = False

        self.state = 'morph_in'
        self._timer = 0.0
        self._float_time = 0.0

    def _finish(self, world: World) -> None:
        cfg = world.meta.config
        cfg['shape_preset'] = self._original_shape
        cfg['morph_progress'] = 0.0
        cfg['cube_scale'] = self._original_scale
        cfg['char_mode'] = self._original_char_mode
        cfg['rotation_speed'] = self._original_rotation_speed
        world.meta.text_mode = False
        world.meta.mood = 'idle'
        world.meta.color_shift = 0.0

        # Восстанавливаем цвета
        if self._start_colors is not None:
            n = min(len(self._start_colors), world.sim.active_count)
            world.sim.color[:n] = self._start_colors[:n]

        if getattr(self, '_trails_was_enabled', False):
            world.render.trail_enabled = True

        self.state = 'idle'
        self._last_response = ''
        self._text_pos = None
        self._start_base = None
        self._start_colors = None
        print("[AvatarText] ← restored", flush=True)

    @staticmethod
    def _extract_text(raw: str) -> str:
        if not raw:
            return ''
        cleaned = raw.strip()
        if cleaned.startswith('```'):
            cleaned = cleaned.strip('` \n')
            if cleaned.startswith('json'):
                cleaned = cleaned[4:].strip()
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                text = parsed.get('text', '') or str(parsed.get('display_text', ''))
                if text:
                    return text
        except (json.JSONDecodeError, ValueError):
            pass
        return cleaned
