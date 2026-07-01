"""systems/avatar_text.py — Аватар: текст из частиц куба.

Полный контроль над позициями частиц во время текстового режима.
Записывает напрямую в world_position (вместо animated),
отключает rotation и animation в конфиге.

Никакого параллельного pipeline morph — lerp руками.
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
        self._text_scale: float = get_text_scale_override()
        self._float_time: float = 0.0
        self._start_pos: NDArray[np.float64] | None = None
        self._target_pos: NDArray[np.float64] | None = None
        self._start_colors: NDArray[np.float64] | None = None

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
            # Float: Y-покачивание
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
        """Записать lerp-позицию в world_position (без pipeline morph)."""
        n = world.sim.active_count
        if n == 0 or self._target_pos is None or self._start_pos is None:
            return

        n_used = min(n, len(self._target_pos), len(self._start_pos))
        if n_used == 0:
            return

        # Lerp в world_position (→ projection прочитает)
        for i in range(n_used):
            world.sim.world_position[i] = self._start_pos[i] * (1.0 - t) + self._target_pos[i] * t
        # Остальные частицы (не текст) — держим на стартовой
        if n_used < n:
            world.sim.world_position[n_used:] = self._start_pos[n_used:]

        # Плавный scale
        cfg = world.meta.config
        cfg['cube_scale'] = self._original_scale * (1.0 - t) + self._text_scale * t

        # Цвета: плавно переключаем между оригинальным и текстовым
        if self._start_colors is not None:
            for i in range(n_used):
                world.sim.color[i] = self._start_colors[i] * (1.0 - t) + np.array([220, 230, 255]) * t

    def _apply_float(self, world: World) -> None:
        """Лёгкое вертикальное покачивание букв."""
        n = world.sim.active_count
        if n == 0 or self._target_pos is None:
            return
        wave = np.sin(self._float_time * 1.5 + np.arange(min(n, len(self._target_pos))) * 0.7) * 0.02
        for i in range(len(wave)):
            world.sim.world_position[i, 1] += wave[i]

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

        # Стартовая позиция = текущая повёрнутая
        self._start_pos = world.sim.world_position[:n].copy()

        # Генерация раскладки
        positions, char_indices, n_used = layout_text(text, n)
        self._target_pos = positions.copy()

        # Сохраняем цвета
        self._start_colors = world.sim.color[:n].copy()

        # Индексы символов
        world.sim.symbol_idx[:] = char_indices

        # Отключаем pipeline системы
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

        # Цвета
        if self._start_colors is not None:
            n = min(len(self._start_colors), world.sim.active_count)
            world.sim.color[:n] = self._start_colors[:n]

        if getattr(self, '_trails_was_enabled', False):
            world.render.trail_enabled = True

        self.state = 'idle'
        self._last_response = ''
        self._start_pos = None
        self._target_pos = None
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
