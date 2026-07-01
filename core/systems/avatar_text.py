"""systems/avatar_text.py — Аватар: текст из частиц куба.

Самостоятельно управляет morph-переходом, не полагаясь на pipeline.
Использует текущую world_position (повёрнутую) как стартовую точку.

Состояния: idle → morph_in → display → morph_out → idle
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
        # Стартовая позиция для lerp (сохраняется при начале morph_in)
        self._start_pos: NDArray[np.float64] | None = None
        # Целевая позиция текста
        self._target_pos: NDArray[np.float64] | None = None

    def update(self, world: World, dt: float) -> None:
        response = world.meta.ai_response
        if response and response != self._last_response and not world.meta.ai_thinking:
            self._last_response = response
            self._start_text_display(world)
            world.meta.ai_response = ''

        if self.state == 'morph_in':
            self._tick_morph_in(world, dt)
        elif self.state == 'display':
            self._tick_display(world, dt)
        elif self.state == 'morph_out':
            self._tick_morph_out(world, dt)

    def apply_animated(self, world: World) -> None:
        """Записать финальные позиции частиц в animated.
        Вызывается из main.py ПОСЛЕ pipeline (после rotation).
        """
        if self.state == 'idle':
            return

        n = world.sim.active_count
        if n == 0 or self._target_pos is None or self._start_pos is None:
            return

        n_used = min(n, len(self._target_pos))
        progress = world.meta.config.get('morph_progress', 0.0)

        if self.state == 'morph_in':
            # Lerp от стартовой (повёрнутой) к тексту
            t = progress
            for i in range(n_used):
                world.sim.animated[i] = self._start_pos[i] * (1.0 - t) + self._target_pos[i] * t
            if n_used < n:
                world.sim.animated[n_used:] = self._start_pos[n_used:]
            # Масштабируем постепенно
            self._apply_scale(world, t)

        elif self.state == 'display':
            # Текст — финальная позиция + float
            world.sim.animated[:n_used] = self._target_pos[:n_used]
            # Float: Y-покачивание
            wave = np.sin(self._float_time * 1.5 + np.arange(n_used) * 0.7) * 0.015
            world.sim.animated[:n_used, 1] += wave
            if n_used < n:
                world.sim.animated[n_used:] = self._start_pos[n_used:]
            self._apply_scale(world, 1.0)

        elif self.state == 'morph_out':
            # Lerp от текста к стартовой
            t = 1.0 - progress  # 1 → 0
            for i in range(n_used):
                world.sim.animated[i] = self._start_pos[i] * t + self._target_pos[i] * (1.0 - t)
            if n_used < n:
                world.sim.animated[n_used:] = self._start_pos[n_used:]
            self._apply_scale(world, t)

    def _apply_scale(self, world: World, t: float) -> None:
        """Плавно переключаем cube_scale между оригинальным и текстовым."""
        cfg = world.meta.config
        cfg['cube_scale'] = self._original_scale * (1.0 - t) + self._text_scale * t

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

        # Сохраняем текущую повёрнутую позицию как стартовую для lerp
        n_sim = min(n, len(world.sim.world_position))
        self._start_pos = world.sim.world_position[:n_sim].copy()

        # Генерируем текст
        positions, char_indices, n_used = layout_text(text, n)
        self._target_pos = positions.copy()

        # Цвета
        self._original_colors = world.sim.color[:n_sim].copy()
        world.sim.color[:n_used] = [220, 230, 255]

        # Индексы символов
        world.sim.symbol_idx[:] = char_indices

        # Настройки
        cfg['morph_progress'] = 0.0
        cfg['cube_scale'] = self._original_scale
        cfg['char_mode'] = 'symbols'
        cfg['rotation_speed'] = 0.0
        cfg['shape_preset'] = 'text'
        world.meta.text_mode = True

        # Трейлы off
        if world.render.trail_enabled:
            world.render.trail_enabled = False
            self._trails_was_enabled = True
        else:
            self._trails_was_enabled = False

        self.state = 'morph_in'
        self._timer = 0.0
        self._float_time = 0.0

    def _tick_morph_in(self, world: World, dt: float) -> None:
        self._timer += dt
        t = min(1.0, self._timer / MORPH_IN_TIME)
        world.meta.config['morph_progress'] = t
        if t >= 1.0:
            world.meta.config['morph_progress'] = 1.0
            self.state = 'display'
            self._timer = 0.0

    def _tick_display(self, world: World, dt: float) -> None:
        self._timer += dt
        self._float_time += dt
        if self._timer >= HOLD_TIME:
            self.state = 'morph_out'
            self._timer = 0.0

    def _tick_morph_out(self, world: World, dt: float) -> None:
        self._timer += dt
        t = max(0.0, 1.0 - self._timer / MORPH_OUT_TIME)
        world.meta.config['morph_progress'] = t
        if t <= 0.02:
            self._finish(world)

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

        if hasattr(self, '_original_colors'):
            n_restore = min(len(self._original_colors), world.sim.active_count)
            world.sim.color[:n_restore] = self._original_colors[:n_restore]
        if getattr(self, '_trails_was_enabled', False):
            world.render.trail_enabled = True

        self.state = 'idle'
        self._last_response = ''
        self._start_pos = None
        self._target_pos = None
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
