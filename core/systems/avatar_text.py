"""systems/avatar_text.py — Аватар: текст из частиц куба.

Управляет жизненным циклом текстового сообщения,
которое выкладывается частицами куба.

Состояния:
  idle → morph_in → display → morph_out → idle

Читает:
  meta.ai_response — новый ответ AI
  meta.config — shape_preset, morph_progress, cube_scale, char_mode

Пишет:
  sim.shape_cache['text'] — позиции букв
  sim.symbol_idx — per-particle индексы символов в font atlas
  meta.config — shape_preset='text', morph_progress, cube_scale, char_mode
  meta.text_mode — флаг активности
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import numpy as np
from numpy.typing import NDArray

from core.world import World
from core.text_layout import layout_text, get_text_scale_override

# Длительность фаз (секунды)
MORPH_IN_TIME: float = 0.8
HOLD_TIME: float = 3.5
MORPH_OUT_TIME: float = 0.8

# Минимальный прогресс чтобы считать morph завершённым
MORPH_THRESHOLD: float = 0.98


class AvatarTextSystem:
    """Управляет отображением AI ответа как текста из частиц."""

    def __init__(self) -> None:
        # State machine
        self.state: str = 'idle'
        self._timer: float = 0.0
        self._last_response: str = ''
        self._original_shape: str = 'cube'
        self._original_scale: float = 0.27
        self._original_char_mode: str = 'dots'
        self._text_scale: float = get_text_scale_override()
        self._float_time: float = 0.0  # для float-анимации

    def update(self, world: World, dt: float) -> None:
        # Проверяем новый ответ
        response = world.meta.ai_response
        if response and response != self._last_response and not world.meta.ai_thinking:
            self._last_response = response
            self._start_text_display(world)
            world.meta.ai_response = ''  # очищаем после обработки

        # Tick state machine
        if self.state == 'morph_in':
            self._tick_morph_in(world, dt)
        elif self.state == 'display':
            self._tick_display(world, dt)
        elif self.state == 'morph_out':
            self._tick_morph_out(world, dt)

    def _start_text_display(self, world: World) -> None:
        """Начать показ текста: сгенерировать раскладку, настроить morph."""
        n = world.sim.active_count
        if n == 0:
            return

        # Берём текст из ai_response
        # Парсим JSON если там он
        text = self._extract_text(world.meta.ai_response)
        if not text:
            return

        print(f"[AvatarText] → \"{text[:60]}...\" ({len(text)} chars)", flush=True)

        # Сохраняем оригинальные настройки для восстановления
        cfg = world.meta.config
        self._original_shape = cfg.get('shape_preset', 'cube')
        self._original_scale = float(cfg.get('cube_scale', 0.27))
        self._original_char_mode = cfg.get('char_mode', 'dots')
        self._original_rotation_speed = float(cfg.get('rotation_speed', 0.28))

        # Генерируем раскладку текста
        positions, char_indices, n_used = layout_text(text, n)

        # Сохраняем оригинальные цвета для текстовых частиц
        self._original_colors = world.sim.color[:n_used].copy()

        # Делаем текст ярким (белый/голубой оттенок)
        world.sim.color[:n_used] = [220, 230, 255]  # светло-голубой

        # Сохраняем в shape_cache — morph будет lerp от cube к text
        world.sim.shape_cache['text'] = positions

        # Сохраняем per-particle индексы символов для рендера
        world.sim.symbol_idx[:] = char_indices

        # Переключаем режимы
        cfg['shape_preset'] = 'text'
        cfg['morph_progress'] = 0.0
        cfg['cube_scale'] = self._text_scale
        cfg['char_mode'] = 'symbols'
        cfg['rotation_speed'] = 0.0  # отключаем вращение для читаемости
        world.meta.text_mode = True

        # Отключаем трейлы на время текста
        if world.render.trail_enabled:
            world.render.trail_enabled = False
            self._trails_was_enabled = True
        else:
            self._trails_was_enabled = False

        # Запускаем morph-in
        self.state = 'morph_in'
        self._timer = 0.0
        self._float_time = 0.0

    def _tick_morph_in(self, world: World, dt: float) -> None:
        """Morph куб → текст."""
        self._timer += dt
        progress = min(1.0, self._timer / MORPH_IN_TIME)
        world.meta.config['morph_progress'] = progress

        if progress >= MORPH_THRESHOLD:
            world.meta.config['morph_progress'] = 1.0
            self.state = 'display'
            self._timer = 0.0

    def _tick_display(self, world: World, dt: float) -> None:
        """Текст виден — лёгкая float-анимация."""
        self._timer += dt
        self._float_time += dt

        # По таймеру — начинаем morph-out
        if self._timer >= HOLD_TIME:
            self.state = 'morph_out'
            self._timer = 0.0

    def apply_float(self, world: World) -> None:
        """Применить float-анимацию к частицам (вызывается внутри рендер-кадра)."""
        if self.state != 'display':
            return
        n = world.sim.active_count
        if n == 0:
            return
        text_pos = world.sim.shape_cache.get('text')
        if text_pos is None:
            return
        n_used = min(n, len(text_pos))

        # Y-покачивание
        wave = np.sin(self._float_time * 1.5 + np.arange(n_used) * 0.7) * 0.015
        world.sim.animated[:n_used] = world.sim.morphed[:n_used]
        world.sim.animated[:n_used, 1] += wave

        # Остальные частицы — без анимации
        if n_used < n:
            world.sim.animated[n_used:] = world.sim.morphed[n_used:]

    def _tick_morph_out(self, world: World, dt: float) -> None:
        """Morph текст → куб."""
        self._timer += dt
        # Идём от 1.0 к 0.0
        progress = max(0.0, 1.0 - self._timer / MORPH_OUT_TIME)
        world.meta.config['morph_progress'] = progress

        # Возвращаем cube_scale обратно
        cfg = world.meta.config
        cur_scale = cfg.get('cube_scale', self._original_scale)
        # Плавно возвращаем scale
        mix = 1.0 - progress  # 0 → 1
        cfg['cube_scale'] = self._original_scale + (self._text_scale - self._original_scale) * (1.0 - mix)

        if progress <= (1.0 - MORPH_THRESHOLD):
            self._finish(world)

    def _finish(self, world: World) -> None:
        """Вернуть куб в исходное состояние."""
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
        if hasattr(self, '_original_colors'):
            n_restore = min(len(self._original_colors), world.sim.active_count)
            world.sim.color[:n_restore] = self._original_colors[:n_restore]

        # Восстанавливаем трейлы
        if getattr(self, '_trails_was_enabled', False):
            world.render.trail_enabled = True

        self.state = 'idle'
        self._last_response = ''
        print("[AvatarText] ← restored", flush=True)

    @staticmethod
    def _extract_text(raw: str) -> str:
        """Извлечь чистый текст из AI ответа (поддерживает JSON)."""
        if not raw:
            return ''

        cleaned = raw.strip()
        if cleaned.startswith('```'):
            cleaned = cleaned.strip('` \n')
            if cleaned.startswith('json'):
                cleaned = cleaned[4:].strip()

        # Пробуем JSON
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                text = parsed.get('text', '') or str(parsed.get('display_text', ''))
                if text:
                    return text
        except (json.JSONDecodeError, ValueError):
            pass

        # Не JSON — возвращаем как есть
        return cleaned
