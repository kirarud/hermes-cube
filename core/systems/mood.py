"""systems/mood.py — Анализ настроения AI-ответа.

Читает:
  meta.ai_response — сырой ответ AI

Пишет:
  meta.mood — 'idle' | 'thinking' | 'speaking' | 'happy' | 'sad'
  meta.color_shift — 0.0-1.0 HSV shift
  meta.config — pulse_rate, pulse_amplitude, rotation_speed

Чистая функция: не вызывает AI, не работает с сетью.

Memo: не сбрасывает mood при пустом ответе — хранит последнее.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from core.world import World
from core.ai_constants import AI_MOODS


class MoodSystem:
    """Определяет настроение по ответу AI.

    Если ответ пустой — не меняет текущее настроение (stateful).
    """
    
    def __init__(self) -> None:
        self._last_mood: str = 'idle'
        self._last_color_shift: float = 0.0

    def update(self, world: World, dt: float) -> None:
        response = world.meta.ai_response
        if not response:
            # Не сбрасываем на idle — оставляем последнее настроение
            world.meta.mood = self._last_mood
            world.meta.color_shift = self._last_color_shift
            return

        mood, color_shift, display_text = self._analyze(response)

        # Заменяем сырой JSON на чистый текст для TextOverlay
        if display_text:
            world.meta.ai_response = display_text
        else:
            world.meta.ai_response = ''

        self._last_mood = mood
        self._last_color_shift = color_shift

        world.meta.mood = mood
        world.meta.color_shift = color_shift

        # Применить параметры настроения в конфиг
        mood_params = AI_MOODS.get(mood)
        if mood_params:
            cfg = world.meta.config
            for key in ('pulse_rate', 'pulse_amplitude', 'rotation_speed'):
                if key in mood_params:
                    cfg[key] = mood_params[key]

    @staticmethod
    def _analyze(text: str) -> tuple[str, float, str]:
        """Проанализировать текст и вернуть (mood, color_shift, display_text)."""
        cleaned = text.strip()
        if cleaned.startswith('```'):
            cleaned = cleaned.strip('` \n')
            if cleaned.startswith('json'):
                cleaned = cleaned[4:].strip()

        try:
            parsed = json.loads(cleaned)
            mood = str(parsed.get('mood', 'speaking'))
            if mood not in ('happy', 'sad', 'thinking', 'speaking', 'idle'):
                mood = MoodSystem._keyword_mood(text)
            color_hue = max(0.0, min(1.0, float(parsed.get('color_hue', 0.0))))
            display_text = str(parsed.get('text', ''))
            return mood, color_hue, display_text
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        mood = MoodSystem._keyword_mood(text)
        shift = AI_MOODS.get(mood, AI_MOODS['idle'])['color_shift']
        return mood, shift, text

    @staticmethod
    def _keyword_mood(text: str) -> str:
        t = text.lower()
        if any(w in t for w in ['груст', 'печал', 'устал', 'тоск', 'один']):
            return 'sad'
        if any(w in t for w in ['рад', 'счаст', 'весел', 'крут', 'класс', 'любл']):
            return 'happy'
        if any(w in t for w in ['дума', 'размыш', 'представ', 'может']):
            return 'thinking'
        return 'speaking'
