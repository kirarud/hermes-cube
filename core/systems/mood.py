"""systems/mood.py — Анализ настроения AI-ответа.

Читает:
  meta.ai_response — сырой ответ AI

Пишет:
  meta.mood — 'idle' | 'thinking' | 'speaking' | 'happy' | 'sad'
  meta.color_shift — 0.0-1.0 HSV shift

Чистая функция: не вызывает AI, не работает с сетью.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from core.world import World
from core.ai_constants import AI_MOODS


class MoodSystem:
    """Определяет настроение по ответу AI.

    Если ответ пустой — idle.
    Если ответ содержит JSON — парсит mood/color_hue.
    Иначе — keyword-based fallback.
    """

    def update(self, world: World, dt: float) -> None:
        response = world.meta.ai_response
        if not response:
            world.meta.mood = 'idle'
            world.meta.color_shift = 0.0
            return

        mood, color_shift = self._analyze(response)
        world.meta.mood = mood
        world.meta.color_shift = color_shift

        # Применить параметры настроения в конфиг (проекция/анимация читают meta.config)
        mood_params = AI_MOODS.get(mood)
        if mood_params:
            # Через meta.config — projection и color смотрят сюда
            cfg = world.meta.config
            for key in ('pulse_rate', 'pulse_amplitude', 'rotation_speed'):
                if key in mood_params:
                    cfg[key] = mood_params[key]

    @staticmethod
    def _analyze(text: str) -> tuple[str, float]:
        """Проанализировать текст и вернуть (mood, color_shift)."""
        # Попробовать JSON
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
            return mood, color_hue
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # Keyword fallback
        mood = MoodSystem._keyword_mood(text)
        shift = AI_MOODS.get(mood, AI_MOODS['idle'])['color_shift']
        return mood, shift

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
