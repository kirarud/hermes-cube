"""systems/mood.py — Анализ настроения AI-ответа.

Читает:
  meta.ai_response — сырой ответ AI (JSON или текст)

Пишет:
  meta.mood, meta.color_shift
  meta.config pulse_rate / pulse_amplitude / rotation_speed
  meta.ai_response — чистый display text (для TextOverlay)
  meta.speak_buffer — тот же text (для AvatarText), если response_mode позволяет

Обрабатывает каждый ответ один раз (_last_raw).
"""

from __future__ import annotations

import json
from typing import Tuple

from core.world import World
from core.ai_constants import AI_MOODS, DEFAULT_RESPONSE_MODE


class MoodSystem:
    """Определяет настроение по ответу AI и замыкает UX-цикл."""

    def __init__(self) -> None:
        self._last_mood: str = 'idle'
        self._last_color_shift: float = 0.0
        self._last_raw: str = ''

    def update(self, world: World, dt: float) -> None:
        response = world.meta.ai_response
        if not response:
            world.meta.mood = self._last_mood
            world.meta.color_shift = self._last_color_shift
            return

        # Уже обработали этот raw-ответ
        if response == self._last_raw:
            return
        # Ещё думает — ждём финальный ответ (AI кладёт response, потом thinking=False;
        # если thinking=True и response пустой — выше return; если оба — можно разбирать)
        if world.meta.ai_thinking and response.startswith('{') is False and not response.strip():
            return

        mood, color_shift, display_text = self._analyze(response)

        self._last_mood = mood
        self._last_color_shift = color_shift
        world.meta.mood = mood
        world.meta.color_shift = color_shift

        # Применить параметры настроения (ключи = config)
        mood_params = AI_MOODS.get(mood, AI_MOODS['idle'])
        cfg = world.meta.config
        for key in ('pulse_rate', 'pulse_amplitude', 'rotation_speed'):
            if key in mood_params:
                cfg[key] = mood_params[key]

        mode = str(cfg.get('response_mode', DEFAULT_RESPONSE_MODE)).lower()
        text = (display_text or '').strip()
        # Помечаем и raw, и display — чтобы не разбирать один ответ дважды
        self._last_raw = text or response

        if not text:
            world.meta.ai_response = ''
            return

        # letters / both → TextOverlay читает ai_response
        # avatar / both → AvatarText читает speak_buffer
        if mode in ('letters', 'both', 'overlay'):
            world.meta.ai_response = text
        else:
            world.meta.ai_response = ''

        if mode in ('avatar', 'both', 'cube'):
            world.meta.speak_buffer = text

    @staticmethod
    def _analyze(text: str) -> Tuple[str, float, str]:
        cleaned = text.strip()
        if cleaned.startswith('```'):
            cleaned = cleaned.strip('` \n')
            if cleaned.startswith('json'):
                cleaned = cleaned[4:].strip()

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                mood = str(parsed.get('mood', 'speaking'))
                if mood not in ('happy', 'sad', 'thinking', 'speaking', 'idle'):
                    mood = MoodSystem._keyword_mood(text)
                color_hue = max(0.0, min(1.0, float(parsed.get('color_hue', 0.0))))
                display_text = str(parsed.get('text', '') or '')
                if not display_text:
                    display_text = str(parsed.get('display_text', '') or '')
                return mood, color_hue, display_text
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        mood = MoodSystem._keyword_mood(text)
        shift = float(AI_MOODS.get(mood, AI_MOODS['idle']).get('color_shift', 0.0))
        return mood, shift, text

    @staticmethod
    def _keyword_mood(text: str) -> str:
        t = text.lower()
        if any(w in t for w in ('груст', 'печал', 'устал', 'тоск', 'один', 'error', 'ошиб')):
            return 'sad'
        if any(w in t for w in ('рад', 'счаст', 'весел', 'крут', 'класс', 'любл', 'привет')):
            return 'happy'
        if any(w in t for w in ('дума', 'размыш', 'представ', 'может', 'offline', 'студия')):
            return 'thinking'
        return 'speaking'
