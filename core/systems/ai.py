"""systems/ai.py — AISystem: мост между InputSystem и AI-ядром.

Читает:
  meta.input_buffer — текст от пользователя
  meta.config — настройки AI

Пишет:
  meta.ai_response — ответ AI (для TextOverlaySystem)
  meta.ai_thinking — True пока модель думает
  meta.ai_requested — True если нужно запустить LM Studio
  meta.mood — текущее настроение
  meta.color_shift — HSV-сдвиг от настроения

НЕ вызывает LM Studio напрямую — использует ai_module.ai_chat().
НЕ знает про TextOverlay, InputWindow, Tkinter.
"""

from __future__ import annotations

import json
import threading
from typing import Any, Dict, List, Optional

from core.world import World

# Импорт AI-ядра (пока существует ai_module.py)
from ai_module import ai_chat, analyze_mood, AI_MOODS


class AISystem:
    """Система AI-диалога.

    При появлении input_buffer запускает поток с ai_chat().
    Результат: mood + response + color_shift в world.meta.
    """

    def __init__(self) -> None:
        self._history: List[Dict[str, str]] = []

    def update(self, world: World, dt: float) -> None:
        """Проверить input_buffer, отправить в AI, записать ответ."""
        text = world.meta.input_buffer
        if not text:
            return

        # Сброс (чтобы не повторять)
        world.meta.input_buffer = ''

        # Добавить в историю
        self._history.append({"role": "user", "text": text})

        # Сигнал: AI нужен
        world.meta.ai_requested = True
        world.meta.ai_thinking = True

        # Запустить AI в потоке
        def _do_ai() -> None:
            response: str = ai_chat(text, history=self._history)
            world.meta.ai_response = response
            world.meta.mood = analyze_mood(response)
            mood_data = AI_MOODS.get(world.meta.mood, AI_MOODS['idle'])
            world.meta.color_shift = mood_data.get('color_shift', 0.0)
            self._history.append({"role": "ai", "text": response})
            world.meta.ai_thinking = False

        threading.Thread(target=_do_ai, daemon=True).start()
