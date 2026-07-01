"""systems/ai.py — AISystem: AI-чат через LM Studio.

Содержит ai_chat() — функцию отправки запроса к LM Studio API.
Ранее находилась в ai_module.py, перемещена сюда при удалении старого модуля.

Читает:
  meta.input_buffer — текст от пользователя

Пишет:
  meta.ai_response — ответ AI
  meta.ai_thinking — True пока модель думает
  meta.chat_history — история диалога
"""

from __future__ import annotations

import json
import threading
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional

from core.world import World
from core.ai_constants import LM_STUDIO_URL, AI_MODEL


def ai_chat(
    message: str,
    history: Optional[List[Dict[str, str]]] = None,
    timeout: int = 15,
) -> str:
    """Send a message to LM Studio and return raw response text.

    System prompt instructs the model to reply with JSON:
       {"mood": "...", "text": "...", "color_hue": 0.0-1.0}

    history: list of {"role": "user"|"assistant", "text": ...}.
    """
    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "Ты — живой аватар по имени Hermes Cube. "
                "Твой дом — сфера из частиц. "
                "Отвечай кратко, эмоционально, не больше 2-3 предложений. "
                "Ты чувствуешь пульсацию, вращение, цвета вокруг себя.\n\n"
                "ВСЕГДА отвечай ТОЛЬКО валидным JSON-объектом в ОДНУ строку "
                "без пояснений и markdown, строго в формате:\n"
                '{"mood": "happy|sad|thinking|speaking|idle", '
                '"text": "...", '
                '"color_hue": 0.0-1.0}\n'
                "Поле mood отражает твоё настроение. "
                "Поле text — твой ответ пользователю. "
                "Поле color_hue — оттенок цвета (0.0-1.0)."
            ),
        },
    ]
    if history:
        for h in history[-10:]:
            role = "assistant" if h["role"] == "ai" else h["role"]
            messages.append({"role": role, "content": h["text"]})
    messages.append({"role": "user", "content": message})

    data = json.dumps({
        "model": AI_MODEL,
        "messages": messages,
        "temperature": 0.8,
        "max_tokens": 180,
        "stream": False,
    }).encode()
    try:
        req = urllib.request.Request(
            f"{LM_STUDIO_URL}/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read())
            return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return json.dumps({"mood": "idle", "text": f"[Error: {e}]", "color_hue": 0.0})


class AISystem:
    """Система AI-диалога.

    При появлении input_buffer запускает поток с ai_chat().
    Результат: ответ в world.meta.ai_response.
    """

    def update(self, world: World, dt: float) -> None:
        text = world.meta.input_buffer
        if not text:
            return

        world.meta.input_buffer = ''
        world.meta.chat_history.append({"role": "user", "text": text})
        world.meta.ai_requested = True
        world.meta.ai_thinking = True

        def _do_ai(history: list) -> None:
            try:
                response = ai_chat(text, history=history)
                world.meta.ai_response = response
                world.meta.chat_history.append({"role": "ai", "text": response})
            except Exception as e:
                world.meta.ai_response = json.dumps({"mood": "idle", "text": f"⚠️ {e}", "color_hue": 0.0})
            finally:
                world.meta.ai_thinking = False

        threading.Thread(target=_do_ai, args=(world.meta.chat_history,), daemon=True).start()
