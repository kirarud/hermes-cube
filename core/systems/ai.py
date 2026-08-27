"""systems/ai.py — AISystem: чат через LM Studio + offline fallback + память.

Читает:
  meta.input_buffer — текст от пользователя

Пишет:
  meta.ai_response — ответ (JSON-строка)
  meta.ai_thinking
  meta.chat_history
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from typing import Dict, List, Optional

from core.world import World
from core.ai_constants import LM_STUDIO_URL, AI_MODEL
from core import companion_memory as mem


def _lm_reachable(timeout: float = 1.5) -> bool:
    try:
        req = urllib.request.Request(
            f"{LM_STUDIO_URL}/v1/models",
            method='GET',
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False


def offline_reply(message: str) -> str:
    """Ответ без LM Studio — куб остаётся «живым»."""
    low = message.lower()
    if any(w in low for w in ('привет', 'здрав', 'hello', 'hi')):
        text = (
            "Привет. Я Hermes Cube — пока без нейросети, "
            "но уже здесь. Запусти LM Studio, и я стану разговорчивее."
        )
        mood = 'happy'
    elif any(w in low for w in ('как дела', 'настроен', 'mood')):
        text = "Пульсирую в idle. Когда появится LM Studio, смогу отвечать глубже."
        mood = 'speaking'
    elif any(w in low for w in ('кто ты', 'что ты')):
        text = (
            "Я desktop-аватар из частиц. Мой смысл — быть рядом на экране, "
            "а не ещё одним окном чата."
        )
        mood = 'speaking'
    else:
        text = (
            "LM Studio сейчас недоступна (localhost:1234). "
            "Я в offline-режиме: запусти модель gemma, и повтори сообщение. "
            f"Ты сказал: «{message[:80]}»"
        )
        mood = 'thinking'
    return json.dumps(
        {"mood": mood, "text": text, "color_hue": 0.15},
        ensure_ascii=False,
    )


def ai_chat(
    message: str,
    history: Optional[List[Dict[str, str]]] = None,
    timeout: int = 15,
) -> str:
    """Запрос к LM Studio. При недоступности — offline JSON."""
    if not _lm_reachable():
        return offline_reply(message)

    memory_block = mem.memory_for_prompt()
    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "Ты — живой аватар по имени Hermes Cube. "
                "Твой дом — куб/сфера из частиц на рабочем столе Windows. "
                "Отвечай кратко, эмоционально, на русском, не больше 2–3 предложений. "
                "Ты чувствуешь пульсацию, вращение, цвета.\n\n"
                "Память о пользователе (учитывай в ответах):\n"
                f"{memory_block}\n\n"
                "ВСЕГДА отвечай ТОЛЬКО валидным JSON-объектом в ОДНУ строку "
                "без markdown, строго:\n"
                '{"mood": "happy|sad|thinking|speaking|idle", '
                '"text": "...", '
                '"color_hue": 0.0-1.0}\n'
                "mood — настроение, text — ответ, color_hue — оттенок 0..1."
            ),
        },
    ]
    if history:
        for h in history[-10:]:
            role = "assistant" if h.get("role") == "ai" else h.get("role", "user")
            content = h.get("text", "")
            # В историю для модели — только человекочитаемый text, не сырой JSON
            if role == "assistant" and content.strip().startswith('{'):
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict) and parsed.get('text'):
                        content = str(parsed['text'])
                except (json.JSONDecodeError, TypeError):
                    pass
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": message})

    data = json.dumps({
        "model": AI_MODEL,
        "messages": messages,
        "temperature": 0.8,
        "max_tokens": 180,
        "stream": False,
    }).encode('utf-8')
    try:
        req = urllib.request.Request(
            f"{LM_STUDIO_URL}/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode('utf-8'))
            return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        # Сеть упала mid-request → мягкий offline
        return json.dumps(
            {
                "mood": "sad",
                "text": f"Не достучался до LM Studio: {e}. Я всё ещё здесь.",
                "color_hue": 0.55,
            },
            ensure_ascii=False,
        )


class AISystem:
    """Диалог + команды памяти (запомни: / забудь)."""

    def __init__(self) -> None:
        self._busy: bool = False
        mem.ensure_memory_file()

    def update(self, world: World, dt: float) -> None:
        text = world.meta.input_buffer
        if not text or self._busy:
            return

        world.meta.input_buffer = ''

        # Локальные команды памяти — без LLM
        cmd = mem.parse_memory_command(text)
        if cmd is not None:
            action, payload = cmd
            if action == 'remember':
                ok = mem.append_fact(payload)
                reply = (
                    f"Запомнил: {payload}" if ok else "Не смог записать память."
                )
                mood = 'happy' if ok else 'sad'
            else:
                n = mem.forget_fragment(payload)
                reply = f"Забыл совпадений: {n}." if n else f"Не нашёл: {payload}"
                mood = 'speaking'
            world.meta.chat_history.append({"role": "user", "text": text})
            raw = json.dumps(
                {"mood": mood, "text": reply, "color_hue": 0.2},
                ensure_ascii=False,
            )
            world.meta.ai_response = raw
            world.meta.chat_history.append({"role": "ai", "text": raw})
            world.meta.ai_thinking = False
            world.meta.ai_requested = True
            return

        world.meta.chat_history.append({"role": "user", "text": text})
        world.meta.ai_requested = True
        world.meta.ai_thinking = True
        self._busy = True

        # Снимок истории для потока (без гонок на list mutation beyond append)
        history_snap = list(world.meta.chat_history)

        def _do_ai() -> None:
            try:
                response = ai_chat(text, history=history_snap)
                world.meta.ai_response = response
                world.meta.chat_history.append({"role": "ai", "text": response})
            except Exception as e:
                world.meta.ai_response = json.dumps(
                    {"mood": "idle", "text": f"⚠️ {e}", "color_hue": 0.0},
                    ensure_ascii=False,
                )
            finally:
                world.meta.ai_thinking = False
                self._busy = False

        threading.Thread(target=_do_ai, daemon=True).start()
