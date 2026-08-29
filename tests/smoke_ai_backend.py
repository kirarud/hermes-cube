"""smoke_ai_backend.py — проверка чат-бэкенда куба без запуска GUI.

Запуск:  python tests/smoke_ai_backend.py "привет"

Проверяет:
  1. resolve_ai_backend() нашёл endpoint
  2. endpoint отвечает (_lm_reachable)
  3. ai_chat вернул валидный JSON с mood/text/color_hue
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ai_constants import AI_BASE_URL, AI_API_KEY, AI_MODEL, AI_MOODS
from core.systems import ai as ai_mod


def main() -> int:
    message = sys.argv[1] if len(sys.argv) > 1 else "привет"

    print(f"base_url : {AI_BASE_URL}")
    print(f"model    : {AI_MODEL}")
    print(f"api_key  : {'set (' + str(len(AI_API_KEY)) + ' chars)' if AI_API_KEY else 'NONE'}")

    reachable = ai_mod._lm_reachable()
    print(f"reachable: {reachable}")

    raw = ai_mod.ai_chat(message)
    print(f"raw      : {raw[:300]}")

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"FAIL: ответ не JSON ({e})")
        return 1

    mood = parsed.get("mood")
    text = parsed.get("text")
    hue = parsed.get("color_hue")

    print(f"mood     : {mood}")
    print(f"text     : {text}")
    print(f"color_hue: {hue}")

    ok = True
    if mood not in AI_MOODS:
        print(f"FAIL: mood '{mood}' не из {sorted(AI_MOODS)}")
        ok = False
    if not text:
        print("FAIL: пустой text")
        ok = False
    if not isinstance(hue, (int, float)) or not 0.0 <= float(hue) <= 1.0:
        print(f"FAIL: color_hue '{hue}' не число 0..1")
        ok = False
    if not reachable:
        print("WARN: сработал offline_reply, не настоящая модель")
        ok = False

    print("OK: живой чат-бэкенд" if ok else "NOT OK")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
