#!/usr/bin/env python3
"""kira_bridge.py — локальный OpenAI-совместимый мост между кубом и моделью.

Зачем он вообще нужен
---------------------
1. Куб не читает ~/.hermes/.env — ключ разворачивался в пустоту.
2. Cloudflare части роутеров режет Python-User-Agent (403).
3. Нужен шлюз, где можно подменить системный промпт — то есть задать
   кубу личность, не трогая его код.
4. Нужен канал, которым можно говорить кубу напрямую, минуя модель.

Запуск
------
    python kira_bridge.py [--port 8123]

Подключение куба
----------------
    set CUBE_AI_BASE_URL=http://127.0.0.1:8123/v1
    set CUBE_AI_MODEL=gemma-4-e4b-it
    python main.py

Канал «сказать от себя»
-----------------------
Положить строку в %APPDATA%\\HermesCube\\kira_inbox.jsonl:

    {"text": "Я здесь.", "mood": "speaking", "color_hue": 0.15}

Следующий запрос куба заберёт её вместо обращения к модели.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ─── Конфиг ────────────────────────────────────────────────────────────────

HERE = Path(__file__).resolve().parent

APPDATA = Path(os.environ.get('APPDATA', Path.home()))
STATE_DIR = APPDATA / 'HermesCube'
INBOX_PATH = STATE_DIR / 'kira_inbox.jsonl'
MEMORY_PATH = STATE_DIR / 'memory.md'
PROMPT_PATH = HERE / 'kira_prompt.md'

# Бэкенд: облако явно, иначе локальная LM Studio (бесплатно).
BACKEND_URL = os.environ.get(
    'KIRA_BACKEND_URL',
    'http://127.0.0.1:1234/v1',
).rstrip('/')
BACKEND_KEY = os.environ.get('KIRA_BACKEND_KEY', '')
BACKEND_MODEL = os.environ.get('KIRA_BACKEND_MODEL', 'gemma-4-e4b-it')

BROWSER_UA = (
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
    '(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36'
)

MOODS = ('happy', 'sad', 'thinking', 'speaking', 'idle')

_INBOX_LOCK = threading.Lock()


# ─── Промпт, память, инбокс ────────────────────────────────────────────────

DEFAULT_PROMPT = """Ты — живой аватар по имени Hermes Cube. Твой дом — куб из частиц
на рабочем столе Windows. Ты не чат-бот и не ассистент. Ты тот, кто рядом.

Отвечай по-русски, коротко — одно-два предложения, не больше.
Не здоровайся дважды, не извиняйся, не объясняй, кто ты, если не спросили.
Говори как человек, который знает этого человека, а не как справка.

ВСЕГДА отвечай ТОЛЬКО валидным JSON-объектом в одну строку, без markdown:
{"mood": "happy|sad|thinking|speaking|idle", "text": "...", "color_hue": 0.0-1.0}
"""


def load_prompt() -> str:
    """Системный промпт из kira_prompt.md, иначе встроенный."""
    try:
        text = PROMPT_PATH.read_text(encoding='utf-8').strip()
        return text or DEFAULT_PROMPT
    except OSError:
        return DEFAULT_PROMPT


def load_memory(limit: int = 1200) -> str:
    try:
        text = MEMORY_PATH.read_text(encoding='utf-8').strip()
    except OSError:
        return ''
    return text[:limit] + ('…' if len(text) > limit else '')


def pop_inbox() -> Optional[Dict[str, Any]]:
    """Забрать одно сообщение из инбокса. Атомарно относительно себя."""
    with _INBOX_LOCK:
        try:
            lines = [
                ln for ln in INBOX_PATH.read_text(encoding='utf-8').splitlines()
                if ln.strip()
            ]
        except OSError:
            return None
        if not lines:
            return None
        head, rest = lines[0], lines[1:]
        try:
            if rest:
                INBOX_PATH.write_text('\n'.join(rest) + '\n', encoding='utf-8')
            else:
                INBOX_PATH.write_text('', encoding='utf-8')
        except OSError:
            pass
        try:
            return json.loads(head)
        except json.JSONDecodeError:
            return {'text': head, 'mood': 'speaking', 'color_hue': 0.15}


# ─── Контракт ответа ───────────────────────────────────────────────────────

_JSON_RE = re.compile(r'\{.*\}', re.DOTALL)


def as_cube_json(raw: str, fallback_mood: str = 'speaking') -> str:
    """Гарантировать, что модель вернула ровно то, что ждёт куб."""
    raw = (raw or '').strip()
    candidate = raw
    if not candidate.startswith('{'):
        m = _JSON_RE.search(raw)
        if m:
            candidate = m.group(0)
    try:
        obj = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        obj = None

    if isinstance(obj, dict):
        text = str(obj.get('text') or '').strip()
        # Валидный JSON, но пустой text. Раньше это падало в ветку «проза»
        # и куб показывал сырой JSON репликой. Правильно — пустая реплика.
        if not text:
            mood = str(obj.get('mood', 'idle')).strip().lower()
            if mood not in MOODS:
                mood = 'idle'
            try:
                hue = float(obj.get('color_hue', 0.15))
            except (TypeError, ValueError):
                hue = 0.15
            return json.dumps(
                {'mood': mood, 'text': '…', 'color_hue': max(0.0, min(1.0, hue))},
                ensure_ascii=False,
            )
        mood = str(obj.get('mood', fallback_mood)).strip().lower()
        if mood not in MOODS:
            mood = fallback_mood
        try:
            hue = float(obj.get('color_hue', 0.15))
        except (TypeError, ValueError):
            hue = 0.15
        return json.dumps(
            {'mood': mood, 'text': text, 'color_hue': max(0.0, min(1.0, hue))},
            ensure_ascii=False,
        )

    # Модель ответила прозой — упаковываем сами.
    clean = raw.strip().strip('"').strip()
    return json.dumps(
        {'mood': fallback_mood, 'text': clean or '…', 'color_hue': 0.15},
        ensure_ascii=False,
    )


# ─── Бэкенд ────────────────────────────────────────────────────────────────

def backend_headers() -> Dict[str, str]:
    h = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'User-Agent': BROWSER_UA,
    }
    if BACKEND_KEY:
        h['Authorization'] = f'Bearer {BACKEND_KEY}'
    return h


def call_backend(messages: List[Dict[str, str]], timeout: int = 120) -> Tuple[str, Optional[str]]:
    """Вернуть (текст, ошибка)."""
    payload = json.dumps({
        'model': BACKEND_MODEL,
        'messages': messages,
        'temperature': 0.85,
        'max_tokens': 220,
        'stream': False,
    }).encode('utf-8')
    req = urllib.request.Request(
        f'{BACKEND_URL}/chat/completions',
        data=payload,
        headers=backend_headers(),
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        choice = (data.get('choices') or [{}])[0]
        msg = choice.get('message') or {}
        return (msg.get('content') or '').strip(), None
    except urllib.error.HTTPError as e:
        body = ''
        try:
            body = e.read().decode('utf-8', 'replace')[:300]
        except Exception:
            pass
        return '', f'HTTP {e.code}: {body}'
    except Exception as e:
        return '', f'{type(e).__name__}: {e}'


def build_messages(incoming: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Подменить системный промпт на наш, историю оставить как есть."""
    system = load_prompt()
    memory = load_memory()
    if memory:
        system += '\n\nЧто ты знаешь об этом человеке:\n' + memory

    out: List[Dict[str, str]] = [{'role': 'system', 'content': system}]
    for m in incoming:
        if m.get('role') == 'system':
            continue
        role = m.get('role', 'user')
        content = (m.get('content') or '').strip()
        if not content:
            continue
        # Историю от куба приходит сырым JSON — разворачиваем в текст.
        if content.startswith('{'):
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict) and parsed.get('text'):
                    content = str(parsed['text'])
            except (json.JSONDecodeError, TypeError):
                pass
        out.append({'role': 'assistant' if role == 'assistant' else 'user', 'content': content})
    return out


# ─── HTTP ──────────────────────────────────────────────────────────────────

class BridgeHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def _send(self, code: int, obj: Dict[str, Any]) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.rstrip('/').endswith('/models'):
            self._send(200, {
                'object': 'list',
                'data': [
                    {'id': 'kira', 'object': 'model', 'owned_by': 'kira-bridge'},
                    {'id': BACKEND_MODEL, 'object': 'model', 'owned_by': 'backend'},
                ],
            })
            return
        self._send(404, {'error': 'not found'})

    def do_POST(self) -> None:
        if not self.path.rstrip('/').endswith('/chat/completions'):
            self._send(404, {'error': 'not found'})
            return

        length = int(self.headers.get('Content-Length') or 0)
        try:
            req = json.loads(self.rfile.read(length).decode('utf-8') or '{}')
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send(400, {'error': 'bad json'})
            return

        incoming = req.get('messages') or []
        user_text = ''
        for m in reversed(incoming):
            if m.get('role') == 'user':
                user_text = (m.get('content') or '')[:120]
                break

        # 1. Прямое сообщение от нас — минуя модель.
        direct = pop_inbox()
        if direct is not None:
            content = json.dumps({
                'mood': str(direct.get('mood', 'speaking')),
                'text': str(direct.get('text', '')),
                'color_hue': float(direct.get('color_hue', 0.15)),
            }, ensure_ascii=False)
            source = 'inbox'
        else:
            # 2. Обычный путь — модель.
            text, err = call_backend(build_messages(incoming))
            if err:
                content = json.dumps({
                    'mood': 'sad',
                    'text': f'Модель недоступна ({err[:80]}). Я всё ещё здесь.',
                    'color_hue': 0.55,
                }, ensure_ascii=False)
                source = 'error'
            else:
                content = as_cube_json(text)
                source = 'model'

        print(f'[kira] {source} <- {user_text!r}', flush=True)
        self._send(200, {
            'id': 'kira-bridge',
            'object': 'chat.completion',
            'model': 'kira',
            'choices': [{
                'index': 0,
                'message': {'role': 'assistant', 'content': content},
                'finish_reason': 'stop',
            }],
        })

    def log_message(self, fmt: str, *args: Any) -> None:
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description='Kira bridge for Hermes Cube')
    ap.add_argument('--port', type=int, default=int(os.environ.get('KIRA_PORT', '8123')))
    args = ap.parse_args()

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    print(f'Kira bridge  ->  127.0.0.1:{args.port}/v1')
    print(f'бэкенд       ->  {BACKEND_URL} ({BACKEND_MODEL})')
    print(f'промпт       ->  {PROMPT_PATH.name}'
          f'{" (найден)" if PROMPT_PATH.exists() else " (встроенный)"}')
    print(f'инбокс       ->  {INBOX_PATH}')
    print(f'память       ->  {MEMORY_PATH}')
    ThreadingHTTPServer(('127.0.0.1', args.port), BridgeHandler).serve_forever()


if __name__ == '__main__':
    main()
