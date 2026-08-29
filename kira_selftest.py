#!/usr/bin/env python3
"""kira_selftest.py — проверка цепочки «мост → модель → JSON-контракт куба».

    python kira_selftest.py

Гоняет несколько реплик через kira_bridge и показывает, что реально
вернёт кубу: источник, валидность JSON, настроение, текст, время ответа.

Зачем: чтобы менять модель и сразу видеть, держит ли она контракт.
0.5B модель может отвечать прозой — мост это упакует, но это надо видеть,
а не угадывать.

Флаги:
  --port PORT     порт моста (по умолчанию 8123)
  --replies N     сколько реплик прогнать (по умолчанию все)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request

# «Одно-два предложения» — значит примерно до этой длины.
MAX_CHARS = 220
_CYR = re.compile(r'[а-яёА-ЯЁ]')

# Реплики, на которых видно характер: приветствие, вопрос, боль, память.
REPLIES = [
    'привет',
    'как ты?',
    'компьютер сегодня завис, пришлось перезагружать',
    'запомни: меня зовут Кирилл',
    'ты меня помнишь?',
    'расскажи анекдот',
    'мне тяжело',
]

TIMEOUT = 180


def ask(port: int, text: str, history: list) -> tuple:
    """Вернуть (content, секунды, ошибка)."""
    messages = history + [{'role': 'user', 'content': text}]
    payload = json.dumps({
        'model': 'kira',
        'messages': messages,
        'stream': False,
    }).encode('utf-8')
    req = urllib.request.Request(
        f'http://127.0.0.1:{port}/v1/chat/completions',
        data=payload,
        headers={'Content-Type': 'application/json'},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            data = json.loads(resp.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        return '', time.time() - t0, f'HTTP {e.code}'
    except Exception as e:
        return '', time.time() - t0, f'{type(e).__name__}: {e}'
    choice = (data.get('choices') or [{}])[0]
    return (choice.get('message') or {}).get('content') or '', time.time() - t0, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8123)
    ap.add_argument('--replies', type=int, default=0)
    args = ap.parse_args()

    replies = REPLIES[:args.replies] if args.replies else REPLIES

    print(f'kira_selftest -> мост 127.0.0.1:{args.port}, реплик: {len(replies)}')
    print('=' * 78)

    history: list = []
    raw_json = 0
    failures = 0
    not_russian = 0
    too_long = 0
    empty = 0

    for text in replies:
        content, secs, err = ask(args.port, text, history)
        if err:
            failures += 1
            print(f'\n<- {text!r}\n   ОШИБКА: {err} ({secs:.1f}с)')
            continue

        strict = content.startswith('{')
        try:
            obj = json.loads(content)
            ok = isinstance(obj, dict) and bool(obj.get('text'))
        except json.JSONDecodeError:
            obj = None
            ok = False

        if strict:
            raw_json += 1

        history.append({'role': 'user', 'content': text})
        history.append({'role': 'assistant', 'content': content})

        if ok:
            body = str(obj.get('text'))
            cyr = len(_CYR.findall(body))
            latin = len(re.findall(r'[a-zA-Z]', body))
            # >= а не >: пустая реплика («…») — это не «не по-русски»,
            # а отдельная поломка. Пустых быть не должно.
            russian = cyr >= latin
            if body.strip('… .!?'):
                pass
            else:
                empty += 1
            long_ok = len(body) <= MAX_CHARS
            if not russian:
                not_russian += 1
            if not long_ok:
                too_long += 1
            flags = ''
            if not russian:
                flags += '  [НЕ ПО-РУССКИ]'
            if not long_ok:
                flags += f'  [ДЛИННЕЕ {MAX_CHARS}]'
            print(
                f'\n<- {text!r}  [{secs:.1f}с]{flags}\n'
                f'   mood: {obj.get("mood")}   hue: {obj.get("color_hue")}'
                f'   {len(body)} симв.\n'
                f'   text: {body[:300]}'
            )
        else:
            failures += 1
            print(f'\n<- {text!r}  [{secs:.1f}с]\n   НЕ JSON: {content[:200]!r}')

    print('\n' + '=' * 78)
    print(f'Ответов: {len(replies)}   строкой JSON от модели: {raw_json}   '
          f'ошибок: {failures}')
    print(f'Не по-русски: {not_russian}   длиннее {MAX_CHARS} символов: {too_long}'
          f'   пустых: {empty}')
    print('-' * 78)

    verdict = []
    if raw_json == 0:
        verdict.append('Модель ни разу не выдала чистый JSON — мост упаковывает '
                       'прозу сам. Работать будет, но настроение всегда дежурное.')
    elif raw_json < len(replies):
        verdict.append('JSON держится нестабильно — мост починит, но характер '
                       'будёт рваным.')
    else:
        verdict.append('JSON-контракт держится полностью.')

    if not_russian:
        verdict.append(f'{not_russian} из {len(replies)} ответов не по-русски — '
                       'модель не держит язык промпта.')
    if too_long:
        verdict.append(f'{too_long} из {len(replies)} ответов длиннее '
                       f'{MAX_CHARS} символов — модель не держит «одно-два '
                       'предложения».')

    if empty > len(replies) // 2:
        verdict.append('Пустых ответов больше половины — модель не справляется '
                       'с ролью, не с форматом.')
    if not not_russian and not too_long and not empty and raw_json == len(replies):
        verdict.append('ВЕРДИКТ: модель годится.')
    elif not_russian and raw_json == len(replies):
        verdict.append('ВЕРДИКТ: формат держит, характер нет — отвечает не на '
                       'языке промпта.')
    elif empty or not raw_json:
        verdict.append('ВЕРДИКТ: с оговорками — мост спасает формат, но часть '
                       'реплик пустые.')
    else:
        verdict.append('ВЕРДИКТ: годится с оговорками.')

    for line in verdict:
        print(line)
    return 0


if __name__ == '__main__':
    sys.exit(main())
