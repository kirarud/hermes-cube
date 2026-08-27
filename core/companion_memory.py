"""companion_memory.py — Мини-память Companion (Markdown на диске).

Файл: %APPDATA%/HermesCube/memory.md

Читается в system prompt AI. Можно править вручную.
Команды в чате:
  запомни: <факт>   — добавить строку
  /forget <фрагмент> — удалить строки, содержащие фрагмент
"""

from __future__ import annotations

import os
import re
from typing import List, Optional, Tuple

_MEMORY_DIR = os.path.join(
    os.environ.get('APPDATA', os.path.expanduser('~')),
    'HermesCube',
)
MEMORY_PATH = os.path.join(_MEMORY_DIR, 'memory.md')

_DEFAULT = """# Hermes Cube — память о пользователе

## Preferences
- Язык общения: русский

## Facts
- (добавляй факты: «запомни: ...» в чате куба)

## Notes
- North Star: живой companion на рабочем столе
"""


def ensure_memory_file() -> str:
    """Создать memory.md если нет. Вернуть путь."""
    os.makedirs(_MEMORY_DIR, exist_ok=True)
    if not os.path.isfile(MEMORY_PATH):
        with open(MEMORY_PATH, 'w', encoding='utf-8') as f:
            f.write(_DEFAULT)
    return MEMORY_PATH


def load_memory() -> str:
    """Прочитать память (пусто → default file)."""
    try:
        ensure_memory_file()
        with open(MEMORY_PATH, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except OSError:
        return _DEFAULT.strip()


def append_fact(fact: str) -> bool:
    """Добавить факт в секцию Facts."""
    fact = fact.strip()
    if not fact:
        return False
    try:
        ensure_memory_file()
        with open(MEMORY_PATH, 'r', encoding='utf-8') as f:
            text = f.read()
        bullet = f'- {fact}\n'
        if '## Facts' in text:
            # вставить после заголовка Facts
            parts = text.split('## Facts', 1)
            rest = parts[1]
            # пропустить первую строку после заголовка
            if rest.startswith('\n'):
                rest = rest[1:]
            text = parts[0] + '## Facts\n' + bullet + rest
            if not text.endswith('\n'):
                text += '\n'
        else:
            text = text.rstrip() + '\n\n## Facts\n' + bullet
        with open(MEMORY_PATH, 'w', encoding='utf-8') as f:
            f.write(text)
        return True
    except OSError:
        return False


def forget_fragment(fragment: str) -> int:
    """Удалить строки-буллеты, содержащие fragment. Вернуть число удалений."""
    fragment = fragment.strip().lower()
    if not fragment:
        return 0
    try:
        ensure_memory_file()
        with open(MEMORY_PATH, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        kept: List[str] = []
        removed = 0
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('- ') and fragment in stripped.lower():
                removed += 1
                continue
            kept.append(line)
        if removed:
            with open(MEMORY_PATH, 'w', encoding='utf-8') as f:
                f.writelines(kept)
        return removed
    except OSError:
        return 0


def parse_memory_command(text: str) -> Optional[Tuple[str, str]]:
    """Если сообщение — команда памяти, вернуть (action, payload).

    action: 'remember' | 'forget'
    """
    t = text.strip()
    m = re.match(r'^(?:запомни|remember)\s*[:：]\s*(.+)$', t, re.IGNORECASE | re.DOTALL)
    if m:
        return 'remember', m.group(1).strip()
    m = re.match(r'^(?:/forget|забудь)\s+(.+)$', t, re.IGNORECASE | re.DOTALL)
    if m:
        return 'forget', m.group(1).strip()
    return None


def memory_for_prompt(max_chars: int = 1200) -> str:
    """Сжатый фрагмент для system prompt."""
    raw = load_memory()
    if len(raw) > max_chars:
        raw = raw[:max_chars] + '\n…'
    return raw
