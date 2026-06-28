#!/usr/bin/env python3
"""progress_watcher.py — Следит за прогрессом Engine Blueprint v2.

На каждый тик:
  1. Читает текущий progress из Obsidian-плана
  2. Проверяет git-репозиторий куба на новые коммиты
  3. Если есть новые коммиты → +10% в плане и вывод процента
  4. Если нет → тишина (ничего не отправляется)

Скрипт для cron-режима no_agent=True.
stdout: "NEXT:20" → доставляется в чат как сообщение.
stdout: "" → тишина (ничего не отправляется).
"""

import json
import os
import re
import subprocess
import sys

# --- Конфиг ---
PLAN_FILE = r"C:\Users\kirarud\Documents\Obsidian Vault\Hermes Cube\Development Plan\Engine Blueprint v2 — 100 шагов.md"
CUBE_REPO = r"C:\Users\kirarud\hermes-cube-repo"
STATE_DIR = os.path.join(os.environ.get("HOME", os.environ.get("USERPROFILE", ".")), ".hermes")
STATE_FILE = os.path.join(STATE_DIR, "progress_watcher_state.json")


def get_current_progress(content: str) -> int:
    """Извлечь progress: N из frontmatter."""
    m = re.search(r'progress:\s*(\d+)', content)
    return int(m.group(1)) if m else 0


def get_latest_commit() -> str:
    """Последний commit hash в репозитории куба."""
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "-1", "--format=%H"],
            cwd=CUBE_REPO,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def main() -> None:
    # Читаем план
    try:
        with open(PLAN_FILE, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        print("ERROR: plan file not found")
        sys.exit(0)

    current = get_current_progress(content)
    if current >= 100:
        print("DONE:100")
        return

    # Читаем состояние последнего чека
    last_commit = ""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                state = json.load(f)
            last_commit = state.get("last_commit", "")
        except Exception:
            pass

    latest_commit = get_latest_commit()
    if not latest_commit:
        # Нет коммитов — тишина
        return

    if latest_commit == last_commit:
        # Коммит не менялся — прогресса нет
        return

    # Есть новый коммит → +10%
    new_progress = min(current + 10, 100)

    # Обновляем план
    new_content = re.sub(r'progress:\s*\d+', f'progress: {new_progress}', content)
    # Также обновляем updated и строку прогресса в теле
    from datetime import date
    today = date.today().strftime("%Y-%m-%d")
    new_content = re.sub(r'updated: \d{4}-\d{2}-\d{2}', f'updated: {today}', new_content)
    new_content = re.sub(
        r'\*\*Прогресс:\*\* \d+/\d+ шагов \(\d+%\)',
        f'**Прогресс:** {new_progress}/100 шагов ({new_progress}%)',
        new_content,
    )

    os.makedirs(os.path.dirname(PLAN_FILE), exist_ok=True)
    with open(PLAN_FILE, "w", encoding="utf-8") as f:
        f.write(new_content)

    # Сохраняем состояние
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump({"last_commit": latest_commit}, f)

    # Выводим — это попадёт в чат
    print(f"NEXT:{new_progress}")


if __name__ == "__main__":
    main()
