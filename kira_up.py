#!/usr/bin/env python3
"""kira_up.py — поднять всё одной командой, без LM Studio.

    python kira_up.py

Почему без LM Studio
--------------------
LM Studio вешает машину при запуске (проверено 29.08.2026: глухой фриз,
только ребут). Но внутри неё лежит обычный llama.cpp — `llama-server.exe`,
настоящий OpenAI-совместимый сервер. Мы берём его напрямую, минуя GUI.
Ни Electron, ни GPU-акселерации, ничего лишнего.

Что делает по порядку:
  1. Находит llama-server.exe среди бэкендов LM Studio.
  2. Поднимает модель на 127.0.0.1:1234 (по умолчанию — CPU-сборка).
  3. Поднимает kira_bridge.py на 8123.
  4. Запускает куб: python main.py с CUBE_AI_BASE_URL на мост.

Всё локально. Ни копейки, ни одного внешнего запроса.

Флаги:
  --model NAME     короткое имя или путь к .gguf
                   (по умолчанию gemma-4-e4b-it — та, под которую писался куб)
  --list           показать найденные модели и выйти
  --gpu            использовать CUDA-сборку вместо CPU
  --ctx N          контекст, по умолчанию 4096
  --threads N      потоков, по умолчанию 4
  --no-cube        не запускать сам куб (только инфраструктуру)
  --wait SECONDS   сколько ждать подъёма модели (по умолчанию 300)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
BRIDGE = HERE / 'kira_bridge.py'
DEFAULT_MODEL = 'gemma-4-e4b-it'
MODEL_DIR = Path.home() / '.lmstudio' / 'models'
BACKENDS_DIR = Path.home() / '.lmstudio' / 'extensions' / 'backends'

LLAMA_PORT = 1234
BRIDGE_PORT = 8123

PY = sys.executable


# ─── Поиск бэкенда и моделей ───────────────────────────────────────────────

def find_backend(prefer_gpu: bool) -> Path | None:
    """Найти llama-server.exe. CPU по умолчанию — GPU остаётся кубу."""
    if not BACKENDS_DIR.is_dir():
        return None
    found = sorted(BACKENDS_DIR.glob('*/llama-server.exe'))
    if not found:
        return None

    cpu = [p for p in found if 'cuda' not in p.parent.name.lower()
           and 'vulkan' not in p.parent.name.lower()]
    gpu = [p for p in found if 'cuda' in p.parent.name.lower()]

    if prefer_gpu and gpu:
        return gpu[-1]          # самая свежая CUDA-сборка
    if cpu:
        return cpu[-1]
    return found[-1]


def find_models() -> list[Path]:
    if not MODEL_DIR.is_dir():
        return []
    return sorted(
        p for p in MODEL_DIR.rglob('*.gguf')
        if 'mmproj' not in p.name.lower()
    )


def resolve_model(name: str) -> Path | None:
    """По короткому имени найти .gguf. Имя может быть и готовым путём."""
    direct = Path(name)
    if direct.is_file():
        return direct

    needle = name.lower()
    models = find_models()
    for m in models:
        if m.stem.lower() == needle:
            return m
    for m in models:
        if needle in str(m).lower():
            return m
    return None


# ─── Сеть ──────────────────────────────────────────────────────────────────

def _get(url: str, timeout: float = 3.0) -> bool:
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'kira-up/1.0'})
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except Exception:
        return False


def llama_up() -> bool:
    return _get(f'http://127.0.0.1:{LLAMA_PORT}/v1/models')


def bridge_up() -> bool:
    return _get(f'http://127.0.0.1:{BRIDGE_PORT}/v1/models')


def wait_for(probe, seconds: int, label: str) -> bool:
    deadline = time.time() + seconds
    spin = 0
    while time.time() < deadline:
        if probe():
            print(f'\r   {label}: готово                ')
            return True
        spin += 1
        print(f'\r   {label}: жду {"·" * (spin % 4):<4} '
              f'{int(deadline - time.time()):>3}с', end='', flush=True)
        time.sleep(2)
    print()
    return False


# ─── Сценарий ──────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=DEFAULT_MODEL)
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--gpu', action='store_true')
    ap.add_argument('--ctx', type=int, default=4096)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--no-cube', action='store_true')
    ap.add_argument('--wait', type=int, default=300)
    args = ap.parse_args()

    if args.list:
        models = find_models()
        if not models:
            print(f'Моделей не нашёл в {MODEL_DIR}')
            return 1
        print(f'Модели в {MODEL_DIR}:\n')
        for m in models:
            print(f'  {m.stem:<62} {m.stat().st_size / 1e9:5.2f} ГБ')
        backend = find_backend(args.gpu)
        print(f'\nБэкенд: {backend or "НЕ НАЙДЕН"}')
        return 0

    backend = find_backend(args.gpu)
    if not backend:
        print(f'Не нашёл llama-server.exe в {BACKENDS_DIR}')
        print('Поставь LM Studio хотя бы один раз — она принесёт бэкенды.')
        return 1

    model_path = resolve_model(args.model)
    if not model_path:
        print(f'Не нашёл модель {args.model!r}. Список: python kira_up.py --list')
        return 1

    print('=== 1. Модель ===')
    print(f'   бэкенд: {backend.parent.name}')
    print(f'   модель: {model_path.name} ({model_path.stat().st_size / 1e9:.2f} ГБ)')

    if llama_up():
        print('   порт 1234 уже занят — использую то, что там отвечает')
    else:
        cmd = [
            str(backend),
            '-m', str(model_path),
            '--port', str(LLAMA_PORT),
            '--host', '127.0.0.1',
            '-c', str(args.ctx),
            '-t', str(args.threads),
            # Мост шлёт именно это имя модели. Алиас снимает расхождение.
            '-a', DEFAULT_MODEL,
        ]
        try:
            subprocess.Popen(cmd, cwd=str(HERE),
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f'   не удалось запустить: {e}')
            return 1
        if not wait_for(llama_up, args.wait, 'модель'):
            print('   не дождался. Попробуй --wait больше или другую модель.')
            return 1

    print(f'=== 2. Kira bridge (:{BRIDGE_PORT}) ===')
    if bridge_up():
        print('   уже работает')
    else:
        subprocess.Popen(
            [PY, str(BRIDGE), '--port', str(BRIDGE_PORT)],
            cwd=str(HERE),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if not wait_for(bridge_up, 20, 'мост'):
            print('   мост не поднялся')
            return 1

    if args.no_cube:
        print('\nГотово. Куб запускай так:')
        print(f'  set CUBE_AI_BASE_URL=http://127.0.0.1:{BRIDGE_PORT}/v1')
        print(f'  set CUBE_AI_MODEL={DEFAULT_MODEL}')
        print('  python main.py')
        print('\nПроверить цепочку: python kira_selftest.py')
        return 0

    print('=== 3. Куб ===')
    env = dict(__import__('os').environ)
    env['CUBE_AI_BASE_URL'] = f'http://127.0.0.1:{BRIDGE_PORT}/v1'
    env['CUBE_AI_MODEL'] = DEFAULT_MODEL
    print(f'   CUBE_AI_BASE_URL={env["CUBE_AI_BASE_URL"]}')
    print('   (Ctrl+C, чтобы выйти)')
    return subprocess.run([PY, 'main.py'], cwd=str(HERE), env=env).returncode


if __name__ == '__main__':
    sys.exit(main())
