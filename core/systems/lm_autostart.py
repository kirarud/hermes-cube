"""systems/lm_autostart.py — Автоматический запуск LM Studio.

Читает:
  meta.ai_requested — True когда нужен AI

Пишет:
  meta.ai_ready — True когда LM Studio готова

Не вызывает AI-чат — только управляет процессом LM Studio.
"""

from __future__ import annotations

import os
import subprocess
import time
import urllib.request
import urllib.error
import json
from typing import Any, Dict, List, Optional

from core.world import World

from core.ai_constants import LM_STUDIO_URL, AI_MODEL, AI_MODEL_ID

LM_STUDIO_PATH: str = os.path.join(
    os.environ.get('LOCALAPPDATA', 'C:\\Users\\kirarud\\AppData\\Local'),
    'Programs\\LM Studio\\LM Studio.exe',
)

_LM_PROCESS: Optional[subprocess.Popen] = None


class LMAutoStartSystem:
    """Система, запускающая LM Studio и загружающая модель по необходимости.

    Однократная: после успешного запуска отключается.
    """

    def __init__(self) -> None:
        self._started: bool = False
        self._attempted: bool = False

    def update(self, world: World, dt: float) -> None:
        if self._started:
            return
        if not world.meta.ai_requested:
            return
        if self._attempted:
            # Уже пробовали — не повторяем каждый кадр
            return

        self._attempted = True

        # Проверить, не запущен ли уже
        if self._check_api():
            self._started = True
            world.meta.ai_ready = True
            return

        # Запустить процесс
        if not os.path.isfile(LM_STUDIO_PATH):
            print("[LMAutoStart] LM Studio not found", flush=True)
            return

        global _LM_PROCESS
        try:
            _LM_PROCESS = subprocess.Popen(
                [LM_STUDIO_PATH],
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        except Exception as e:
            print(f"[LMAutoStart] Failed: {e}", flush=True)
            return

        # Ждать API
        for _ in range(60):
            if self._check_api():
                break
            time.sleep(1.0)
        else:
            print("[LMAutoStart] API timeout", flush=True)
            return

        # Загрузить модель
        self._load_model()
        for _ in range(30):
            if self._check_model():
                self._started = True
                world.meta.ai_ready = True
                print("[LMAutoStart] Model ready", flush=True)
                return
            time.sleep(1.0)

        print("[LMAutoStart] Model load timeout", flush=True)

    @staticmethod
    def _check_api() -> bool:
        try:
            req = urllib.request.Request(f"{LM_STUDIO_URL}/v1/models", method='GET')
            with urllib.request.urlopen(req, timeout=2) as resp:
                return resp.status == 200
        except Exception:
            return False

    @staticmethod
    def _load_model() -> None:
        try:
            data = json.dumps({"model": AI_MODEL_ID}).encode()
            req = urllib.request.Request(
                f"{LM_STUDIO_URL}/v1/models/load",
                data=data,
                headers={"Content-Type": "application/json"},
                method='POST',
            )
            with urllib.request.urlopen(req, timeout=10):
                pass
        except urllib.error.HTTPError as e:
            if e.code != 404:
                print(f"[LMAutoStart] Load model HTTP {e.code}", flush=True)
        except Exception as e:
            print(f"[LMAutoStart] Load model error: {e}", flush=True)

    @staticmethod
    def _check_model() -> bool:
        try:
            req = urllib.request.Request(f"{LM_STUDIO_URL}/v1/models")
            with urllib.request.urlopen(req, timeout=2) as resp:
                data = json.loads(resp.read())
                for m in data.get('data', []):
                    mid = m.get('id', '') or m.get('name', '')
                    if AI_MODEL in mid or AI_MODEL_ID in mid:
                        return True
        except Exception:
            pass
        return False
