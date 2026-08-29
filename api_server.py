"""api_server.py — HTTP API для управления кубом.

Даёт Hermes агенту управлять кубом через HTTP запросы (localhost:8081).

Команды:
  POST /api/config  {"key": "particle_density", "value": 12}
  POST /api/mood    {"mood": "happy"}
  POST /api/speak   {"text": "Привет мир!"}
  GET  /api/status
"""

from __future__ import annotations

import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Callable, Dict, Optional


class CubeAPIHandler(BaseHTTPRequestHandler):
    """Обработчик HTTP запросов к кубу."""

    # Ссылка на engine (ставится извне)
    engine: Any = None

    def do_POST(self) -> None:
        content_len = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_len) if content_len else b'{}'

        # Битые байты — честная 400, а не молчаливый мусор из errors='replace'
        try:
            text_body = body.decode('utf-8') if isinstance(body, bytes) else body
        except UnicodeDecodeError as e:
            self._send(400, {
                "error": "body must be valid UTF-8",
                "detail": str(e),
                "hint": "curl: используйте --data-binary @file.json в UTF-8",
            })
            return

        try:
            data = json.loads(text_body)
        except (json.JSONDecodeError, ValueError):
            self._send(400, {"error": "invalid JSON"})
            return

        if self.path == '/api/config':
            self._handle_config(data)
        elif self.path == '/api/mood':
            self._handle_mood(data)
        elif self.path == '/api/speak':
            self._handle_speak(data)
        elif self.path == '/api/text_settings':
            self._handle_text_settings(data)
        elif self.path == '/api/quit':
            self._handle_quit()
        else:
            self._send(404, {"error": "not found"})

    def do_GET(self) -> None:
        if self.path == '/api/status':
            self._handle_status()
        else:
            self._send(404, {"error": "not found"})

    # Ключи, при росте которых куб может не поместиться в окно
    RESIZE_KEYS = ('cube_scale', 'wave_amp', 'pulse_amplitude', 'cell_size')

    def _handle_config(self, data: Dict[str, Any]) -> None:
        key = data.get('key', '')
        value = data.get('value')
        if not key:
            self._send(400, {"error": "key required"})
            return
        if self.engine and hasattr(self.engine, 'config'):
            self.engine.config[key] = value
            if key in ('particle_density', 'cell_size'):
                self.engine.recalc(self.engine.config)
            resized = False
            if key in self.RESIZE_KEYS:
                # Tk не потокобезопасен — планируем в его цикле
                try:
                    self.engine._tk_root.after(
                        0, self.engine._auto_resize_window)
                    resized = True
                except Exception:
                    pass
            self._send(200, {
                "ok": True, "key": key, "value": value,
                "resize_scheduled": resized,
            })
        else:
            self._send(503, {"error": "engine not ready"})

    def _handle_mood(self, data: Dict[str, Any]) -> None:
        mood = data.get('mood', 'idle')
        if self.engine and hasattr(self.engine, 'world'):
            world = self.engine.world
            world.meta.mood = mood
            # lock=False снимает фиксацию и отдаёт настроение обратно AI
            world.meta.mood_lock = bool(data.get('lock', True))
            # Применяем параметры настроения сразу (пульс/вращение)
            try:
                from core.ai_constants import AI_MOODS
                params = AI_MOODS.get(mood)
                if params and world.meta.config is not None:
                    for key in ('pulse_rate', 'pulse_amplitude', 'rotation_speed'):
                        if key in params:
                            world.meta.config[key] = params[key]
            except Exception:
                pass
            self._send(200, {
                "ok": True,
                "mood": mood,
                "lock": world.meta.mood_lock,
            })
        else:
            self._send(503, {"error": "engine not ready"})

    # Настройки вывода текста — длина, размер, тип отображения, тайминги
    TEXT_KEYS: Dict[str, type] = {
        'text_font_size': int,
        'text_hold_time': float,
        'text_morph_time': float,
        'text_char_mode': str,
        'text_style': str,
        'text_max_chars': int,
        'text_fill': float,
        'text_freeze_spin': bool,
        'text_auto_scale': bool,
    }
    CHAR_MODES = ('symbols', 'dots', 'glow', 'words')

    def _handle_text_settings(self, data: Dict[str, Any]) -> None:
        if not (self.engine and hasattr(self.engine, 'config')):
            self._send(503, {"error": "engine not ready"})
            return

        applied: Dict[str, Any] = {}
        errors: Dict[str, str] = {}

        for key, raw in data.items():
            caster = self.TEXT_KEYS.get(key)
            if caster is None:
                errors[key] = f"unknown key (allowed: {sorted(self.TEXT_KEYS)})"
                continue
            try:
                value = bool(raw) if caster is bool else caster(raw)
            except (TypeError, ValueError):
                errors[key] = f"expected {caster.__name__}"
                continue

            if key == 'text_char_mode' and value not in self.CHAR_MODES:
                errors[key] = f"expected one of {list(self.CHAR_MODES)}"
                continue
            if key == 'text_style':
                from core.text_layout import TEXT_STYLES
                if value not in TEXT_STYLES:
                    errors[key] = f"expected one of {list(TEXT_STYLES)}"
                    continue
            if key == 'text_font_size' and not 6 <= value <= 200:
                errors[key] = "expected 6..200"
                continue
            if key == 'text_max_chars' and not 0 <= value <= 500:
                errors[key] = "expected 0..500 (0 = без обрезки)"
                continue
            if key == 'text_fill' and not 0.1 <= value <= 1.0:
                errors[key] = "expected 0.1..1.0"
                continue
            if key in ('text_hold_time', 'text_morph_time') and not 0.05 <= value <= 60.0:
                errors[key] = "expected 0.05..60.0"
                continue

            self.engine.config[key] = value
            applied[key] = value

        # Стиль применяется и к надписи, которая уже на экране, а не только
        # к следующей — иначе переключатель выглядит мёртвым.
        if 'text_style' in applied and hasattr(self.engine, 'set_text_style'):
            try:
                self.engine.set_text_style(applied['text_style'])
            except Exception as e:
                errors['text_style'] = f"в конфиг записан, на кадр не попал: {e}"

        code = 200 if applied and not errors else (400 if errors and not applied else 200)
        self._send(code, {"ok": not errors, "applied": applied, "errors": errors})

    def _handle_speak(self, data: Dict[str, Any]) -> None:
        text = data.get('text', '')
        if not text:
            self._send(400, {"error": "text required"})
            return
        # Используем speak_buffer — напрямую в avatar_text (минуя AI)
        if self.engine and hasattr(self.engine, 'world'):
            self.engine.world.meta.speak_buffer = text
            self._send(200, {"ok": True, "text": text})
        else:
            self._send(503, {"error": "engine not ready"})

    def _handle_quit(self) -> None:
        self._send(200, {"ok": True, "message": "quitting"})
        if self.engine:
            threading.Thread(target=self.engine.quit_app, daemon=True).start()

    def _handle_status(self) -> None:
        if self.engine and hasattr(self.engine, 'world'):
            w = self.engine.world
            cfg = w.meta.config or {}
            return self._send(200, {
                "particles": w.sim.active_count,
                "shape": cfg.get('shape_preset', 'cube'),
                "mood": w.meta.mood,
                # monitor хранит FPS в .fps (не last_fps — там был вечный 0)
                "fps": round(float(getattr(self.engine.monitor, 'fps', 0.0)), 1),
                "text_mode": w.meta.text_mode,
                "text_freeze": round(float(getattr(w.meta, 'text_freeze', 0.0)), 3),
                "mood_lock": bool(getattr(w.meta, 'mood_lock', False)),
                "window": [w.meta.w, w.meta.h],
                "cube_scale": cfg.get('cube_scale'),
                "anim": {
                    k: cfg.get(k) for k in (
                        'rotation_speed', 'pulse_rate', 'pulse_amplitude',
                        'morph_progress', 'char_mode', 'particle_mode',
                    )
                },
                "text_settings": {
                    k: cfg.get(k) for k in (
                        'text_font_size', 'text_hold_time', 'text_morph_time',
                        'text_char_mode', 'text_max_chars', 'text_fill',
                        'text_freeze_spin', 'text_auto_scale',
                    )
                },
            })
        self._send(503, {"error": "engine not ready"})

    def _send(self, code: int, data: Dict) -> None:
        payload = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt: str, *args: Any) -> None:
        pass  # тихо


def start_api_server(engine: Any, port: int = 8081) -> None:
    """Запустить HTTP API в фоновом потоке."""
    CubeAPIHandler.engine = engine
    server = HTTPServer(('127.0.0.1', port), CubeAPIHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"[API] Server on http://127.0.0.1:{port}", flush=True)
