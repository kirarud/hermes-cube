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

        try:
            data = json.loads(body if isinstance(body, str) else body.decode('utf-8', errors='replace'))
        except (json.JSONDecodeError, ValueError, UnicodeDecodeError):
            self._send(400, {"error": "invalid JSON"})
            return

        if self.path == '/api/config':
            self._handle_config(data)
        elif self.path == '/api/mood':
            self._handle_mood(data)
        elif self.path == '/api/speak':
            self._handle_speak(data)
        elif self.path == '/api/quit':
            self._handle_quit()
        else:
            self._send(404, {"error": "not found"})

    def do_GET(self) -> None:
        if self.path == '/api/status':
            self._handle_status()
        else:
            self._send(404, {"error": "not found"})

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
            self._send(200, {"ok": True, "key": key, "value": value})
        else:
            self._send(503, {"error": "engine not ready"})

    def _handle_mood(self, data: Dict[str, Any]) -> None:
        mood = data.get('mood', 'idle')
        if self.engine and hasattr(self.engine, 'world'):
            self.engine.world.meta.mood = mood
            self._send(200, {"ok": True, "mood": mood})
        else:
            self._send(503, {"error": "engine not ready"})

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
            self._send(200, {
                "particles": w.sim.active_count,
                "shape": w.meta.config.get('shape_preset', 'cube'),
                "mood": w.meta.mood,
                "fps": getattr(self.engine.monitor, 'last_fps', 0),
                "text_mode": w.meta.text_mode,
            })
        else:
            self._send(503, {"error": "engine not ready"})

    def _send(self, code: int, data: Dict) -> None:
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, fmt: str, *args: Any) -> None:
        pass  # тихо


def start_api_server(engine: Any, port: int = 8081) -> None:
    """Запустить HTTP API в фоновом потоке."""
    CubeAPIHandler.engine = engine
    server = HTTPServer(('127.0.0.1', port), CubeAPIHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"[API] Server on http://127.0.0.1:{port}", flush=True)
