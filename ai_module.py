#!/usr/bin/env python3
"""
ai_module.py — AI-ядро Hermes Cube.

Содержание:
  - LM Studio клиент (gemma-4-e4b-it)
  - Анализ настроения (mood detection)
  - TextOverlay — парящие буквы над кубом
  - Окно ввода (C / контекстное меню)
  - История сообщений

Интеграция: CubeApp создаёт AiCore, вызывает show_input() / spawn_text().
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import colorsys
import json
import threading
import tkinter as tk
import math
import random
import time
import urllib.request
import urllib.error

# ---------------------------------------------------------------------------
# LM Studio config
# ---------------------------------------------------------------------------

LM_STUDIO_URL: str = "http://127.0.0.1:1234"
AI_MODEL: str = "gemma-4-e4b-it"

AI_MOODS: Dict[str, Dict[str, float]] = {
    'idle':     {'pulse_rate': 1.8, 'pulse_amp': 0.12, 'speed': 0.28, 'color_shift': 0.0},
    'thinking': {'pulse_rate': 3.5, 'pulse_amp': 0.25, 'speed': 0.50, 'color_shift': 0.15},
    'speaking': {'pulse_rate': 2.5, 'pulse_amp': 0.18, 'speed': 0.40, 'color_shift': 0.08},
    'happy':    {'pulse_rate': 2.8, 'pulse_amp': 0.22, 'speed': 0.50, 'color_shift': 0.12},
    'sad':      {'pulse_rate': 0.8, 'pulse_amp': 0.05, 'speed': 0.10, 'color_shift': 0.55},
}


def ai_chat(
    message: str,
    history: Optional[List[Dict[str, str]]] = None,
    timeout: int = 15,
) -> str:
    """Send a message to LM Studio and return raw response text.

    The system prompt instructs the model to reply with a JSON object:
       {"mood": "happy|sad|thinking|speaking|idle",
        "text": "...",
        "color_hue": 0.0-1.0}

    history: list of {"role": "user"|"assistant", "text": ...} to include as
             conversation context (last 10 messages kept).
    """
    # Build message list
    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "Ты — живой аватар по имени Hermes Cube. "
                "Твой дом — сфера из частиц. "
                "Отвечай кратко, эмоционально, не больше 2-3 предложений. "
                "Ты чувствуешь пульсацию, вращение, цвета вокруг себя.\n\n"
                "ВСЕГДА отвечай ТОЛЬКО валидным JSON-объектом в ОДНУ строку "
                "без пояснений и markdown, строго в формате:\n"
                '{"mood": "happy|sad|thinking|speaking|idle", '
                '"text": "...", '
                '"color_hue": 0.0-1.0}\n'
                "Поле mood отражает твоё настроение. "
                "Поле text — твой ответ пользователю. "
                "Поле color_hue — оттенок цвета (0.0-1.0), "
                "соответствующий твоему настроению."
            ),
        },
    ]
    if history:
        # Take last 10 messages
        for h in history[-10:]:
            role: str = "assistant" if h["role"] == "ai" else h["role"]
            messages.append({"role": role, "content": h["text"]})
    messages.append({"role": "user", "content": message})

    data: bytes = json.dumps({
        "model": AI_MODEL,
        "messages": messages,
        "temperature": 0.8,
        "max_tokens": 180,
        "stream": False,
    }).encode()
    try:
        req = urllib.request.Request(
            f"{LM_STUDIO_URL}/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result: Dict[str, Any] = json.loads(resp.read())
            return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return json.dumps({"mood": "idle", "text": f"[Ошибка: {e}]", "color_hue": 0.0})


def analyze_mood(text: str) -> str:
    """Detect mood from AI response text via keywords (fallback)."""
    t: str = text.lower()
    if any(w in t for w in ['груст', 'печал', 'устал', 'тоск', 'один']):
        return 'sad'
    if any(w in t for w in ['рад', 'счаст', 'весел', 'крут', 'класс', 'любл']):
        return 'happy'
    if any(w in t for w in ['дума', 'размыш', 'представ', 'может']):
        return 'thinking'
    return 'speaking'


def parse_ai_response(raw: str) -> Dict[str, Any]:
    """Parse JSON response from LM Studio into structured dict.

    Expected format:
        {"mood": "happy|sad|thinking|speaking|idle",
         "text": "...",
         "color_hue": 0.0-1.0}

    If JSON parsing fails, falls back to keyword-based mood detection
    and returns raw text with hue=0.0.
    """
    # Try to extract JSON from the response (handle model wrapping it in backticks)
    cleaned: str = raw.strip()
    if cleaned.startswith('```'):
        # Remove markdown code fences
        cleaned = cleaned.strip('` \n')
        if cleaned.startswith('json'):
            cleaned = cleaned[4:].strip()
    try:
        parsed: Dict[str, Any] = json.loads(cleaned)
        mood: str = str(parsed.get('mood', 'speaking'))
        text: str = str(parsed.get('text', raw))
        color_hue: float = float(parsed.get('color_hue', 0.0))
        # Validate mood
        if mood not in ('happy', 'sad', 'thinking', 'speaking', 'idle'):
            mood = analyze_mood(text)
        # Clamp color_hue
        color_hue = max(0.0, min(1.0, color_hue))
        return {'mood': mood, 'text': text, 'color_hue': color_hue}
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Fallback: keyword-based mood + raw text
    mood = analyze_mood(raw)
    return {'mood': mood, 'text': raw, 'color_hue': AI_MOODS.get(mood, AI_MOODS['idle'])['color_shift']}


def apply_hsv_shift(
    r: float, g: float, b: float,
    hue_shift: float,
) -> Tuple[float, float, float]:
    """Apply a hue rotation to an RGB colour using HSV colour space.

    hue_shift: rotation amount in normalized 0.0-1.0 range (maps to 0-360°).
    0.0 = no change, 0.5 = 180° rotation, etc.
    Returns shifted (r, g, b) each in 0-255 range.
    """
    h: float
    s: float
    v: float
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    h = (h + hue_shift) % 1.0
    r2, g2, b2 = colorsys.hsv_to_rgb(h, s, v)
    return (r2 * 255.0, g2 * 255.0, b2 * 255.0)


# ---------------------------------------------------------------------------
# TextOverlay — flying letters
# ---------------------------------------------------------------------------

class TextOverlay:
    """
    Прозрачное окно поверх куба для парящих букв.
    Каждая буква — canvas текст, летит от центра куба в случайную сторону.
    """

    BG_COLOR: str = '#000001'
    FONT: Tuple[str, int, str] = ('Segoe UI', 18, 'bold')
    LETTER_LIFE: float = 2.5  # seconds
    LETTER_SPEED: float = 80.0  # pixels/sec
    MAX_PARTICLES: int = 60

    def __init__(self, root: tk.Tk) -> None:
        self.root: tk.Tk = root
        self.window: Optional[tk.Toplevel] = None
        self.canvas: Optional[tk.Canvas] = None
        self.particles: List[Dict[str, Any]] = []
        self._queue: str = ''  # next text to spawn
        self._running: bool = False
        self._setup()

    def _setup(self) -> None:
        """Create the transparent overlay window."""
        self.window = tk.Toplevel(self.root)
        self.window.title('♢ Hermes Text')
        sw: int = self.root.winfo_screenwidth()
        sh: int = self.root.winfo_screenheight()
        self.window.geometry(f'{sw}x{sh}+0+0')
        self.window.configure(bg=self.BG_COLOR)
        self.window.attributes('-transparentcolor', self.BG_COLOR)
        self.window.overrideredirect(True)
        self.window.withdraw()  # hidden by default

        self.canvas = tk.Canvas(
            self.window, bg=self.BG_COLOR, highlightthickness=0,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

    def show(self) -> None:
        """Reveal the overlay window and start animation."""
        if self.window:
            self.window.deiconify()
            self.window.lift()
        self._running = True

    def hide(self) -> None:
        """Hide overlay and clear particles."""
        self._running = False
        if self.window:
            self.window.withdraw()
        self.particles.clear()

    def spawn_text(self, text: str, from_x: int, from_y: int) -> None:
        """Queue text to burst from a position (cube center)."""
        if not self._running or not self.canvas:
            return
        if self.window and self.window.state() == 'withdrawn':
            self.show()
        for ch in text:
            if ch.strip() or ch in '.,!?—…':
                angle: float = random.uniform(0, 2 * math.pi)
                speed: float = self.LETTER_SPEED * random.uniform(0.7, 1.3)
                r: int = random.randint(200, 255)
                g: int = random.randint(150, 255)
                b: int = random.randint(100, 255)
                colour: str = f'#{r:02x}{g:02x}{b:02x}'
                item = self.canvas.create_text(
                    from_x, from_y, text=ch,
                    fill=colour, font=self.FONT,
                )
                self.particles.append({
                    'item': item,
                    'x': float(from_x),
                    'y': float(from_y),
                    'vx': math.cos(angle) * speed,
                    'vy': math.sin(angle) * speed,
                    'life': self.LETTER_LIFE,
                    'max_life': self.LETTER_LIFE,
                    'r': r, 'g': g, 'b': b,
                })

    def update(self, dt: float) -> None:
        """Animate existing particles and spawn queued ones."""
        if not self._running or not self.canvas:
            return

        # Clean dead particles
        alive: List[Dict[str, Any]] = []
        for p in self.particles:
            p['life'] -= dt
            if p['life'] <= 0:
                try:
                    self.canvas.delete(p['item'])
                except Exception:
                    pass
                continue
            p['x'] += p['vx'] * dt
            p['y'] += p['vy'] * dt
            p['vy'] += 40.0 * dt  # gravity
            fade: float = max(0.0, p['life'] / p['max_life'])
            alpha: int = int(255 * fade)
            colour: str = f'#{int(p["r"]):02x}{int(p["g"]):02x}{int(p["b"]):02x}'
            try:
                self.canvas.coords(p['item'], int(p['x']), int(p['y']))
                self.canvas.itemconfig(p['item'], fill=colour)
            except Exception:
                pass
            alive.append(p)

        self.particles = alive[-self.MAX_PARTICLES:]

        # Auto-hide when empty
        if not self.particles and self.window and self.window.state() != 'withdrawn':
            self.hide()

    def close(self) -> None:
        """Destroy all resources."""
        self._running = False
        self.particles.clear()
        if self.window:
            try:
                self.window.destroy()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# InputWindow — text input at screen bottom
# ---------------------------------------------------------------------------

class InputWindow:
    """
    Компактное окно ввода внизу экрана.
    Открывается по C / контекстному меню.
    При Enter отправляет сообщение в AI и возвращает текст через callback.
    """

    WIDTH: int = 440
    HEIGHT: int = 48

    def __init__(self, root: tk.Tk,
                 on_submit: Optional[callable] = None,
                 on_close: Optional[callable] = None) -> None:
        self.root: tk.Tk = root
        self.on_submit: Optional[callable] = on_submit
        self.on_close: Optional[callable] = on_close
        self.window: Optional[tk.Toplevel] = None
        self._entry: Optional[tk.Entry] = None
        self._var: tk.StringVar = tk.StringVar()

    def show(self) -> None:
        """Create and display the input window."""
        self.hide()
        self.window = tk.Toplevel(self.root)
        self.window.title('♢ Hermes Cube — Ввод')
        sw: int = self.root.winfo_screenwidth()
        sh: int = self.root.winfo_screenheight()
        ix: int = (sw - self.WIDTH) // 2
        iy: int = sh - self.HEIGHT - 50
        self.window.geometry(f'{self.WIDTH}x{self.HEIGHT}+{ix}+{iy}')
        self.window.configure(bg='#0d0d1a')
        self.window.attributes('-topmost', True)
        self.window.resizable(False, False)
        self.window.overrideredirect(True)

        frame = tk.Frame(self.window, bg='#0d0d1a',
                         highlightbackground='#e94560',
                         highlightthickness=1, bd=0)
        frame.pack(fill='both', expand=True)

        self._var.set('')
        self._entry = tk.Entry(
            frame, textvariable=self._var,
            bg='#0d0d1a', fg='#e0e0e0', relief=tk.FLAT,
            font=('Segoe UI', 14), insertbackground='#e94560',
            highlightthickness=0, bd=4,
        )
        self._entry.pack(fill='both', expand=True, padx=6, pady=4)
        self._entry.focus_set()
        self._entry.bind('<Return>', self._on_enter)
        self._entry.bind('<Escape>', lambda e: self.hide())

    def hide(self) -> None:
        """Close the input window."""
        if self.window:
            try:
                self.window.destroy()
            except Exception:
                pass
            self.window = None
        if self.on_close:
            self.on_close()

    def _on_enter(self, _event: Any = None) -> None:
        text: str = self._var.get().strip()
        if not text:
            return
        self.hide()
        if self.on_submit:
            self.on_submit(text)


# ---------------------------------------------------------------------------
# AiCore — объединяет всё
# ---------------------------------------------------------------------------

class AiCore:
    """
    Единая точка входа для AI-функций куба.

    CubeApp создаёт один AiCore и вызывает:
      - toggle_input() — открыть/закрыть окно ввода
      - update(dt) — обновить парящие буквы
      - get_mood_override() — dict для наложения на конфиг
      - spawn_response(text, cx, cy) — выпустить ответ куба как буквы
      - close()
    """

    def __init__(self, root: tk.Tk) -> None:
        self.root: tk.Tk = root
        self.mood: str = 'idle'
        self.last_response: str = ''
        self.color_hue: float = 0.0
        self.history: List[Dict[str, str]] = []  # [{"role": ..., "text": ...}]
        self._thinking: bool = False

        self.text_overlay: TextOverlay = TextOverlay(root)
        self.input_win: InputWindow = InputWindow(
            root,
            on_submit=self._on_input_submit,
            on_close=self._on_input_close,
        )

    def toggle_input(self) -> None:
        """Open/close the input window."""
        if self.input_win.window and self.input_win.window.winfo_exists():
            self.input_win.hide()
        else:
            self.input_win.show()

    def _on_input_submit(self, text: str) -> None:
        """Handle submitted text: chat with AI, spawn response, update mood."""
        self.history.append({"role": "user", "text": text})
        self._thinking = True

        def do_ai() -> None:
            response: str = ai_chat(text)
            self.last_response = response
            self.mood = analyze_mood(response)
            self.history.append({"role": "ai", "text": response})
            self._thinking = False

        threading.Thread(target=do_ai, daemon=True).start()

    def _on_input_close(self) -> None:
        """Input window closed."""
        pass

    def get_mood_override(self) -> Dict[str, float]:
        """
        Return mood params to temporarily override cube config.
        Returns empty dict if idle (no override).
        """
        if self.mood != 'idle' and self.mood in AI_MOODS:
            return dict(AI_MOODS[self.mood])
        return {}

    def get_color_shift(self) -> float:
        """Return current color shift from mood (0.0 = none)."""
        mood_data = AI_MOODS.get(self.mood, AI_MOODS['idle'])
        return mood_data['color_shift'] if self.mood != 'idle' else 0.0

    def update(self, dt: float) -> None:
        """Called each frame. Updates text particles."""
        self.text_overlay.update(dt)

        # If AI finished thinking, spawn letters from cube position
        if (not self._thinking
                and self.last_response
                and self.text_overlay.window
                and self.text_overlay.window.state() == 'withdrawn'):
            # Will be spawned on next spawn_response call
            pass

    def spawn_response(self, text: str, from_x: int, from_y: int) -> None:
        """Burst AI response as flying letters from (x, y)."""
        self.text_overlay.spawn_text(text, from_x, from_y)

    def close(self) -> None:
        """Clean up AI resources."""
        self.text_overlay.close()
        self.input_win.hide()
