#!/usr/bin/env python3
"""
PixelGrid — Full-screen pixel framebuffer for Hermes Cube agents.

Architecture:
  - PixelGrid: numpy (H, W, 3) RGB buffer with paint primitives + hit_zones
  - PixelGridWindow: full-screen transparent Toplevel that renders buffer + handles clicks
  - Font 5×7: ASCII glyphs as bitmask matrices for pixel-perfect text

Integration:
  CubeApp spawns PixelGridWindow as a third overlay (on top of cube).
  Agents draw into the buffer, register click zones, and get mouse events.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

import threading

try:
    from PIL import Image, ImageTk
except ImportError:
    Image = None  # type: ignore[assignment]
    ImageTk = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Must match TRANSPARENT_COLOR in cube_app.py (#000001)
TRANSPARENT_RGB: Tuple[int, int, int] = (0, 0, 1)

# ---------------------------------------------------------------------------
# 5×7 Pixel font — bit masks for ASCII printable chars
# Each char is a list of 7 rows, each row is a 5-bit int (LSB = leftmost pixel)
# 0 = transparent, 1 = foreground colour
# ---------------------------------------------------------------------------

# fmt: off
_FONT_5x7: Dict[str, List[int]] = {
    # UPPERCASE
    'A': [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
    'B': [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110],
    'C': [0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110],
    'D': [0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110],
    'E': [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
    'F': [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000],
    'G': [0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01110],
    'H': [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
    'I': [0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
    'J': [0b00111, 0b00010, 0b00010, 0b00010, 0b00010, 0b10010, 0b01100],
    'K': [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001],
    'L': [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
    'M': [0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001],
    'N': [0b10001, 0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001],
    'O': [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
    'P': [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000],
    'Q': [0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101],
    'R': [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
    'S': [0b01110, 0b10001, 0b10000, 0b01110, 0b00001, 0b10001, 0b01110],
    'T': [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
    'U': [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
    'V': [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100],
    'W': [0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001],
    'X': [0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001],
    'Y': [0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100],
    'Z': [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111],
    # DIGITS
    '0': [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
    '1': [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
    '2': [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111],
    '3': [0b01110, 0b10001, 0b00001, 0b00110, 0b00001, 0b10001, 0b01110],
    '4': [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
    '5': [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
    '6': [0b01110, 0b10000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110],
    '7': [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
    '8': [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
    '9': [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00001, 0b01110],
    # LOWERCASE
    'a': [0b00000, 0b00000, 0b01110, 0b00001, 0b01111, 0b10001, 0b01111],
    'b': [0b10000, 0b10000, 0b11110, 0b10001, 0b10001, 0b10001, 0b11110],
    'c': [0b00000, 0b00000, 0b01110, 0b10001, 0b10000, 0b10001, 0b01110],
    'd': [0b00001, 0b00001, 0b01111, 0b10001, 0b10001, 0b10001, 0b01111],
    'e': [0b00000, 0b00000, 0b01110, 0b10001, 0b11111, 0b10000, 0b01110],
    'f': [0b00110, 0b01001, 0b01000, 0b11100, 0b01000, 0b01000, 0b01000],
    'g': [0b00000, 0b00000, 0b01111, 0b10001, 0b10001, 0b01111, 0b00001],
    'h': [0b10000, 0b10000, 0b11110, 0b10001, 0b10001, 0b10001, 0b10001],
    'i': [0b00100, 0b00000, 0b01100, 0b00100, 0b00100, 0b00100, 0b01110],
    'j': [0b00010, 0b00000, 0b00110, 0b00010, 0b00010, 0b10010, 0b01100],
    'k': [0b10000, 0b10000, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010],
    'l': [0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
    'm': [0b00000, 0b00000, 0b11010, 0b10101, 0b10101, 0b10001, 0b10001],
    'n': [0b00000, 0b00000, 0b11110, 0b10001, 0b10001, 0b10001, 0b10001],
    'o': [0b00000, 0b00000, 0b01110, 0b10001, 0b10001, 0b10001, 0b01110],
    'p': [0b00000, 0b00000, 0b11110, 0b10001, 0b10001, 0b11110, 0b10000],
    'q': [0b00000, 0b00000, 0b01111, 0b10001, 0b10001, 0b01111, 0b00001],
    'r': [0b00000, 0b00000, 0b10110, 0b11001, 0b10000, 0b10000, 0b10000],
    's': [0b00000, 0b00000, 0b01110, 0b10000, 0b01110, 0b00001, 0b11110],
    't': [0b01000, 0b01000, 0b11100, 0b01000, 0b01000, 0b01001, 0b00110],
    'u': [0b00000, 0b00000, 0b10001, 0b10001, 0b10001, 0b10001, 0b01111],
    'v': [0b00000, 0b00000, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100],
    'w': [0b00000, 0b00000, 0b10001, 0b10001, 0b10101, 0b10101, 0b01010],
    'x': [0b00000, 0b00000, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001],
    'y': [0b00000, 0b00000, 0b10001, 0b10001, 0b10001, 0b01111, 0b00001],
    'z': [0b00000, 0b00000, 0b11111, 0b00010, 0b00100, 0b01000, 0b11111],
    # PUNCTUATION & SYMBOLS
    ' ': [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000],
    '.': [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00100],
    ',': [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00100, 0b01000],
    '!': [0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00000, 0b00100],
    '?': [0b01110, 0b10001, 0b00001, 0b00110, 0b00100, 0b00000, 0b00100],
    ':': [0b00000, 0b00100, 0b00000, 0b00000, 0b00000, 0b00100, 0b00000],
    ';': [0b00000, 0b00100, 0b00000, 0b00000, 0b00000, 0b00100, 0b01000],
    '-': [0b00000, 0b00000, 0b00000, 0b01110, 0b00000, 0b00000, 0b00000],
    '_': [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b11111],
    '+': [0b00000, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0b00000],
    '=': [0b00000, 0b00000, 0b11111, 0b00000, 0b11111, 0b00000, 0b00000],
    '/': [0b00001, 0b00010, 0b00010, 0b00100, 0b01000, 0b01000, 0b10000],
    '\\': [0b10000, 0b01000, 0b01000, 0b00100, 0b00010, 0b00010, 0b00001],
    '|': [0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
    '(': [0b00010, 0b00100, 0b01000, 0b01000, 0b01000, 0b00100, 0b00010],
    ')': [0b01000, 0b00100, 0b00010, 0b00010, 0b00010, 0b00100, 0b01000],
    '[': [0b01110, 0b01000, 0b01000, 0b01000, 0b01000, 0b01000, 0b01110],
    ']': [0b01110, 0b00010, 0b00010, 0b00010, 0b00010, 0b00010, 0b01110],
    '{': [0b00010, 0b00100, 0b00100, 0b01000, 0b00100, 0b00100, 0b00010],
    '}': [0b01000, 0b00100, 0b00100, 0b00010, 0b00100, 0b00100, 0b01000],
    '<': [0b00000, 0b00010, 0b00100, 0b01000, 0b00100, 0b00010, 0b00000],
    '>': [0b00000, 0b01000, 0b00100, 0b00010, 0b00100, 0b01000, 0b00000],
    '#': [0b01010, 0b01010, 0b11111, 0b01010, 0b11111, 0b01010, 0b01010],
    '%': [0b10001, 0b10010, 0b00100, 0b00100, 0b01000, 0b10001, 0b00001],
    '&': [0b01100, 0b10010, 0b10100, 0b01100, 0b10101, 0b10010, 0b01101],
    '*': [0b00000, 0b00100, 0b10101, 0b01110, 0b10101, 0b00100, 0b00000],
    '@': [0b01110, 0b10001, 0b10111, 0b10101, 0b10111, 0b10000, 0b01110],
    '\'': [0b00100, 0b01000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000],
    '"': [0b01010, 0b10100, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000],
    '`': [0b01000, 0b00100, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000],
    '~': [0b00000, 0b00000, 0b01000, 0b10101, 0b00010, 0b00000, 0b00000],
    '^': [0b00100, 0b01010, 0b10001, 0b00000, 0b00000, 0b00000, 0b00000],
    '♥': [0b01010, 0b11111, 0b11111, 0b01110, 0b00100, 0b00000, 0b00000],
    '♢': [0b00100, 0b01010, 0b10001, 0b01010, 0b00100, 0b00000, 0b00000],
    '●': [0b00000, 0b01110, 0b11111, 0b11111, 0b01110, 0b00000, 0b00000],
}
# fmt: on

CHAR_W: int = 5  # glyph width
CHAR_H: int = 7  # glyph height
CHAR_GAP_X: int = 1  # horizontal gap between chars
CHAR_GAP_Y: int = 1  # vertical gap between lines


# ---------------------------------------------------------------------------
# Hit zone helpers
# ---------------------------------------------------------------------------

HitCallback = Callable[[int, int, str], None]
"""Callback signature: (mouse_x, mouse_y, action) where action is 'click' | 'drag' | 'release'."""


def _pixel_to_rgb(pixel: Any) -> Tuple[int, int, int]:
    """Normalise a pixel colour to (R, G, B) tuple."""
    if isinstance(pixel, (tuple, list)):
        return (int(pixel[0]), int(pixel[1]), int(pixel[2]))
    if isinstance(pixel, int):
        return ((pixel >> 16) & 0xFF, (pixel >> 8) & 0xFF, pixel & 0xFF)
    return (255, 255, 255)


# ---------------------------------------------------------------------------
# PixelGrid — framebuffer
# ---------------------------------------------------------------------------


class PixelGrid:
    """Full-screen RGBA-like framebuffer with paint primitives and hit zones.

    The buffer is RGB uint8; transparent pixels carry TRANSPARENT_RGB so the
    window colour key makes them invisible.
    """

    def __init__(self, width: int, height: int) -> None:
        self.w: int = width
        self.h: int = height
        self.buffer: NDArray[np.uint8] = np.full(
            (height, width, 3), TRANSPARENT_RGB, dtype=np.uint8,
        )
        # hit_zones: {(x1, y1, x2, y2): {"on_click": fn, "on_drag": fn, "on_release": fn, "agent": ref}}
        self.hit_zones: Dict[Tuple[int, int, int, int], Dict[str, Any]] = {}
        self._modified: bool = True  # flag for render loop
        self._lock: threading.Lock = threading.Lock()

    # ── Paint primitives ──────────────────────────────────────────────

    def paint(self, x: int, y: int, r: int, g: int, b: int) -> None:
        """Set one pixel. Clips to buffer bounds."""
        if 0 <= x < self.w and 0 <= y < self.h:
            with self._lock:
                self.buffer[y, x] = (r, g, b)
            self._modified = True

    def paint_rect(self, x1: int, y1: int, x2: int, y2: int,
                   color: Tuple[int, int, int]) -> None:
        """Fill a rectangle [x1, x2) × [y1, y2). Inclusive-exclusive."""
        x1s: int = max(0, x1)
        y1s: int = max(0, y1)
        x2s: int = min(self.w, x2)
        y2s: int = min(self.h, y2)
        if x1s >= x2s or y1s >= y2s:
            return
        with self._lock:
            self.buffer[y1s:y2s, x1s:x2s] = color
        self._modified = True

    def paint_outline(self, x1: int, y1: int, x2: int, y2: int,
                      color: Tuple[int, int, int]) -> None:
        """Draw a 1-pixel rectangle outline."""
        self.paint_rect(x1, y1, x2, y1 + 1, color)
        self.paint_rect(x1, y2 - 1, x2, y2, color)
        self.paint_rect(x1, y1, x1 + 1, y2, color)
        self.paint_rect(x2 - 1, y1, x2, y2, color)

    def paint_sprite(self, x: int, y: int, matrix: List[List[int]],
                     palette: List[Tuple[int, int, int]]) -> None:
        """Draw a pixel-art sprite from a matrix of palette indices.

        Args:
            x, y: Top-left screen position.
            matrix: Rows of int indices into palette (-1 = transparent).
            palette: List of (R, G, B) colours.
        """
        for dy, row in enumerate(matrix):
            for dx, idx in enumerate(row):
                if idx >= 0 and idx < len(palette):
                    self.paint(x + dx, y + dy, *palette[idx])

    def paint_text(self, x: int, y: int, text: str,
                   color: Tuple[int, int, int]) -> None:
        """Draw a 5×7 pixel-font text string at (x, y)."""
        cx: int = x
        for ch in text:
            glyph: Optional[List[int]] = _FONT_5x7.get(ch)
            if glyph is None:
                glyph = _FONT_5x7.get('?', _FONT_5x7[' '])
            for row_idx, row_bits in enumerate(glyph):
                for col_idx in range(CHAR_W):
                    if row_bits & (1 << (4 - col_idx)):
                        self.paint(cx + col_idx, y + row_idx, *color)
            cx += CHAR_W + CHAR_GAP_X

    def paint_text_block(self, x: int, y: int, text: str,
                         color: Tuple[int, int, int],
                         max_width: int = 0) -> None:
        """Draw multi-line text. max_width=0 means single line."""
        if max_width <= 0:
            self.paint_text(x, y, text, color)
            return
        line_y: int = y
        line_chars: int = max_width // (CHAR_W + CHAR_GAP_X)
        if line_chars < 1:
            line_chars = 1
        for i in range(0, len(text), line_chars):
            self.paint_text(x, line_y, text[i:i + line_chars], color)
            line_y += CHAR_H + CHAR_GAP_Y

    def clear(self) -> None:
        """Fill entire buffer with transparent colour."""
        with self._lock:
            self.buffer[:] = TRANSPARENT_RGB
        self._modified = True

    def clear_rect(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """Set a rectangle to transparent."""
        self.paint_rect(x1, y1, x2, y2, TRANSPARENT_RGB)

    # ── Hit zones ────────────────────────────────────────────────────

    def register_zone(self, x1: int, y1: int, x2: int, y2: int,
                      on_click: Optional[HitCallback] = None,
                      on_drag: Optional[HitCallback] = None,
                      on_release: Optional[HitCallback] = None,
                      agent: Any = None) -> None:
        """Register a clickable area.

        Calls on_click(mouse_x, mouse_y, 'click') when clicked,
        on_drag(mx, my, 'drag') during drag,
        on_release(mx, my, 'release') on release.
        """
        key: Tuple[int, int, int, int] = (x1, y1, x2, y2)
        self.hit_zones[key] = {
            'on_click': on_click,
            'on_drag': on_drag,
            'on_release': on_release,
            'agent': agent,
        }

    def unregister_zone(self, x1: int, y1: int, x2: int, y2: int) -> None:
        key: Tuple[int, int, int, int] = (x1, y1, x2, y2)
        self.hit_zones.pop(key, None)

    def hit_test(self, mx: int, my: int) -> Optional[Dict[str, Any]]:
        """Find the top-most (last-registered) hit zone at (mx, my)."""
        matched: Optional[Dict[str, Any]] = None
        for (x1, y1, x2, y2), info in self.hit_zones.items():
            if x1 <= mx < x2 and y1 <= my < y2:
                matched = info
        return matched

    # ── Rendering ─────────────────────────────────────────────────────

    def render_to_photo(self) -> Any:
        """Convert current buffer to a tkinter-compatible PhotoImage.

        Returns None if PIL is not available.
        """
        if Image is None or ImageTk is None:
            return None
        with self._lock:
            img: Image.Image = Image.fromarray(self.buffer, mode='RGB')
        self._modified = False
        return ImageTk.PhotoImage(img)


# ---------------------------------------------------------------------------
# PixelGridWindow — full-screen transparent Toplevel overlay
# ---------------------------------------------------------------------------


class PixelGridWindow:
    """Full-screen transparent overlay window that renders a PixelGrid buffer.

    Runs its own render loop (~15 fps) independent of the cube animation.
    Forwards mouse events to the grid's hit_zones.
    """

    TRANSPARENT_COLOR: str = '#000001'

    def __init__(self, grid: PixelGrid,
                 on_close: Optional[Callable[[], None]] = None) -> None:
        import tkinter as tk

        self.grid: PixelGrid = grid
        self.on_close: Optional[Callable[[], None]] = on_close
        self.running: bool = False
        self._photo: Any = None  # keep ref to prevent GC
        self._show_hint: bool = True

        # ── Create Toplevel ───────────────────────────────────────────
        self.root = tk.Toplevel()
        self.root.title('♢ PixelGrid')
        self.root.overrideredirect(True)
        self.root.configure(bg=self.TRANSPARENT_COLOR)
        self.root.attributes('-transparentcolor', self.TRANSPARENT_COLOR)
        self.root.attributes('-topmost', True)

        # Full screen
        sw: int = self.root.winfo_screenwidth()
        sh: int = self.root.winfo_screenheight()
        self.root.geometry(f'{sw}x{sh}+0+0')

        # ── Canvas ────────────────────────────────────────────────────
        self.canvas = tk.Canvas(
            self.root, bg=self.TRANSPARENT_COLOR, highlightthickness=0,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        # Placeholder for the rendered image
        self._image_item: int = self.canvas.create_image(
            0, 0, anchor='nw', image=None,
        )

        # ── Mouse handling ────────────────────────────────────────────
        self.canvas.bind('<Button-1>', self._on_click)
        self.canvas.bind('<B1-Motion>', self._on_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_release)
        self.root.bind('<Escape>', lambda e: self._hide())
        self.root.bind('h', lambda e: self._hide())

        # ── Start render loop ─────────────────────────────────────────
        self.running = True
        self.root.after(66, self._render_loop)  # ~15 fps

    # ── Visibility ────────────────────────────────────────────────────

    def show(self) -> None:
        """Show the PixelGrid overlay."""
        self.root.deiconify()
        self.root.lift()
        self.root.lift()
        self.running = True
        self.root.after(66, self._render_loop)

    def _hide(self) -> None:
        """Hide the overlay."""
        self.running = False
        self.root.withdraw()
        if self.on_close:
            self.on_close()

    def toggle(self) -> None:
        """Flip visibility."""
        if self.root.state() == 'withdrawn':
            self.show()
        else:
            self._hide()

    def close(self) -> None:
        """Destroy the window entirely."""
        self.running = False
        try:
            self.root.destroy()
        except Exception:
            pass

    # ── Mouse dispatchers ─────────────────────────────────────────────

    def _on_click(self, event: Any) -> None:
        mx: int = event.x
        my: int = event.y
        zone = self.grid.hit_test(mx, my)
        if zone and zone.get('on_click'):
            zone['on_click'](mx, my, 'click')

    def _on_drag(self, event: Any) -> None:
        mx: int = event.x
        my: int = event.y
        zone = self.grid.hit_test(mx, my)
        if zone and zone.get('on_drag'):
            zone['on_drag'](mx, my, 'drag')

    def _on_release(self, event: Any) -> None:
        mx: int = event.x
        my: int = event.y
        zone = self.grid.hit_test(mx, my)
        if zone and zone.get('on_release'):
            zone['on_release'](mx, my, 'release')

    # ── Render loop ───────────────────────────────────────────────────

    def _render_loop(self) -> None:
        """Periodically blit grid buffer to canvas."""
        if not self.running:
            return
        try:
            if self.grid._modified:
                photo = self.grid.render_to_photo()
                if photo is not None:
                    self._photo = photo  # prevent GC
                    self.canvas.itemconfig(self._image_item, image=self._photo)
        except Exception:
            pass
        self.root.after(66, self._render_loop)
