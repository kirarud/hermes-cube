#!/usr/bin/env python3
"""char_cube.py — Символьная поверхность куба.

Вместо цветных точек — каждая частица куба отображается как символ/буква.
AI-ответ печатается на гранях куба.

Modes:
  'dots'    — обычные точки (текущее поведение)
  'symbols' — каждая точка = символ (из набора)
  'words'   — символы группируются в слова на гранях
  'glow'    — символы пульсируют

Integration:
  char_mode добавляется в config как 'char_mode'
  CubeEngine.get_frame() возвращает char_matrix
  CubeApp._render_frame() использует create_text вместо create_rectangle
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import math

import numpy as np
from numpy.typing import NDArray

# ── Symbol sets ────────────────────────────────────────────────────────

SYMBOL_SETS: Dict[str, List[str]] = {
    'default': ['◆', '◇', '●', '○', '■', '□', '▲', '△', '♥', '♢', '★', '☆'],
    'hex':     list('0123456789ABCDEF'),
    'binary':  ['0', '1'],
    'blocks':  ['█', '▓', '▒', '░'],
    'arrows':  ['↑', '→', '↓', '←', '↗', '↘', '↙', '↖'],
    'moods':   ['😊', '😢', '🤖', '💡', '✨', '🔥', '💧', '🌀'],
    'rus':     list('АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'),
    'rus_lower': list('абвгдеёжзийклмнопрстуфхцчшщъыьэюя'),
    'custom':  ['#', '@', '%', '&', '+', '=', '~', '*'],
}


# ═══════════════════════════════════════════════════════════════════════════
# CharCubeEngine — extension of CubeEngine with char matrix
# ═══════════════════════════════════════════════════════════════════════════


class CharCubeMixin:
    """Mixin for CubeEngine that adds char_matrix support.

    Usage:
        class CubeEngine(CharCubeMixin, ...):
            ...

    Adds:
        - self.char_mode: str ('dots'|'symbols'|'words'|'glow')
        - self.char_matrix: NDArray of chars (same shape as particles)
        - self.surface_text: str — text to display on cube faces
        - set_surface_text(text) — fill faces with text
        - get_char_at(index) — return char for particle at index
    """

    def __init_char_cube(self, n_particles: int) -> None:
        self.char_mode: str = 'dots'
        self.symbol_set: str = 'default'
        # char_matrix: one char per particle
        self.char_matrix: NDArray = np.array(
            ['◆'] * n_particles, dtype='<U2')
        self.surface_text: str = ''
        self._text_pos: int = 0

    def set_surface_text(self, text: str) -> None:
        """Queue text to display on cube faces.
        Characters will be assigned to particles in order.
        """
        self.surface_text = text
        self._text_pos = 0

    def get_char_at(self, idx: int) -> str:
        """Return the character for particle at index, based on current mode."""
        if self.char_mode == 'dots':
            return ''

        symbols = SYMBOL_SETS.get(self.symbol_set, SYMBOL_SETS['default'])

        if self.char_mode == 'symbols':
            # Cycle through symbol set
            return symbols[idx % len(symbols)]

        if self.char_mode in ('words', 'glow'):
            # Try to show surface text
            if self.surface_text and self._text_pos < len(self.surface_text):
                return self.surface_text[self._text_pos]
            return symbols[idx % len(symbols)]

        return ''

    def advance_text(self, count: int = 1) -> None:
        """Advance text position (called each frame)."""
        if self.surface_text and self._text_pos < len(self.surface_text):
            self._text_pos += count
            if self._text_pos >= len(self.surface_text):
                self._text_pos = 0
                self.surface_text = ''  # cleared after full display

    def get_symbol_sets(self) -> List[str]:
        """Return available symbol set names."""
        return list(SYMBOL_SETS.keys())
