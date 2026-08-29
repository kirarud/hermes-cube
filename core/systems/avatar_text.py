"""systems/avatar_text.py — Минимальный аватар: текст из частиц.

Ничего не трогает в основном цикле куба.
Только пишет в shape_cache и управляет morph_progress.

Настройки (config, все опциональны):
  text_font_size   int   — размер шрифта маски (по умолчанию 42)
  text_hold_time   float — сколько секунд держать надпись (3.5)
  text_morph_time  float — длительность morph_in/morph_out (0.8)
  text_char_mode   str   — 'symbols' (буквы) | 'dots' | 'glow'
  text_max_chars   int   — обрезка длинного текста (48; 0 = без обрезки)
  text_fill        float — какую долю кадра занимает надпись (0.92)
  text_freeze_spin bool  — останавливать вращение на время текста (True)
  text_auto_scale  bool  — подбирать кегль так, чтобы 1 частица = 1 глиф (True)
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import numpy as np

from core.world import World
from core.text_layout import layout_text_mask, glyph_grid_cols

# Дефолты — используются, если ключа нет в config
MORPH_IN_TIME: float = 0.8
HOLD_TIME: float = 3.5
MORPH_OUT_TIME: float = 0.8

DEFAULTS: Dict[str, Any] = {
    'text_font_size': 42,
    'text_hold_time': HOLD_TIME,
    'text_morph_time': MORPH_IN_TIME,
    'text_char_mode': 'symbols',
    'text_max_chars': 48,
    'text_fill': 0.92,
    'text_freeze_spin': True,
    'text_auto_scale': True,
}

# Какие ключи конфига трогает система — их надо вернуть как было
_RESTORE_KEYS = ('shape_preset', 'morph_progress', 'rotation_speed', 'char_mode')


class AvatarTextSystem:

    def __init__(self) -> None:
        self.state: str = 'idle'
        self._timer: float = 0.0
        self._last_response: str = ''
        # Снимок конфига до вывода текста — чтобы вернуть анимацию как была
        self._saved: Dict[str, Any] = {}
        # Заморозка вращения на время надписи (из text_freeze_spin)
        self._freeze_enabled: bool = True

    # ── helpers ──────────────────────────────────────────────────────────
    @staticmethod
    def _opt(cfg: Dict[str, Any], key: str) -> Any:
        val = cfg.get(key)
        return DEFAULTS[key] if val is None else val

    def _morph_time(self, cfg: Dict[str, Any]) -> float:
        return max(0.05, float(self._opt(cfg, 'text_morph_time')))

    # ── main loop ────────────────────────────────────────────────────────
    def update(self, world: World, dt: float) -> None:
        buf = world.meta.speak_buffer
        if buf and buf != self._last_response:
            self._last_response = buf
            world.meta.speak_buffer = ''
            self._start_text(world, buf)
            return

        if self.state == 'idle':
            return

        cfg = world.meta.config
        morph_t = self._morph_time(cfg)

        if self.state == 'morph_in':
            self._timer += dt
            t = min(1.0, self._timer / morph_t)
            cfg['morph_progress'] = t
            self._sync_freeze(world, t)
            if t >= 1.0:
                self.state = 'display'
                self._timer = 0.0
        elif self.state == 'display':
            self._timer += dt
            self._sync_freeze(world, 1.0)
            if self._timer >= max(0.1, float(self._opt(cfg, 'text_hold_time'))):
                self.state = 'morph_out'
                self._timer = 0.0
        elif self.state == 'morph_out':
            # Чистый реверс morph_in: только morph_progress 1→0.
            self._timer += dt
            t = max(0.0, 1.0 - self._timer / morph_t)
            cfg['morph_progress'] = t
            self._sync_freeze(world, t)
            if t <= 0.02:
                self._finish(world)

    def _sync_freeze(self, world: World, progress: float) -> None:
        """text_freeze строго следует за morph_progress → morph_out = реверс.

        smoothstep даёт мягкое замирание вращения вместо рывка. Функция
        монотонная, поэтому обратный проход по progress остаётся точным
        реверсом прямого.
        """
        if not self._freeze_enabled:
            world.meta.text_freeze = 0.0
            return
        p = min(1.0, max(0.0, progress))
        world.meta.text_freeze = p * p * (3.0 - 2.0 * p)

    def _start_text(self, world: World, text: str) -> None:
        n = world.sim.active_count
        if n == 0 or not text:
            return
        cfg = world.meta.config

        max_chars = int(self._opt(cfg, 'text_max_chars') or 0)
        if max_chars > 0 and len(text) > max_chars:
            text = text[:max_chars - 1].rstrip() + '…'

        fill = float(self._opt(cfg, 'text_fill'))

        # Сколько глифов cell_size влезает в надпись на экране. Один глиф =
        # одна частица, иначе символы рисуются плотнее своего размера и
        # наслаиваются друг на друга (буквы «слипаются», за ними видны
        # другие буквы поменьше).
        grid_cols = None
        if bool(self._opt(cfg, 'text_auto_scale')):
            win_min = min(world.meta.w or 600, world.meta.h or 600)
            grid_cols = glyph_grid_cols(
                window_min_side=win_min,
                cube_scale=float(cfg.get('cube_scale', 0.27)),
                cell_size=int(cfg.get('cell_size', 6)),
                fill=fill,
            )

        positions, n_used, _ = layout_text_mask(
            text,
            n,
            font_size=int(self._opt(cfg, 'text_font_size')),
            fill=fill,
            grid_cols=grid_cols,
        )
        world.sim.shape_cache['text'] = positions.copy()

        # Запоминаем текущие настройки анимации ДО того, как их менять.
        self._saved = {k: cfg.get(k) for k in _RESTORE_KEYS}

        cfg['morph_progress'] = 0.0
        cfg['shape_preset'] = 'text'
        cfg['char_mode'] = str(self._opt(cfg, 'text_char_mode'))
        # rotation_speed НЕ трогаем: угол = t * speed, смена speed на ходу
        # даёт скачок на t*Δspeed рад. Вращение гасится через meta.text_freeze
        # (lerp результата), который идёт строго по morph_progress.
        self._freeze_enabled = bool(self._opt(cfg, 'text_freeze_spin'))
        world.meta.text_freeze = 0.0

        world.meta.text_mode = True

        # Per-particle символы из текста
        self._map_text_to_symbols(world, text, n_used)
        self.state = 'morph_in'
        self._timer = 0.0

    def _finish(self, world: World) -> None:
        cfg = world.meta.config
        # Возвращаем ровно то, что было — не хардкод 0.28/'dots'/'cube'
        for key, value in self._saved.items():
            if value is not None:
                cfg[key] = value
        cfg['morph_progress'] = self._saved.get('morph_progress') or 0.0
        if not self._saved.get('shape_preset'):
            cfg['shape_preset'] = 'cube'
        world.meta.text_mode = False
        world.meta.text_freeze = 0.0
        self._saved = {}
        self.state = 'idle'
        self._last_response = ''

    def _map_text_to_symbols(self, world: World, text: str, n_used: int) -> None:
        """Назначить каждой частице индекс символа из font atlas.

        Частицы идут в том же порядке, что пиксели маски (слева-вниз по строкам),
        поэтому символ выбирается по X-координате частицы — так буква под
        частицей совпадает с буквой, которую та рисует.
        """
        char_map = _build_char_map()
        n_chars = len(text)
        if n_chars == 0 or n_used == 0:
            world.sim.symbol_idx[:] = 0
            return

        pos = world.sim.shape_cache.get('text')
        if pos is None:
            world.sim.symbol_idx[:] = 0
            return

        xs = pos[:n_used, 0]
        x_min, x_max = float(xs.min()), float(xs.max())
        span = max(1e-6, x_max - x_min)
        # Индекс символа по горизонтальной позиции частицы
        col = ((xs - x_min) / span * n_chars).astype(np.int64)
        np.clip(col, 0, n_chars - 1, out=col)

        idx_lookup = np.array(
            [char_map.get(ch, char_map.get(' ', 0)) for ch in text],
            dtype=np.int32,
        )
        world.sim.symbol_idx[:n_used] = idx_lookup[col]
        if n_used < world.sim.active_count:
            world.sim.symbol_idx[n_used:] = 0

    @staticmethod
    def _extract_text(raw):
        if not raw:
            return ''
        cleaned = raw.strip()
        if cleaned.startswith('```'):
            cleaned = cleaned.strip('` \n')
            if cleaned.startswith('json'):
                cleaned = cleaned[4:].strip()
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                text = parsed.get('text', '') or str(parsed.get('display_text', ''))
                if text:
                    return text
        except Exception:
            pass
        return cleaned


_CHAR_MAP_CACHE: Optional[Dict[str, int]] = None


def _build_char_map() -> Dict[str, int]:
    """Индексы символов в font atlas. Порядок ДОЛЖЕН совпадать с font_atlas."""
    global _CHAR_MAP_CACHE
    if _CHAR_MAP_CACHE is not None:
        return _CHAR_MAP_CACHE

    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from char_cube import SYMBOL_SETS

    char_map: Dict[str, int] = {}
    for name, syms in SYMBOL_SETS.items():
        for ch in syms:
            if ch not in char_map and len(char_map) < 256:
                char_map[ch] = len(char_map)
    extra = (
        list('abcdefghijklmnopqrstuvwxyz') +
        list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') +
        list('0123456789') +
        list(".,!?—…:;'\"()[]{}@#$%^&*+=<>/~`|\\- ") +
        list('абвгдеёжзийклмнопрстуфхцчшщъыьэюя')
    )
    for ch in extra:
        if ch not in char_map and len(char_map) < 256:
            char_map[ch] = len(char_map)

    _CHAR_MAP_CACHE = char_map
    return char_map
