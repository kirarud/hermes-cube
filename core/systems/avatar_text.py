"""systems/avatar_text.py — Аватар: текст через pipeline morph.

Ничего не пишет в world_position вручную.
Кладёт позиции текста в shape_cache['text'].
Morph pipeline lerp-ит base_position → text_position.
Управляет morph_progress и overlay текстом.
"""

from __future__ import annotations

import json
from typing import Any, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from core.world import World
from core.text_layout import layout_text_mask

MORPH_IN_TIME: float = 0.8
HOLD_TIME: float = 3.5
MORPH_OUT_TIME: float = 0.8


class AvatarTextSystem:

    def __init__(self, canvas: Any) -> None:
        self.canvas = canvas
        self.state: str = 'idle'
        self._timer: float = 0.0
        self._last_response: str = ''
        self._original_scale: float = 0.27
        self._original_speed: float = 0.28
        self._text_scale: float = 0.65
        self._n_used: int = 0
        self._display_text: str = ''
        self._text_item: int | None = None

    def update(self, world: World, dt: float) -> None:
        response = world.meta.ai_response
        if response and response != self._last_response and not world.meta.ai_thinking:
            self._last_response = response
            self._start_text_display(world)
            world.meta.ai_response = ''

        if self.state == 'idle':
            return

        cfg = world.meta.config
        if self.state == 'morph_in':
            self._timer += dt
            t = min(1.0, self._timer / MORPH_IN_TIME)
            cfg['morph_progress'] = t
            cfg['cube_scale'] = self._original_scale * (1.0 - t) + self._text_scale * t
            if t >= 1.0:
                self.state = 'display'
                self._timer = 0.0
                self._show_overlay_text(world)
        elif self.state == 'display':
            self._timer += dt
            if self._timer >= HOLD_TIME:
                self.state = 'morph_out'
                self._timer = 0.0
                self._hide_overlay_text()
        elif self.state == 'morph_out':
            self._timer += dt
            t = max(0.0, 1.0 - self._timer / MORPH_OUT_TIME)
            cfg['morph_progress'] = t
            cfg['cube_scale'] = self._original_scale * t + self._text_scale * (1.0 - t)
            if t <= 0.02:
                self._finish(world)

    def _show_overlay_text(self, world: World) -> None:
        if not self.canvas or not self._display_text:
            return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            return

        # Позиция текста: проецируем target позицию через scale
        n = world.sim.active_count
        tex = world.sim.shape_cache.get('text')
        if tex is None or n == 0:
            return
        n_used = min(n, len(tex))
        base = min(cw, ch) / 2.0 * self._text_scale
        cx = tex[:n_used, 0].mean() * base + cw / 2.0 + world.meta.cube_ox
        cy = -tex[:n_used, 1].mean() * base + ch / 2.0 + world.meta.cube_oy
        tw = (tex[:n_used, 0].max() - tex[:n_used, 0].min()) * base
        font_size = max(14, min(48, int(tw / max(1, len(self._display_text)) * 2.0)))

        self._text_item = self.canvas.create_text(
            int(cx), int(cy), text=self._display_text,
            fill='#d0e0ff', font=('Segoe UI', font_size, 'bold'),
            anchor='center', justify='center', tags='avatar_overlay',
        )

    def _hide_overlay_text(self) -> None:
        try:
            self.canvas.delete('avatar_overlay')
        except Exception:
            pass
        self._text_item = None

    def _start_text_display(self, world: World) -> None:
        n = world.sim.active_count
        if n == 0:
            return

        text = self._extract_text(world.meta.ai_response)
        if not text:
            return

        print(f"[AvatarText] → \"{text[:60]}...\" ({len(text)} chars)", flush=True)

        cfg = world.meta.config
        self._original_scale = float(cfg.get('cube_scale', 0.27))
        self._original_speed = float(cfg.get('rotation_speed', 0.28))

        # Генерируем позиции текста в shape_cache — Morph прочитает
        positions, n_used, _ = layout_text_mask(text, n, font_size=48)
        world.sim.shape_cache['text'] = positions.copy()
        self._n_used = n_used
        self._display_text = text

        # Включаем char_mode='symbols' для отображения частиц как символов
        # (частицы будут циклически выбирать символы из SYMBOL_SETS['default'])
        cfg['char_mode'] = 'dots'
        cfg['shape_preset'] = 'text'
        cfg['morph_progress'] = 0.0
        cfg['rotation_speed'] = 0.0
        cfg['cube_scale'] = self._original_scale
        world.meta.text_mode = True

        if world.render.trail_enabled:
            world.render.trail_enabled = False
            self._trails_was_enabled = True
        else:
            self._trails_was_enabled = False

        self.state = 'morph_in'
        self._timer = 0.0

    def _finish(self, world: World) -> None:
        self._hide_overlay_text()
        cfg = world.meta.config
        cfg['shape_preset'] = 'cube'
        cfg['morph_progress'] = 0.0
        cfg['cube_scale'] = self._original_scale
        cfg['rotation_speed'] = self._original_speed
        cfg['char_mode'] = 'dots'
        world.meta.text_mode = False

        if getattr(self, '_trails_was_enabled', False):
            world.render.trail_enabled = True

        self.state = 'idle'
        self._last_response = ''
        self._display_text = ''
        print("[AvatarText] ← restored", flush=True)

    @staticmethod
    def _extract_text(raw: str) -> str:
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
        except (json.JSONDecodeError, ValueError):
            pass
        return cleaned
