"""systems/avatar_text.py — Аватар: текст маской из частиц.

Частицы куба морфятся в пиксели букв (bitmap маска).
Поверх — один create_text РОВНО в том же месте.
Оставшиеся частицы — рамка вокруг текста.

Визуально: текст и частицы одно целое.
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
        self._original_shape: str = 'cube'
        self._original_scale: float = 0.27
        self._original_rotation_speed: float = 0.28
        self._text_scale: float = 0.65
        self._target_pos: NDArray[np.float64] | None = None
        self._start_base: NDArray[np.float64] | None = None
        self._start_colors: NDArray[np.float64] | None = None
        self._n_used: int = 0
        self._display_text: str = ''
        self._text_item: int | None = None
        self._bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    def update(self, world: World, dt: float) -> None:
        response = world.meta.ai_response
        if response and response != self._last_response and not world.meta.ai_thinking:
            self._last_response = response
            self._start_text_display(world)
            world.meta.ai_response = ''

        if self.state == 'idle':
            return
        if self.state == 'morph_in':
            self._timer += dt
            t = min(1.0, self._timer / MORPH_IN_TIME)
            self._set_morph(world, t)
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
            self._set_morph(world, t)
            if t <= 0.02:
                self._finish(world)

    def _set_morph(self, world: World, t: float) -> None:
        n = world.sim.active_count
        if n == 0 or self._target_pos is None or self._start_base is None:
            return
        n_used = min(n, len(self._target_pos), len(self._start_base))
        if n_used == 0:
            return

        world.sim.world_position[:n_used] = (
            self._start_base[:n_used] * (1.0 - t) + self._target_pos[:n_used] * t
        )
        if n_used < n:
            world.sim.world_position[n_used:] = (
                self._start_base[n_used:] * (1.0 - t) +
                self._target_pos[n_used:] * t
            )

        cfg = world.meta.config
        cfg['cube_scale'] = self._original_scale * (1.0 - t) + self._text_scale * t

        if self._start_colors is not None:
            world.sim.color[:n_used] = (
                self._start_colors[:n_used] * (1.0 - t) +
                np.array([200.0, 220.0, 255.0]) * t
            )

    def _show_overlay_text(self, world: World) -> None:
        """Один create_text в том же месте где частицы-буквы."""
        if not self.canvas or not self._display_text:
            return
        cx, cy, tw, th = self._bbox

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            return

        px = (cx + 1.0) / 2.0 * cw
        py = (1.0 - cy) / 2.0 * ch

        # Размер шрифта: подгоняем чтобы текст помещался по ширине
        text_len = len(self._display_text)
        if text_len > 0:
            font_size = max(14, min(48, int(cw * 0.7 / text_len * 1.5)))
        else:
            font_size = 28

        self._text_item = self.canvas.create_text(
            int(px), int(py),
            text=self._display_text,
            fill='#d0e0ff',
            font=('Segoe UI', font_size, 'bold'),
            anchor='center', justify='center',
            tags='avatar_overlay',
        )

    def _hide_overlay_text(self) -> None:
        if self._text_item:
            try:
                self.canvas.delete(self._text_item)
            except Exception:
                pass
            self._text_item = None
        try:
            self.canvas.delete('avatar_overlay')
        except Exception:
            pass

    def _start_text_display(self, world: World) -> None:
        n = world.sim.active_count
        if n == 0:
            return

        text = self._extract_text(world.meta.ai_response)
        if not text:
            return

        print(f"[AvatarText] → \"{text[:60]}...\" ({len(text)} chars)", flush=True)

        cfg = world.meta.config
        self._original_shape = cfg.get('shape_preset', 'cube')
        self._original_scale = float(cfg.get('cube_scale', 0.27))
        self._original_rotation_speed = float(cfg.get('rotation_speed', 0.28))

        self._start_base = world.sim.base_position[:n].copy()
        self._start_colors = world.sim.color[:n].copy()

        # Генерируем маску текста
        positions, n_used, bbox = layout_text_mask(text, n,
            font_name='Segoe UI', font_size=48)
        self._target_pos = positions.copy()
        self._n_used = n_used
        self._display_text = text
        self._bbox = bbox

        cfg['morph_progress'] = 0.0
        cfg['cube_scale'] = self._original_scale
        cfg['char_mode'] = 'dots'
        cfg['rotation_speed'] = 0.0
        cfg['shape_preset'] = 'text'
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
        cfg['shape_preset'] = self._original_shape
        cfg['morph_progress'] = 0.0
        cfg['cube_scale'] = self._original_scale
        cfg['rotation_speed'] = self._original_rotation_speed
        cfg['char_mode'] = 'dots'
        world.meta.text_mode = False
        world.meta.mood = 'idle'
        world.meta.color_shift = 0.0

        if self._start_colors is not None:
            n = min(len(self._start_colors), world.sim.active_count)
            world.sim.color[:n] = self._start_colors[:n]
        if getattr(self, '_trails_was_enabled', False):
            world.render.trail_enabled = True

        self.state = 'idle'
        self._last_response = ''
        self._target_pos = None
        self._start_base = None
        self._start_colors = None
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
