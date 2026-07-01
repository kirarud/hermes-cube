"""systems/avatar_text.py — Аватар: текст маской из частиц.

Переход куба в текст и обратно.
Overlay текст позиционируется ТОЧНО по формуле проекции — совпадает с частицами.
Плотность временно повышается для читаемости.
Настройки не сбрасываются после анимации.
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

    def __init__(self, canvas: Any, renderer: Any = None) -> None:
        self.canvas = canvas
        self._renderer = renderer
        self.state: str = 'idle'
        self._timer: float = 0.0
        self._last_response: str = ''
        self._original_shape: str = 'cube'
        self._original_scale: float = 0.27
        self._text_scale: float = 1.0
        self._target_pos: NDArray[np.float64] | None = None
        self._start_base: NDArray[np.float64] | None = None
        self._start_colors: NDArray[np.float64] | None = None
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
                self._start_base[n_used:] * (1.0 - t) + self._target_pos[n_used:] * t
            )

        cube_scale = self._original_scale * (1.0 - t) + self._text_scale * t
        world.meta.config['cube_scale'] = cube_scale

        if self._start_colors is not None:
            world.sim.color[:n_used] = (
                self._start_colors[:n_used] * (1.0 - t) +
                np.array([200.0, 220.0, 255.0]) * t
            )

    def _show_overlay_text(self, world: World) -> None:
        """Текст ТОЧНО над частицами — используем ту же формулу проекции."""
        if not self.canvas or not self._display_text:
            return

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            return
        w, h = float(cw), float(ch)

        # Берём актуальные позиции частиц — они уже в world_position
        n = world.sim.active_count
        if n == 0 or self._target_pos is None:
            return
        n_used = min(n, len(self._target_pos))

        # Считаем реальный центр и размер текста по частицам
        tex_x = self._target_pos[:n_used, 0]
        tex_y = self._target_pos[:n_used, 1]
        cx_norm = (tex_x.min() + tex_x.max()) / 2.0
        cy_norm = (tex_y.min() + tex_y.max()) / 2.0
        tw_norm = (tex_x.max() - tex_x.min())
        th_norm = (tex_y.max() - tex_y.min())

        # Проецируем в пиксели: particle_x * scale * min(w,h)/2 + center
        s = self._text_scale
        base = min(w, h) / 2.0 * s
        px = cx_norm * base + w / 2.0 + world.meta.cube_ox
        py = -cy_norm * base + h / 2.0 + world.meta.cube_oy

        # Размер шрифта: width текста в пикселях
        text_width_px = tw_norm * base
        text_height_px = th_norm * base
        font_size = max(12, min(48, int(text_width_px / max(1, len(self._display_text)) * 2.2)))

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

        n = world.sim.active_count
        self._start_base = world.sim.base_position[:n].copy()
        self._start_colors = world.sim.color[:n].copy()

        # Генерируем маску
        positions, n_used, _ = layout_text_mask(text, n,
            font_name='Segoe UI', font_size=48)
        self._target_pos = positions.copy()
        self._n_used = n_used
        self._display_text = text

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
        cfg['char_mode'] = 'dots'
        # НЕ трогаем rotation_speed и mood
        world.meta.text_mode = False

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
