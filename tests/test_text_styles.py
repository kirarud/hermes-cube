"""tests/test_text_styles.py — Стили набора текста из частиц.

Проверяет, что каждый стиль из TEXT_STYLES реально меняет раскладку
(иначе переключатель в трее был бы декорацией), и что старая сигнатура
layout_text_mask не сломалась.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from core.text_layout import (
    DEFAULT_TEXT_STYLE,
    TEXT_STYLES,
    _edge_mask,
    layout_text_mask,
    layout_text_mask_styled,
)

N = 4000
TEXT = 'КИРА'


def _styled(style: str):
    return layout_text_mask_styled(TEXT, N, font_size=42, style=style)


def test_all_styles_return_valid_layout():
    for style in TEXT_STYLES:
        pos, n_used, bbox, w = _styled(style)
        assert pos.shape == (N, 3), style
        assert w.shape == (N,), style
        assert 0 < n_used <= N, style
        assert len(bbox) == 4, style
        # веса в диапазоне 0..1
        assert float(w.min()) >= 0.0, style
        assert float(w.max()) <= 1.0, style


def test_unknown_style_falls_back_to_solid():
    pos, n_used, _, w = _styled('какая-то ерунда')
    ref_pos, ref_n, _, ref_w = _styled(DEFAULT_TEXT_STYLE)
    assert n_used == ref_n
    assert np.array_equal(pos, ref_pos)
    assert np.array_equal(w, ref_w)


def test_legacy_signature_still_three_values():
    pos, n_used, bbox = layout_text_mask(TEXT, N, font_size=42)
    assert pos.shape == (N, 3)
    assert n_used > 0
    ref_pos, ref_n, _, _ = _styled('solid')
    assert n_used == ref_n
    assert np.allclose(pos, ref_pos)


def test_outline_is_thinner_than_solid():
    _, solid_n, _, _ = _styled('solid')
    _, out_n, _, _ = _styled('outline')
    assert 0 < out_n < solid_n, (out_n, solid_n)


def test_grid_is_thinner_than_solid():
    _, solid_n, _, _ = _styled('solid')
    _, grid_n, _, _ = _styled('grid')
    assert 0 < grid_n < solid_n, (grid_n, solid_n)


def test_glow_uses_all_pixels_like_solid():
    """glow не меняет маску — мягкость даёт рендер через char_mode='glow'."""
    _, solid_n, _, _ = _styled('solid')
    _, glow_n, _, glow_w = _styled('glow')
    assert glow_n == solid_n
    assert float(glow_w[:glow_n].min()) == 1.0


def test_halftone_dim_interior_keeps_edges_bright():
    pos, n_used, _, w = _styled('halftone')
    used = w[:n_used]
    assert abs(float(used.max()) - 1.0) < 1e-6
    assert 0.25 < float(used.min()) < 0.35
    # контур и interior присутствуют одновременно — иначе это просто outline
    assert np.count_nonzero(used > 0.9) < n_used


def test_wave_fades_left_to_right():
    pos, n_used, _, w = _styled('wave')
    used_x = pos[:n_used, 0]
    used_w = w[:n_used]
    left = used_w[used_x < np.quantile(used_x, 0.2)].mean()
    right = used_w[used_x > np.quantile(used_x, 0.8)].mean()
    assert left > right + 0.2, (left, right)
    assert abs(float(used_w.max()) - 1.0) < 1e-6
    assert float(used_w.min()) >= 0.3


def test_edge_mask_finds_border_only():
    block = np.zeros((5, 5), dtype=bool)
    block[1:4, 1:4] = True
    edge = _edge_mask(block)
    assert edge.sum() == 8          # рамка 3×3 без центра
    assert not edge[2, 2]
    assert edge[1, 1]


def test_empty_text_is_safe_for_every_style():
    for style in TEXT_STYLES:
        pos, n_used, bbox, w = layout_text_mask_styled('', N, style=style)
        assert n_used == 0
        assert pos.shape == (N, 3)
        assert w.shape == (N,)


# ── Интеграция: переключение стиля на живой системе ─────────────────────

def _live_world(style: str, n: int = 900):
    """World с надписью на экране в заданном стиле."""
    from core.world import World
    from core.systems.avatar_text import AvatarTextSystem
    from cube_app import DEFAULT_CONFIG

    cfg = dict(DEFAULT_CONFIG)
    cfg['text_style'] = style
    world = World.create(cfg, n_particles=n, pool_size=4096)
    world.sim.color[:] = 200.0
    world.meta.w = 600
    world.meta.h = 600
    at = AvatarTextSystem()
    world.meta.speak_buffer = TEXT
    at.update(world, 0.016)
    return world, at


def test_style_does_not_accumulate_dimming():
    """Смена стиля туда-обратно не должна гасить надпись всё сильнее."""
    world, at = _live_world('halftone')
    n = world.sim.active_count
    assert float(world.sim.color[:n].min()) < 100.0   # interior приглушён

    for style in ('solid', 'halftone', 'wave', 'solid'):
        at.set_style(world, style)

    assert float(world.sim.color[:n].min()) > 199.0
    assert float(world.sim.color[:n].max()) < 201.0


def test_glow_style_forces_glow_char_mode():
    world, at = _live_world('glow')
    assert world.meta.config['char_mode'] == 'glow'


def test_solid_style_keeps_configured_char_mode():
    world, at = _live_world('solid')
    assert world.meta.config['char_mode'] == 'symbols'


def test_colors_restored_after_text_finishes():
    world, at = _live_world('halftone')
    n = world.sim.active_count
    for _ in range(500):
        at.update(world, 0.05)
        if at.state == 'idle':
            break
    assert at.state == 'idle'
    assert float(world.sim.color[:n].min()) > 199.0
    assert world.meta.text_mode is False


def test_unknown_style_is_ignored_by_set_style():
    world, at = _live_world('solid')
    at.set_style(world, 'ерунда')
    assert world.meta.config['text_style'] == DEFAULT_TEXT_STYLE
