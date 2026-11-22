"""systems/color.py — Цвет частиц.

Читает:
  sim.color — базовые цвета (r0, g0, b0)
  sim.world_position — для глубины (z)
  meta.color_shift — AI mood HSV shift
  meta.config.color_mode — 'default' | 'z_layers'

Режим 'z_layers': раскрашивает частицы по Z-глубине в палитру Spatial Depth:
  Z 0.00-0.25 → золото   (#f59e0b) — Surface
  Z 0.25-0.50 → зелёный  (#22c55e) — Logic
  Z 0.50-0.75 → индиго   (#6366f1) — Semantic
  Z 0.75-1.00 → розовый  (#ec4899) — Singularity

Пишет:
  render.final_rgb — (N, 3) uint8 готовый к отрисовке
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from core.world import World
from core.face_themes import face_colors
from core.friendly_face import friendly_face_colors

# Палитра Spatial Depth по Z-слоям
Z_LAYER_COLORS = np.array([
    [0.96, 0.62, 0.04],  # gold   — Surface    (Z 0.00-0.25)
    [0.13, 0.77, 0.37],  # green  — Logic      (Z 0.25-0.50)
    [0.39, 0.40, 0.95],  # indigo — Semantic   (Z 0.50-0.75)
    [0.93, 0.27, 0.60],  # pink   — Singularity (Z 0.75-1.00)
], dtype=np.float64)

# Z-границы слоёв
Z_BOUNDARIES = np.array([0.0, 0.25, 0.50, 0.75, 1.0], dtype=np.float64)


def update(world: World, dt: float) -> None:
    """Вычислить финальные цвета."""
    n = world.sim.active_count
    if n == 0:
        world.render.final_rgb = np.array([], dtype=np.uint8).reshape(0, 3)
        return

    color_mode = world.meta.config.get('color_mode', 'default')

    if color_mode == 'z_layers':
        _apply_z_layers(world, n)
    else:
        _apply_default_depth(world, n)
        if world.meta.config.get('shape_preset') == 'cloud':
            _apply_cloud_tint(world, n)
        elif world.meta.config.get('shape_preset') in ('face', 'face_original'):
            _apply_face_theme(world, n)

    # HSV shift от AI mood (поверх всего)
    shift = world.meta.color_shift
    if shift > 0.01:
        r = world.render.final_rgb[:n, 0].astype(np.float64)
        g = world.render.final_rgb[:n, 1].astype(np.float64)
        b = world.render.final_rgb[:n, 2].astype(np.float64)
        r, g, b = _apply_hsv_shift(r, g, b, shift)
        world.render.final_rgb[:n] = np.column_stack((r, g, b)).astype(np.uint8)


def _position_colors(pts: NDArray[np.float64]) -> NDArray[np.float64]:
    """Тот же градиент, что у куба: позиция → RGB (0..255, float)."""
    return np.column_stack((
        (pts[:, 0] + 1.0) * 127.5,
        (pts[:, 1] + 1.0) * 127.5,
        (pts[:, 2] + 1.0) * 127.5,
    ))


def _apply_default_depth(world: World, n: int) -> None:
    """Стандартный depth shading.

    Цвет частицы задаётся её ИСХОДНОЙ позицией в кубе (sim.color). В режиме
    текста частицы перелетают в плоскость надписи, а цвета остаются от старых
    мест — градиент рассыпается в шум. Поэтому при text_freeze цвет плавно
    пересчитывается от ТЕКУЩЕЙ позиции: тот же градиент, что у куба, но
    непрерывный вдоль надписи.
    """
    base = world.sim.color[:n, :3].astype(np.float64)

    freeze = float(getattr(world.meta, 'text_freeze', 0.0))
    freeze = min(1.0, max(0.0, freeze))

    pts = world.sim.world_position[:n]

    if freeze > 0.001:
        target = _position_colors(pts)
        rgb = base * (1.0 - freeze) + target * freeze
    else:
        rgb = base

    pz = pts[:, 2]
    if freeze > 0.001:
        # Надпись плоская (Z=0) → depth_factor становится константой 0.8 и
        # текст выглядит тускло-серым. Ближе к freeze=1 отдаём полную яркость.
        depth_factor = (0.6 + 0.4 * (pz + 1.0) / 2.0) * (1.0 - freeze) + freeze
    else:
        depth_factor = 0.6 + 0.4 * (pz + 1.0) / 2.0

    world.render.final_rgb = np.clip(
        rgb * depth_factor[:, None], 0, 255,
    ).astype(np.uint8)


def _apply_cloud_tint(world: World, n: int) -> None:
    """Soft pale blue/white cloud skin, blended by the existing morph slider."""
    morph = float(np.clip(world.meta.config.get('morph_progress', 0.0), 0.0, 1.0))
    if morph <= 0.0:
        return
    pts = world.sim.world_position[:n]
    # Camera-space lighting: directional key, back shadow, and soft rim.
    # This is inexpensive shading, not physically accurate cast shadows.
    centered = pts - pts.mean(axis=0, keepdims=True)
    normals = centered / np.clip(
        np.linalg.norm(centered, axis=1, keepdims=True), 1e-8, None
    )
    key = np.array([-0.46, -0.57, 0.68], dtype=np.float64)
    key /= np.linalg.norm(key)
    lambert = np.clip(normals @ key, 0.0, 1.0)
    front = np.clip(0.5 + 0.4 * normals[:, 2], 0.0, 1.0)
    rim = (1.0 - np.abs(normals[:, 2])) ** 2

    strength = float(np.clip(world.meta.config.get('cloud_light', 0.85), 0.0, 1.0))
    shadow_strength = float(np.clip(
        world.meta.config.get('cloud_shadow', 0.78), 0.0, 1.0
    ))
    illumination = np.clip(
        (0.23 + 0.22 * front + strength * (0.69 * lambert + 0.18 * rim))
        * (1.0 - 0.40 * shadow_strength * (1.0 - lambert)),
        0.0, 1.0,
    )
    cool_shadow = np.array([43.0, 89.0, 151.0])
    warm_light = np.array([241.0, 249.0, 255.0])
    cloud_rgb = (cool_shadow[None, :]
                 + illumination[:, None] * (warm_light - cool_shadow)[None, :])

    # The original cube colours and avatar text remain unchanged.
    freeze = float(np.clip(getattr(world.meta, 'text_freeze', 0.0), 0.0, 1.0))
    blend = morph * (1.0 - freeze)
    original_rgb = world.render.final_rgb[:n].astype(np.float64)
    world.render.final_rgb[:n] = np.clip(
        original_rgb * (1.0 - blend) + cloud_rgb * blend, 0, 255
    ).astype(np.uint8)



def _apply_face_theme(world: World, n: int) -> None:
    """Blend Hermes Face material with the existing cube and speech colors."""
    morph = float(np.clip(world.meta.config.get('morph_progress',0),0,1))
    freeze = float(np.clip(getattr(world.meta, 'text_freeze',0),0,1))
    blend = morph * (1.0-freeze)
    if blend <= 0.0:
        return
    shape = world.meta.config.get('shape_preset', 'face')
    geometry = world.sim.shape_cache.get(shape)
    if geometry is None or len(geometry) < n:
        return
    palette = friendly_face_colors if shape == 'face' else face_colors
    themed = palette(geometry[:n], world.meta.config.get('face_theme','Holographic'), world.meta.mood)
    if blend >= .999:
        world.render.final_rgb[:n] = themed
    else:
        original = world.render.final_rgb[:n].astype(np.float64)
        world.render.final_rgb[:n] = np.clip(original*(1.0-blend)+themed*blend, 0,255).astype(np.uint8)


def _apply_z_layers(world: World, n: int) -> None:
    """Раскрасить частицы по Z-глубине в палитру Spatial Depth."""
    pz = world.sim.world_position[:n, 2]

    # Нормализовать Z в [0, 1]
    # Spiral: Z уже [0, 1], Cube: Z в [-1, 1]
    z_min = pz.min()
    z_max = pz.max()
    if z_min >= 0.0:
        z_norm = pz.copy()  # уже [0, 1]
    else:
        z_norm = (pz + 1.0) / 2.0  # из [-1, 1] в [0, 1]
    z_norm = np.clip(z_norm, 0.0, 1.0)

    # Для каждой частицы найти индекс слоя
    # z=0 → 0, z=0.25 → 1, z=0.5 → 2, z=0.75 → 3
    idx = np.clip(np.searchsorted(Z_BOUNDARIES[1:], z_norm, side='right'), 0, 3)

    # Взять цвет из палитры
    colors = Z_LAYER_COLORS[idx]  # (N, 3) float64

    # Умножить на 255 для uint8
    world.render.final_rgb = (colors * 255.0).astype(np.uint8)


def _apply_hsv_shift(r, g, b, shift):
    """Векторизованный RGB→HSV→сдвиг→RGB."""
    rn, gn, bn = r / 255.0, g / 255.0, b / 255.0
    mx = np.maximum(np.maximum(rn, gn), bn)
    mn = np.minimum(np.minimum(rn, gn), bn)
    delta = mx - mn

    h = np.zeros_like(rn)
    mask = delta > 1e-6
    rm = mask & (mx == rn)
    gm = mask & (mx == gn)
    bm = mask & (mx == bn)
    h[rm] = ((gn[rm] - bn[rm]) / delta[rm]) % 6.0
    h[gm] = ((bn[gm] - rn[gm]) / delta[gm]) + 2.0
    h[bm] = ((rn[bm] - gn[bm]) / delta[bm]) + 4.0
    h = h / 6.0

    s = np.zeros_like(rn)
    s[mask] = delta[mask] / mx[mask]
    v = mx
    h = (h + shift) % 1.0

    h6 = h * 6.0
    hi = np.floor(h6).astype(np.int32)
    f = h6 - hi.astype(np.float64)
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    r_out = np.clip(np.select(
        [hi == 0, hi == 1, hi == 2, hi == 3, hi == 4, hi == 5],
        [v, q, p, p, t, v]) * 255.0, 0, 255)
    g_out = np.clip(np.select(
        [hi == 0, hi == 1, hi == 2, hi == 3, hi == 4, hi == 5],
        [t, v, v, q, p, p]) * 255.0, 0, 255)
    b_out = np.clip(np.select(
        [hi == 0, hi == 1, hi == 2, hi == 3, hi == 4, hi == 5],
        [p, p, t, v, v, q]) * 255.0, 0, 255)
    return r_out, g_out, b_out
