"""Hermes Entity alpha: one command contract for local UI, gamepads and future agents.

All mutations go through World.  No network, model loading, script execution or
persistence happens in this module.
"""
from __future__ import annotations

import math
from typing import Any

from core.face_themes import THEMES
from core.world import World

SHAPES = ('face', 'face_original', 'cloud', 'cube')
MOODS = ('idle', 'happy', 'sad', 'thinking', 'speaking')
MAX_YAW = 0.85
MAX_PITCH = 0.45


class EntityCommandError(ValueError):
    """Rejected entity command; the world remains unchanged."""


def validate_command(payload: Any) -> dict[str, Any]:
    """Validate the complete allowlisted message before applying any field."""
    if not isinstance(payload, dict):
        raise EntityCommandError('command must be a JSON object')
    action = payload.get('action')
    if action in ('shape', 'skin', 'mood'):
        if set(payload) != {'action', 'value'}:
            raise EntityCommandError('expected only action and value')
        value = payload['value']
        allowed = {'shape': SHAPES, 'skin': tuple(THEMES), 'mood': MOODS}[action]
        if not isinstance(value, str) or value not in allowed:
            raise EntityCommandError(f'invalid {action}; allowed: {", ".join(allowed)}')
        return {'action': action, 'value': value}
    if action == 'look':
        if set(payload) != {'action', 'x', 'y'}:
            raise EntityCommandError('look requires only x and y')
        values: dict[str, float] = {}
        for key in ('x', 'y'):
            value = payload[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise EntityCommandError(f'{key} must be a finite number')
            value = float(value)
            if not math.isfinite(value) or not -1.0 <= value <= 1.0:
                raise EntityCommandError(f'{key} must be in [-1, 1]')
            values[key] = value
        return {'action': 'look', **values}
    if action == 'reset':
        if set(payload) != {'action'}:
            raise EntityCommandError('reset accepts no other fields')
        return {'action': 'reset'}
    raise EntityCommandError('unknown action; allowed: shape, skin, mood, look, reset')


def apply_command(world: World, payload: Any) -> dict[str, Any]:
    """Apply a validated command; return the current public entity state."""
    command = validate_command(payload)
    cfg = world.meta.config
    meta = world.meta
    action = command['action']
    if action == 'shape':
        shape = command['value']
        cfg['shape_preset'] = shape
        cfg['morph_progress'] = 0.0 if shape == 'cube' else 1.0
    elif action == 'skin':
        theme = command['value']
        cfg.update({
            'shape_preset': 'face', 'face_theme': theme, 'morph_progress': 1.0,
            'color_mode': 'default', 'particle_mode': 'breathe',
            'wave_amp': .018 if theme == 'Soft Glow' else .04,
            'pulse_amplitude': .035 if theme == 'Soft Glow' else .08,
            'rotation_speed': .08 if theme == 'Soft Glow' else .12,
            'char_mode': 'dots',
            'symbol': 'square' if theme == 'Retro Hacker' else 'circle',
            'cell_size': 5,
        })
    elif action == 'mood':
        meta.mood = command['value']
        meta.mood_lock = True
    elif action == 'look':
        meta.entity_target_yaw = command['x'] * MAX_YAW
        meta.entity_target_pitch = command['y'] * MAX_PITCH
    elif action == 'reset':
        meta.entity_target_yaw = meta.entity_target_pitch = 0.0
        meta.entity_yaw = meta.entity_pitch = 0.0
    return entity_state(world)


def entity_state(world: World) -> dict[str, Any]:
    """Serializable state shared by the UI and status endpoint."""
    meta, cfg = world.meta, world.meta.config
    return {
        'shape': cfg.get('shape_preset', 'cube'),
        'skin': cfg.get('face_theme', 'Holographic'),
        'mood': meta.mood,
        'look': {
            'x': round(meta.entity_target_yaw / MAX_YAW, 4),
            'y': round(meta.entity_target_pitch / MAX_PITCH, 4),
        },
        'render_mode': 'particles',
    }
