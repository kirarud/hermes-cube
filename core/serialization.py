"""core/serialization.py — Сохранение и загрузка World.

Позволяет:
  - Сериализовать World в bytes (numpy + json)
  - Восстановить World из bytes
  - Делать snapshot (Ctrl+S) и восстанавливать (Ctrl+O)
  - Replay: запись dt + events в буфер для воспроизведения

Формат: {numpy_positions.npy, numpy_colors.npy, meta.json}
"""

from __future__ import annotations

import io
import json
import os
import zipfile
from typing import Any, Dict, Optional, Tuple

import numpy as np

from core.world import World


def serialize_world(world: World) -> bytes:
    """Сериализовать World в zip-архив (bytes).

    Содержимое:
      - sim/position.npy
      - sim/color.npy
      - sim/velocity.npy
      - meta.json (config, mood, frame, время)
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Simulation state
        n = world.sim.active_count
        _add_npy(zf, 'sim/position.npy', world.sim.position[:n])
        _add_npy(zf, 'sim/color.npy', world.sim.color[:n])
        _add_npy(zf, 'sim/velocity.npy', world.sim.velocity[:n])

        # Meta
        meta = {
            't': world.meta.t,
            'frame': world.meta.frame,
            'mood': world.meta.mood,
            'color_shift': world.meta.color_shift,
            'config': world.meta.config,
        }
        zf.writestr('meta.json', json.dumps(meta, indent=2, ensure_ascii=False))

    return buf.getvalue()


def deserialize_world(data: bytes, pool_size: int = 4096) -> Optional[World]:
    """Восстановить World из bytes (zip)."""
    try:
        buf = io.BytesIO(data)
        with zipfile.ZipFile(buf, 'r') as zf:
            pos = _read_npy(zf, 'sim/position.npy')
            col = _read_npy(zf, 'sim/color.npy')
            vel = _read_npy(zf, 'sim/velocity.npy')
            meta = json.loads(zf.read('meta.json'))
    except Exception:
        return None

    n = len(pos)
    config = meta.get('config', {})

    world = World.create(config, n_particles=n, pool_size=pool_size)
    world.sim.position[:n] = pos
    world.sim.color[:n] = col
    world.sim.velocity[:n] = vel
    world.meta.t = meta.get('t', 0.0)
    world.meta.frame = meta.get('frame', 0)
    world.meta.mood = meta.get('mood', 'idle')
    world.meta.color_shift = meta.get('color_shift', 0.0)

    return world


def save_snapshot(world: World, path: str) -> bool:
    """Сохранить snapshot мира в файл."""
    try:
        data = serialize_world(world)
        with open(path, 'wb') as f:
            f.write(data)
        return True
    except Exception:
        return False


def load_snapshot(path: str) -> Optional[World]:
    """Загрузить snapshot мира из файла."""
    try:
        with open(path, 'rb') as f:
            return deserialize_world(f.read())
    except Exception:
        return None


def _add_npy(zf: zipfile.ZipFile, name: str, arr: np.ndarray) -> None:
    buf = io.BytesIO()
    np.save(buf, arr)
    zf.writestr(name, buf.getvalue())


def _read_npy(zf: zipfile.ZipFile, name: str) -> np.ndarray:
    return np.load(io.BytesIO(zf.read(name)))


# ── Replay ──────────────────────────────────────────────────────────


class ReplayBuffer:
    """Кольцевой буфер для записи/воспроизведения кадров.

    Записывает dt и события на каждом кадре.
    Воспроизведение: pipeline.run() без AI, из записи.
    """

    def __init__(self, max_frames: int = 1800) -> None:  # 30s @ 60fps
        self.max_frames = max_frames
        self.frames: list[dict] = []
        self._recording: bool = False
        self._playing: bool = False
        self._play_idx: int = 0

    def start_recording(self) -> None:
        self.frames.clear()
        self._recording = True

    def stop_recording(self) -> None:
        self._recording = False

    def record_frame(self, world: World, dt: float) -> None:
        if not self._recording:
            return
        if len(self.frames) >= self.max_frames:
            self.frames.pop(0)
        self.frames.append({
            'dt': dt,
            'events': list(world.meta.events),
            'mood': world.meta.mood,
        })

    def start_playback(self) -> None:
        self._playing = True
        self._play_idx = 0

    def stop_playback(self) -> None:
        self._playing = False

    def playback_frame(self, world: World) -> bool:
        """Применить следующий кадр. True если кадр есть."""
        if not self._playing or self._play_idx >= len(self.frames):
            return False
        frame = self.frames[self._play_idx]
        self._play_idx += 1
        # Можно восстановить mood, события и т.д.
        world.meta.mood = frame['mood']
        return True

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def is_playing(self) -> bool:
        return self._playing
