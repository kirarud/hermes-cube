"""test_text_freeze.py — заморозка вращения и цвета на время надписи.

Проверяет:
  1. rotation.py гасит вращение по meta.text_freeze (lerp результата)
  2. animation.py гасит колебания частиц
  3. color.py пересчитывает градиент от текущей позиции → плавные цвета
  4. text_freeze идёт строго по morph_progress → morph_out = точный реверс
  5. rotation_speed в config НЕ трогается (иначе скачок на t*Δspeed)
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from core.world import World
from core.systems import animation, color, rotation
from core.systems.avatar_text import AvatarTextSystem


def _world(n: int = 64) -> World:
    w = World.create({}, n_particles=n)
    pts = np.random.default_rng(7).uniform(-1, 1, size=(n, 3))
    w.sim.base_position[:n] = pts
    w.sim.morphed[:n] = pts
    w.sim.animated[:n] = pts
    w.sim.world_position[:n] = pts
    w.sim.color[:n, 0] = (pts[:, 0] + 1.0) * 127.5
    w.sim.color[:n, 1] = (pts[:, 1] + 1.0) * 127.5
    w.sim.color[:n, 2] = (pts[:, 2] + 1.0) * 127.5
    w.sim.active_count = n
    w.meta.w = 600
    w.meta.h = 600
    w.meta.t = 12.34  # ненулевое время — вращение заметно
    w.meta.config = {
        'rotation_speed': 0.5,
        'particle_mode': 'wave',
        'wave_amp': 0.3,
        'wave_speed': 1.5,
        'pulse_amplitude': 0.12,
        'pulse_rate': 1.8,
    }
    return w


class TestRotationFreeze(unittest.TestCase):

    def test_freeze_1_stops_rotation(self):
        w = _world()
        w.meta.text_freeze = 1.0
        rotation.update(w, 0.016)
        n = w.sim.active_count
        np.testing.assert_allclose(
            w.sim.world_position[:n], w.sim.animated[:n], atol=1e-12,
            err_msg="freeze=1 должен полностью убрать вращение",
        )

    def test_freeze_0_rotates(self):
        w = _world()
        w.meta.text_freeze = 0.0
        rotation.update(w, 0.016)
        n = w.sim.active_count
        diff = np.abs(w.sim.world_position[:n] - w.sim.animated[:n]).max()
        self.assertGreater(diff, 0.01, "freeze=0 — куб должен крутиться")

    def test_freeze_monotonic(self):
        """Больше freeze → ближе к неповёрнутому виду (для точного реверса)."""
        dists = []
        for f in (0.0, 0.25, 0.5, 0.75, 1.0):
            w = _world()
            w.meta.text_freeze = f
            rotation.update(w, 0.016)
            n = w.sim.active_count
            d = float(np.abs(w.sim.world_position[:n] - w.sim.animated[:n]).mean())
            dists.append(d)
        for a, b in zip(dists, dists[1:]):
            self.assertLessEqual(b, a + 1e-12, f"не монотонно: {dists}")

    def test_rotation_speed_untouched(self):
        """avatar_text не должен менять rotation_speed в config."""
        w = _world()
        w.meta.speak_buffer = 'ТЕСТ'
        sysm = AvatarTextSystem()
        sysm.update(w, 0.016)
        self.assertEqual(
            w.meta.config['rotation_speed'], 0.5,
            "rotation_speed изменён — будет скачок на t*Δspeed рад",
        )


class TestAnimationFreeze(unittest.TestCase):

    def test_freeze_stops_wobble(self):
        w = _world()
        w.meta.text_freeze = 1.0
        animation.update(w, 0.016)
        n = w.sim.active_count
        np.testing.assert_allclose(
            w.sim.animated[:n], w.sim.morphed[:n], atol=1e-12,
            err_msg="freeze=1 должен убрать колебания частиц",
        )

    def test_no_freeze_wobbles(self):
        w = _world()
        w.meta.text_freeze = 0.0
        animation.update(w, 0.016)
        n = w.sim.active_count
        diff = np.abs(w.sim.animated[:n] - w.sim.morphed[:n]).max()
        self.assertGreater(diff, 0.001)


class TestColorFreeze(unittest.TestCase):

    def test_colors_follow_current_position(self):
        """При freeze=1 цвет = градиент от текущей позиции (плавный вдоль надписи)."""
        w = _world()
        n = w.sim.active_count
        # Частицы улетели в плоскость текста, старые цвета остались от куба
        text_pos = np.zeros((n, 3))
        text_pos[:, 0] = np.linspace(-1, 1, n)
        w.sim.world_position[:n] = text_pos
        w.meta.text_freeze = 1.0
        color.update(w, 0.016)

        rgb = w.render.final_rgb[:n].astype(np.float64)
        expected_r = (text_pos[:, 0] + 1.0) * 127.5
        # R-канал должен монотонно расти вдоль X — это и есть «плавный» градиент
        np.testing.assert_allclose(rgb[:, 0], expected_r, atol=2.0)

    def test_flat_text_not_dim(self):
        """Плоская надпись не должна быть тусклой (depth_factor→1 при freeze=1)."""
        w = _world()
        n = w.sim.active_count
        w.sim.world_position[:n] = 0.0
        w.sim.world_position[:n, 0] = 1.0  # max по X → R=255
        w.meta.text_freeze = 1.0
        color.update(w, 0.016)
        r_max = int(w.render.final_rgb[:n, 0].max())
        self.assertGreater(r_max, 240, f"текст тусклый: R={r_max}")

    def test_no_freeze_keeps_old_behaviour(self):
        """freeze=0 → цвета ровно как раньше (базовые × depth)."""
        w = _world()
        n = w.sim.active_count
        w.meta.text_freeze = 0.0
        color.update(w, 0.016)
        pz = w.sim.world_position[:n, 2]
        depth = 0.6 + 0.4 * (pz + 1.0) / 2.0
        expected = np.clip(
            w.sim.color[:n, :3].astype(np.float64) * depth[:, None], 0, 255,
        ).astype(np.uint8)
        np.testing.assert_array_equal(w.render.final_rgb[:n], expected)


class TestFreezeMirrorsMorph(unittest.TestCase):

    def test_freeze_tracks_morph_progress(self):
        """text_freeze — функция ТОЛЬКО от morph_progress → реверс точный."""
        sysm = AvatarTextSystem()
        w = _world()
        seen = {}
        for p in (0.0, 0.2, 0.5, 0.8, 1.0):
            sysm._sync_freeze(w, p)
            seen[p] = w.meta.text_freeze
        # Обратный проход должен дать те же значения
        for p in (1.0, 0.8, 0.5, 0.2, 0.0):
            sysm._sync_freeze(w, p)
            self.assertAlmostEqual(
                w.meta.text_freeze, seen[p], places=12,
                msg=f"morph_out не реверс morph_in при p={p}",
            )

    def test_freeze_reset_on_finish(self):
        w = _world()
        sysm = AvatarTextSystem()
        w.meta.speak_buffer = 'ТЕСТ'
        sysm.update(w, 0.016)          # старт
        for _ in range(400):           # прогон всего цикла
            sysm.update(w, 0.05)
            if sysm.state == 'idle':
                break
        self.assertEqual(sysm.state, 'idle')
        self.assertEqual(w.meta.text_freeze, 0.0, "freeze не сброшен")
        self.assertFalse(w.meta.text_mode)

    def test_freeze_disabled_by_setting(self):
        w = _world()
        w.meta.config['text_freeze_spin'] = False
        w.meta.speak_buffer = 'ТЕСТ'
        sysm = AvatarTextSystem()
        sysm.update(w, 0.016)
        sysm.update(w, 0.1)
        self.assertEqual(
            w.meta.text_freeze, 0.0,
            "text_freeze_spin=False → заморозки быть не должно",
        )


if __name__ == '__main__':
    unittest.main(verbosity=2)
