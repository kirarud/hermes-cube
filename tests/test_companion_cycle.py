#!/usr/bin/env python3
"""Smoke tests: mood cycle, resize_pool, offline reply, memory parse."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.world import World
from core.systems.mood import MoodSystem
from core.systems.ai import offline_reply
from core import companion_memory as mem
import core.systems.grid_generator as gg


class TestMoodCycle(unittest.TestCase):
    def test_json_sets_mood_config_and_speak_buffer(self) -> None:
        world = World.create({'response_mode': 'both', 'particle_density': 8}, n_particles=8 * 8 * 6)
        mood = MoodSystem()
        raw = json.dumps(
            {"mood": "happy", "text": "Привет, друг!", "color_hue": 0.2},
            ensure_ascii=False,
        )
        world.meta.ai_response = raw
        mood.update(world, 0.016)

        self.assertEqual(world.meta.mood, 'happy')
        self.assertEqual(world.meta.ai_response, 'Привет, друг!')
        self.assertEqual(world.meta.speak_buffer, 'Привет, друг!')
        self.assertEqual(world.meta.config.get('pulse_rate'), 2.8)
        self.assertEqual(world.meta.config.get('pulse_amplitude'), 0.22)
        self.assertEqual(world.meta.config.get('rotation_speed'), 0.50)

        # second frame: no re-process
        world.meta.speak_buffer = ''
        mood.update(world, 0.016)
        self.assertEqual(world.meta.speak_buffer, '')

    def test_avatar_only_mode(self) -> None:
        world = World.create({'response_mode': 'avatar'}, n_particles=64)
        mood = MoodSystem()
        world.meta.ai_response = json.dumps(
            {"mood": "sad", "text": "Тихо", "color_hue": 0.5},
            ensure_ascii=False,
        )
        mood.update(world, 0.016)
        self.assertEqual(world.meta.ai_response, '')
        self.assertEqual(world.meta.speak_buffer, 'Тихо')
        self.assertEqual(world.meta.mood, 'sad')


class TestResizePool(unittest.TestCase):
    def test_grow_and_shrink(self) -> None:
        world = World.create({'particle_density': 6}, n_particles=6 * 6 * 6)
        self.assertEqual(world.sim.active_count, 216)

        world.meta.config['particle_density'] = 14
        gg.update(world, 0.016)
        expected = 14 * 14 * 6
        self.assertEqual(world.sim.active_count, expected)
        self.assertEqual(len(world.render.projected_x), expected)
        self.assertEqual(len(world.sim.base_position), world.sim.pool_size)
        self.assertGreaterEqual(world.sim.pool_size, expected)

        world.meta.config['particle_density'] = 8
        gg.update(world, 0.016)
        expected2 = 8 * 8 * 6
        self.assertEqual(world.sim.active_count, expected2)
        self.assertEqual(len(world.render.projected_x), expected2)


class TestOfflineAndMemory(unittest.TestCase):
    def test_offline_reply_json(self) -> None:
        raw = offline_reply('привет')
        data = json.loads(raw)
        self.assertIn('text', data)
        self.assertIn(data['mood'], ('happy', 'sad', 'thinking', 'speaking', 'idle'))

    def test_memory_commands(self) -> None:
        self.assertEqual(mem.parse_memory_command('запомни: люблю чай')[0], 'remember')
        self.assertEqual(mem.parse_memory_command('забудь чай')[0], 'forget')
        self.assertIsNone(mem.parse_memory_command('просто привет'))

    def test_memory_file_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'memory.md')
            # monkeypatch path
            old = mem.MEMORY_PATH
            old_dir = mem._MEMORY_DIR
            try:
                mem.MEMORY_PATH = path
                mem._MEMORY_DIR = td
                mem.ensure_memory_file()
                self.assertTrue(mem.append_fact('тест-факт-xyz'))
                body = mem.load_memory()
                self.assertIn('тест-факт-xyz', body)
                n = mem.forget_fragment('xyz')
                self.assertGreaterEqual(n, 1)
            finally:
                mem.MEMORY_PATH = old
                mem._MEMORY_DIR = old_dir


if __name__ == '__main__':
    unittest.main()
