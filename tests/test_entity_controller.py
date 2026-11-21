"""Hermes Entity alpha tests: commands, rendering and safe local queue."""
from __future__ import annotations

import math
import queue
import unittest
from types import SimpleNamespace

import numpy as np

from api_server import CubeAPIHandler
from core.entity_controller import (
    EntityCommandError, MAX_PITCH, MAX_YAW,
    apply_command, entity_state, validate_command,
)
from core.pipeline import build_default_pipeline
from core.world import World


class EntityControllerTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {
            'shape_preset': 'face', 'face_theme': 'Holographic',
            'morph_progress': 1.0, 'particle_density': 12,
            'particle_mode': 'off', 'rotation_speed': 0,
            'wave_amp': 0, 'pulse_amplitude': 0,
            'pulse_rate': 1.8, 'cell_size': 5,
            'fit_to_window': True, 'color_mode': 'default',
        }
        self.world = World.create(self.cfg)
        self.world.meta.t = 0
        self.world.meta.w = self.world.meta.h = 650

    def test_shape_changes_without_replacing_the_world(self):
        instance = self.world
        ai_history = self.world.meta.chat_history
        for shape in ('cloud', 'cube', 'face'):
            state = apply_command(self.world, {'action': 'shape', 'value': shape})
            self.assertIs(instance, self.world)
            self.assertIs(ai_history, self.world.meta.chat_history)
            self.assertEqual(state['shape'], shape)
            self.assertEqual(self.cfg['morph_progress'], 0.0 if shape == 'cube' else 1.0)
            build_default_pipeline().run(self.world, .016)
            self.assertTrue(np.isfinite(self.world.sim.world_position).all())

    def test_skin_and_mood_preserve_identity(self):
        state = apply_command(self.world, {'action': 'skin', 'value': 'Retro Hacker'})
        self.assertEqual(state['skin'], 'Retro Hacker')
        self.assertEqual(self.cfg['symbol'], 'square')
        self.assertEqual(self.cfg['shape_preset'], 'face')
        apply_command(self.world, {'action': 'mood', 'value': 'happy'})
        self.assertEqual(entity_state(self.world)['mood'], 'happy')
        self.assertTrue(self.world.meta.mood_lock)

    def test_joystick_changes_real_particle_orientation(self):
        pipeline = build_default_pipeline()
        pipeline.run(self.world, .016)
        original = self.world.sim.world_position[:self.world.sim.active_count].copy()
        apply_command(self.world, {'action': 'look', 'x': 1, 'y': -.5})
        for _ in range(32):
            pipeline.run(self.world, .016)
        self.assertGreater(float(np.abs(self.world.sim.world_position[:len(original)] - original).max()), .1)
        self.assertAlmostEqual(self.world.meta.entity_yaw, MAX_YAW, places=2)
        self.assertAlmostEqual(self.world.meta.entity_pitch, -.5 * MAX_PITCH, places=2)
        self.assertEqual(entity_state(self.world)['look'], {'x': 1.0, 'y': -.5})
        apply_command(self.world, {'action': 'reset'})
        self.assertEqual(self.world.meta.entity_yaw, 0.0)
        self.assertEqual(self.world.meta.entity_target_yaw, 0.0)

    def test_invalid_commands_do_not_mutate_world(self):
        before = dict(self.cfg)
        errors = (
            None, [], {}, {'action': 'hack', 'value': 'anything'},
            {'action': 'shape', 'value': '../model.glb'},
            {'action': 'skin', 'value': 'not a theme'},
            {'action': 'look', 'x': True, 'y': 0},
            {'action': 'look', 'x': float('nan'), 'y': 0},
            {'action': 'look', 'x': float('inf'), 'y': 0},
            {'action': 'look', 'x': 2, 'y': 0},
            {'action': 'look', 'x': 0},
            {'action': 'reset', 'extra': 'run'},
            {'action': 'mood', 'value': 'execute'},
        )
        for bad in errors:
            with self.subTest(command=bad):
                with self.assertRaises(EntityCommandError):
                    validate_command(bad)
        self.assertEqual(self.cfg, before)
        self.assertEqual(self.world.meta.entity_yaw, 0)

    def test_handler_queues_validated_commands_only(self):
        handler = object.__new__(CubeAPIHandler)
        handler.engine = SimpleNamespace(_entity_queue=queue.Queue(maxsize=1))
        results = []
        handler._send = lambda code, data: results.append((code, data))
        handler._handle_entity({'action': 'look', 'x': .2, 'y': -.3})
        self.assertEqual(results[-1][0], 202)
        self.assertEqual(handler.engine._entity_queue.qsize(), 1)
        handler._handle_entity({'action': 'look', 'x': .1, 'y': .1})
        self.assertEqual(results[-1][0], 429)
        handler._handle_entity({'action': 'look', 'x': 99, 'y': 0})
        self.assertEqual(results[-1][0], 400)
        self.assertEqual(handler.engine._entity_queue.qsize(), 1)


if __name__ == '__main__':
    unittest.main()
