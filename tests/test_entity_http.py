"""Loopback HTTP checks for the allowlisted Hermes Entity control endpoint."""
from __future__ import annotations

import json
import queue
import threading
import unittest
from http.server import HTTPServer
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from api_server import CubeAPIHandler
from core.entity_controller import apply_command
from core.world import World


class EntityHttpTests(unittest.TestCase):
    def setUp(self):
        self.world = World.create({'shape_preset': 'face', 'face_theme': 'Holographic'})
        self.engine = SimpleNamespace(world=self.world, monitor=SimpleNamespace(fps=0.0),
                                      _entity_queue=queue.Queue(maxsize=8))
        class LocalHandler(CubeAPIHandler):
            engine = self.engine
        self.server = HTTPServer(('127.0.0.1', 0), LocalHandler)
        self.url = f'http://127.0.0.1:{self.server.server_port}'
        self.worker = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.worker.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.worker.join(timeout=1)

    def post(self, data, headers=None):
        request = Request(self.url+'/api/entity',
                          data=json.dumps(data).encode('utf-8'),
                          headers={'Content-Type':'application/json', **(headers or {})},
                          method='POST')
        try:
            with urlopen(request, timeout=2) as response:
                return response.status, json.load(response)
        except HTTPError as response:
            return response.code, json.load(response)

    def test_queued_command_visible_in_status_once_applied(self):
        status, response = self.post({'action':'look', 'x':.4, 'y':-.3})
        self.assertEqual(status, 202)
        self.assertTrue(response['queued'])
        self.assertEqual(self.engine._entity_queue.qsize(), 1)
        apply_command(self.world, self.engine._entity_queue.get_nowait())
        with urlopen(self.url+'/api/status',timeout=2) as response:
            status_data = json.load(response)
        self.assertEqual(status_data['entity']['look'], {'x':.4, 'y':-.3})
        self.assertEqual(status_data['entity']['shape'], 'face')

    def test_reject_cross_origin_and_unknown_command(self):
        status, _ = self.post({'action':'reset'}, headers={'Origin':'https://example.com'})
        self.assertEqual(status, 403)
        status, _ = self.post({'action':'python', 'value':'os.system'})
        self.assertEqual(status, 400)
        self.assertTrue(self.engine._entity_queue.empty())

    def test_reject_wrong_media_type(self):
        status, _ = self.post({'action':'reset'}, headers={'Content-Type':'text/plain'})
        self.assertEqual(status, 415)
        self.assertTrue(self.engine._entity_queue.empty())


if __name__ == '__main__':
    unittest.main()
