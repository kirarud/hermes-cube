"""Regression tests for Hermes' local Responses API adapter."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.systems.ai import _extract_responses_text


class TestResponsesBackend(unittest.TestCase):
    def test_extracts_completed_text_from_sse(self) -> None:
        sse = (
            'event: response.output_text.delta\n'
            'data: {"type":"response.output_text.delta","delta":"св"}\n\n'
            'event: response.output_text.delta\n'
            'data: {"type":"response.output_text.delta","delta":"язь"}\n\n'
            'event: response.completed\n'
            'data: {"type":"response.completed","response":{"status":"completed"}}\n\n'
        )
        self.assertEqual(_extract_responses_text(sse), 'связь')


if __name__ == '__main__':
    unittest.main()
