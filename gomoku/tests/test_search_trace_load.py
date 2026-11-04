# -*- coding: utf-8 -*-
"""搜索录像读取入口的定向测试。"""
import json
import tempfile
import unittest
from pathlib import Path

from gomoku.tools.search_trace import TraceModel, load_trace


def recording(events, schema='gomoku-search-trace-v1', partial=False):
    return {'schema': schema, 'events': events, 'partial': partial}


BASE = [0] * 225


class LoadTraceTest(unittest.TestCase):
    def write(self, payload):
        f = tempfile.NamedTemporaryFile('w', suffix='.json', delete=False, encoding='utf-8')
        json.dump(payload, f)
        f.close()
        self.addCleanup(lambda: Path(f.name).unlink(missing_ok=True))
        return f.name

    def test_loads_valid_recording_without_search(self):
        events = [
            {'seq': 1, 'kind': 'round', 'node': None, 'round': 1,
             'moves': [[7, 7, 1], [7, 8, 2]], 'turn': 1, 't0': 160.0},
            {'seq': 2, 'kind': 'node', 'node': 'n1', 'round': 1,
             'parent': None, 'move': None, 'origin': 'search',
             'board': BASE, 'base_moves': []},
            {'seq': 3, 'kind': 'root', 'node': 'n1', 'round': 1},
        ]
        path = self.write(recording(events, partial=True))
        model, partial, moves = load_trace(path)
        self.assertIsInstance(model, TraceModel)
        self.assertTrue(partial)
        self.assertEqual(moves, [(7, 7, 1), (7, 8, 2)])
        self.assertEqual(len(model.events), 3)
        self.assertEqual(model.root(), 'n1')

    def test_rejects_wrong_schema_and_bad_sequence(self):
        events = [{'seq': 2, 'kind': 'round', 'node': None, 'round': 1,
                   'moves': [], 'turn': 1, 't0': 160.0}]
        with self.assertRaises(ValueError):
            load_trace(self.write(recording(events, schema='wrong')))
        with self.assertRaises(ValueError):
            load_trace(self.write(recording(events)))

    def test_rejects_missing_round(self):
        events = [{'seq': 1, 'kind': 'node', 'node': 'n1', 'parent': None,
                   'move': None, 'origin': 'search', 'board': BASE, 'base_moves': []}]
        with self.assertRaises(ValueError):
            load_trace(self.write(recording(events)))

    def test_rejects_empty_root_compatibility(self):
        events = [
            {'seq': 1, 'kind': 'round', 'node': None, 'round': 1,
             'moves': [], 'turn': 1, 't0': 160.0},
            {'seq': 2, 'kind': 'node', 'node': 'n1', 'round': 1,
             'parent': None, 'move': None, 'origin': 'search',
             'board': BASE, 'base_moves': []},
            {'seq': 3, 'kind': 'root', 'node': 'n1', 'round': 1},
            {'seq': 4, 'kind': 'node', 'node': 'n2', 'round': 1,
             'parent': None, 'move': None, 'origin': 'search',
             'board': BASE, 'base_moves': []},
            {'seq': 5, 'kind': 'node', 'node': 'n3', 'round': 1,
             'parent': 'n2', 'move': [7, 7, 1], 'origin': 'search'},
        ]
        with self.assertRaises(ValueError):
            load_trace(self.write(recording(events)))


if __name__ == '__main__':
    unittest.main()
