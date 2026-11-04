# -*- coding: utf-8 -*-
"""显式开启的战术梯度与负温8手实验延伸定向测试，非正式默认。"""
import copy
import unittest
from contextlib import ExitStack
from unittest.mock import patch

from gomoku.core.engine import Engine
from gomoku.core.utils import BLACK, WHITE, empty_board


def snapshot(engine, board):
    s = engine.search
    return copy.deepcopy((board, engine.moves, s.sb, s.sw, s._direction_levels,
                          s.zero_history, s.zero_active, s.gear))


class Observer:
    def __init__(self):
        self.events = []

    def record(self, kind, node=None, **values):
        self.events.append((kind, values))


class TacticalTemperatureGradientTest(unittest.TestCase):
    POINTS = [(1, 1), (1, 3), (1, 5), (1, 7), (1, 9), (3, 1), (3, 3),
              (3, 5), (3, 7), (3, 9), (5, 1), (5, 3), (5, 5), (5, 7),
              (5, 9), (7, 1), (7, 3), (7, 5), (7, 7), (7, 9)]

    def test_gradient_boundaries_and_ordinary_is_unchanged(self):
        d = Engine().deep_search
        d.TACTICAL_FREE_PLIES = 8
        expected = {0: 0., 7: 0., 8: 2., 9: 2., 15: 2., 16: 4.,
                    23: 4., 24: 8., 25: 8., 999: 8.}
        for count, cost in expected.items():
            self.assertEqual(d._path_temperature_cost(100., True, count, 160.), cost)
            self.assertEqual(d._path_temperature_cost(100., True, count, 0.), 0.)
        self.assertEqual(d._path_temperature_cost(60., False, 8, 160.), d._dt(60.))
        self.assertEqual(d._path_temperature_cost(60., False, 8, 0.), d._dt(60.))
        self.assertEqual(d._advance_tactical_count(5, False), 5)
        self.assertEqual(d._advance_tactical_count(5, True), 6)

    def chain(self, player, temperature, win_at=None):
        engine = Engine()
        d = engine.deep_search
        d.T0 = temperature
        d.TACTICAL_FREE_PLIES = 8
        d.TACTICAL_EXTENSION_PLIES = 8
        board = empty_board()
        observer = Observer()
        d._debug_observer = observer

        def count(b):
            return sum(value != 0 for row in b for value in row)

        def tactical(b, side):
            index = count(b)
            if index >= len(self.POINTS):
                return None, {}
            return 'my-vcf', {self.POINTS[index]: 100.}

        def candidates(b, side, n=5, plies=0):
            return [(r, c, weight) for (r, c), weight in tactical(b, side)[1].items()]

        before = snapshot(engine, board)
        with ExitStack() as stack:
            stack.enter_context(patch.object(d, '_tactical_result', side_effect=tactical))
            stack.enter_context(patch.object(d, '_candidate_points', side_effect=candidates))
            stack.enter_context(patch.object(d, '_active_threat_points', return_value=frozenset()))
            stack.enter_context(patch.object(d, '_check_win',
                                             side_effect=lambda b, r, c, p: count(b) == win_at))
            rows = d.rank_candidates(board, {self.POINTS[0]: 100.}, player,
                                     complete_scores=True)
            self.assertEqual(snapshot(engine, board), before)
        steps = [values for kind, values in observer.events if kind == 'temperature']
        return d, rows, steps, observer

    def test_positive_chain_is_charged_at_9_17_25_boundaries(self):
        d, rows, steps, _ = self.chain(BLACK, 160.)
        self.assertEqual(len(steps), 20)
        costs = [step['temperature_before'] - step['temperature'] for step in steps]
        self.assertEqual(costs[:8], [0.] * 8)
        self.assertEqual(costs[8:16], [2.] * 8)
        self.assertEqual(costs[16:20], [4.] * 4)
        self.assertEqual(steps[8]['temperature_before'], 160.)
        self.assertEqual(steps[8]['temperature'], 158.)
        self.assertEqual(steps[16]['temperature'], 140.)
        self.assertEqual(steps[19]['temperature'], 128.)
        self.assertTrue(all(step['extension_state'] == (0, 0, 0) for step in steps))
        self.assertEqual(rows[0]['state'], 0)

    def test_positive_crossing_zero_is_not_extension_and_next_eight_are_free(self):
        d, rows, steps, _ = self.chain(BLACK, 3.)
        self.assertEqual(len(steps), 18)
        self.assertGreater(steps[9]['temperature_before'], 0)
        self.assertLessEqual(steps[9]['temperature'], 0)
        self.assertEqual(steps[9]['extension_state'], (0, 0, 0))
        self.assertEqual(steps[10]['temperature'], steps[10]['temperature_before'])
        self.assertEqual([step['extension_state'][0] for step in steps[10:18]],
                         list(range(1, 9)))
        self.assertTrue(all(step['temperature_before'] <= 0 for step in steps[10:]))
        self.assertEqual(rows[0]['state'], 0)

    def test_eighth_extension_win_has_priority(self):
        for player in (BLACK, WHITE):
            d, rows, steps, observer = self.chain(player, 0., win_at=8)
            self.assertEqual(len(steps), 8)
            self.assertEqual(rows[0]['state'], -1)
            self.assertEqual(rows[0]['score'], -1000)
            self.assertFalse(any(kind == 'cutoff' and values.get('leaf_end_reason') ==
                                 'tactical_extension'
                                 for kind, values in observer.events))

    def test_eighth_extension_without_win_cuts_and_does_not_issue_ninth(self):
        for player in (BLACK, WHITE):
            d, rows, steps, observer = self.chain(player, 0.)
            self.assertEqual(len(steps), 8)
            self.assertEqual([step['extension_state'][0] for step in steps],
                             list(range(1, 9)))
            self.assertEqual(rows[0]['state'], 0)
            cuts = [values for kind, values in observer.events if kind == 'cutoff']
            self.assertEqual(cuts[-1]['leaf_end_reason'], 'tactical_extension')
            self.assertEqual(d.cache_stats['tactical_probe_moves'], 0)

    def test_threat_score_range_and_perspective_remain_unchanged(self):
        d = Engine().deep_search
        for black, white, expected in ((2, 1, 200), (4, 0, 800), (8, 0, 800),
                                       (0, 7, -800), (3, 3, 0)):
            context = [{'extension_state': (8, black, white)}]
            self.assertEqual(d._extension_score(context, None, BLACK), expected)
            self.assertEqual(d._extension_score(context, None, WHITE), -expected)


if __name__ == '__main__':
    unittest.main()
