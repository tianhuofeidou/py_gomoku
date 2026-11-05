"""普通棋型扣温与综合分分离；小场景验证各落子及缓存建边入口。"""
import copy
import unittest
from unittest.mock import patch

from gomoku.core.deep_search import DeepSearch
from gomoku.core.engine import Engine
from gomoku.core import search as S
from gomoku.core.utils import BLACK, WHITE, empty_board


class Observer:
    def __init__(self):
        self.steps = []

    def record(self, kind, node=None, **data):
        if kind == 'temperature':
            self.steps.append(data)


class PatternTemperatureTests(unittest.TestCase):
    def test_all_ordinary_subclasses_use_only_type_score(self):
        d = Engine().deep_search
        for player in (BLACK, WHITE):
            for sub, raw in S.TYPE_SCORE.items():
                with self.subTest(player=player, sub=sub), patch.object(d, '_sub_at', return_value=sub):
                    self.assertAlmostEqual(d._pattern_temperature_weight(player, 7, 7), raw/260.*100.)
            with patch.object(d, '_sub_at', return_value=17):
                self.assertEqual(d._pattern_temperature_weight(player, 7, 7), 0.)
                self.assertEqual(d._dt(0), 203.)

    def test_defense_uses_actual_mover_shape(self):
        d = Engine().deep_search
        with patch.object(d, '_sub_at', side_effect=lambda player,*args: 17 if player==WHITE else 10):
            self.assertEqual(d._pattern_temperature_weight(WHITE, 7, 7), 0.)
            self.assertAlmostEqual(d._pattern_temperature_weight(BLACK, 7, 7), 170/260*100.)

    def test_cache_edges_preserve_composite_weight_but_not_its_temperature(self):
        for player in (BLACK, WHITE):
            for composite in (10., 90.):
                d = Engine().deep_search
                with patch.object(d, '_tactical_result', return_value=(None, {})), \
                        patch.object(d, '_sub_at', return_value=11):
                    board = empty_board()
                    d._begin_search_cache(board, False)
                    try:
                        d._cache_edges(board, d._tree_root, player, [(1., 7, 7, composite)], 120., player)
                        edge = d._tree_root['edges'][0]
                        self.assertEqual(edge[:4], (7, 7, player, composite))
                        self.assertEqual(edge[4], 67.)
                        self.assertEqual(edge[5], 1.)
                    finally:
                        d._reuse_active = False

    def test_root_recursive_and_reply_read_type_before_move_and_rollback(self):
        for player in (BLACK, WHITE):
            engine = Engine()
            d = engine.deep_search
            d.T0 = 1.
            board = empty_board()
            points = [(0, 0), (2, 2), (4, 4)]
            before = copy.deepcopy((board, engine.search.sb, engine.search.sw,
                engine.search._direction_levels, engine.search.zero_history,
                engine.search.zero_active, engine.search.gear, engine.search.gear_progress))
            observer = Observer()
            d._debug_observer = observer

            def candidates(b, side, n=5, plies=0):
                count = sum(cell!=0 for row in b for cell in row)
                return [(*points[count], 90.)] if count<3 else []

            def subtype(side, r, c):
                self.assertEqual(board[r][c], 0, '棋型必须在落子前读取')
                return 11

            with patch.object(d, '_candidate_points', side_effect=candidates), \
                    patch.object(d, '_tactical_result', return_value=(None, {})), \
                    patch.object(d, '_sub_at', side_effect=subtype), \
                    patch.object(d, '_check_win', return_value=False):
                rows = d.rank_candidates(board, {points[0]: 90.}, player)
            self.assertTrue(rows)
            self.assertEqual(len(observer.steps), 3)
            self.assertEqual([step['weight'] for step in observer.steps], [90.]*3)
            self.assertEqual([step['cost_weight'] for step in observer.steps], [50.]*3)
            self.assertEqual([step['cost'] for step in observer.steps], [53.]*3)
            self.assertEqual([step['temperature'] for step in observer.steps], [-52., -105., -158.])
            after = (board, engine.search.sb, engine.search.sw, engine.search._direction_levels,
                     engine.search.zero_history, engine.search.zero_active,
                     engine.search.gear, engine.search.gear_progress)
            self.assertEqual(after, before)

    def test_actual_f5_shape_has_weight_fifty_and_cost_fifty_three(self):
        labels = ('H8','H7','G7','F6','I9','J10','G9','G6','H6')
        original = [(int(p[1:])-1, ord(p[0])-65, BLACK if i%2==0 else WHITE)
                    for i,p in enumerate(labels)]
        for swap in (False, True):
            moves = [(r,c,3-p if swap else p) for r,c,p in original]
            engine = Engine()
            engine.reset(moves)
            player = BLACK if swap else WHITE
            d = engine.deep_search
            self.assertEqual(d._sub_at(player, 4, 5), 11)
            self.assertEqual(d._pattern_temperature_weight(player, 4, 5), 50.)
            self.assertEqual(d._path_temperature_cost(50., False, 20, 120.), 53.)

    def test_tactical_gradient_unchanged(self):
        d = Engine().deep_search
        for count, expected in ((0,0.), (7,0.), (8,2.), (15,2.), (16,4.), (24,8.)):
            self.assertEqual(d._path_temperature_cost(100., True, count, 120.), expected)

    def test_changing_type_weight_method_invalidates_cached_tree(self):
        d = Engine().deep_search
        board = empty_board()
        d._begin_search_cache(board, False)
        old = d._tree_root
        try:
            with patch.object(d, '_pattern_temperature_weight', return_value=50.):
                d._begin_search_cache(board, False)
                self.assertIsNot(d._tree_root, old)
        finally:
            d._reuse_active = False


if __name__ == '__main__':
    unittest.main()
