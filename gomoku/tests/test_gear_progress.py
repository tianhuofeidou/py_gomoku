"""五档棋型积分：独立小棋型、分支回滚及带上下文的缓存复用。"""
import copy
import unittest
from unittest.mock import patch

from gomoku.core.engine import Engine
from gomoku.core.pattern import PatternAnalyzer
from gomoku.core.search import Search, ORDINARY_GEAR_PROFILE, GEAR_THREAT_POINTS
from gomoku.core.deep_search import DeepSearch
from gomoku.core.utils import BLACK, WHITE, empty_board


SHAPES = {
    'rush': ([(7, c, BLACK) for c in (5, 6, 7, 8)] + [(7, 4, WHITE)], 30),
    'three': ([(7, c, BLACK) for c in (6, 7, 8)], 40),
    '33': ([(7, c, BLACK) for c in (6, 8)] + [(r, 7, BLACK) for r in (6, 8)], 60),
    '43': ([(7, c, BLACK) for c in (5, 6, 8)] + [(r, 7, BLACK) for r in (6, 8)]
           + [(7, 4, WHITE)], 80),
    '44': ([(7, c, BLACK) for c in (5, 6, 8)] + [(r, 7, BLACK) for r in (5, 6, 8)]
           + [(7, 4, WHITE), (4, 7, WHITE)], 100),
}


def build(moves):
    board = empty_board()
    for r, c, player in moves:
        board[r][c] = player
    search = Search(PatternAnalyzer())
    search.rebuild_all(board)
    return board, search


class GearProgressTests(unittest.TestCase):
    def test_values_and_each_shape_without_component_double_count(self):
        self.assertEqual(GEAR_THREAT_POINTS, {0: 30, 1: 40, 2: 100, 3: 80, 4: 60})
        for name, (moves, expected) in SHAPES.items():
            with self.subTest(name=name):
                board, search = build(moves)
                events = search._gear_threats(board, BLACK)
                self.assertEqual(list(events.values()), [expected])

    def test_rotation_and_black_white_swap(self):
        for moves, expected in SHAPES.values():
            for swapped in (False, True):
                for turns in range(4):
                    changed = []
                    for r, c, player in moves:
                        for _ in range(turns):
                            r, c = c, 14-r
                        changed.append((r, c, 3-player if swapped else player))
                    board, search = build(changed)
                    self.assertEqual(sum(search._gear_threats(board, WHITE if swapped else BLACK).values()),
                                     expected)

    def test_existing_live_four_and_five_are_not_progress(self):
        for count in (4, 5):
            board, search = build([(7, c, BLACK) for c in range(5, 5+count)])
            self.assertEqual(search._gear_threats(board, BLACK), {})

    def test_rebuild_has_no_retroactive_points_and_active_is_not_recounted(self):
        board, search = build(SHAPES['three'][0])
        self.assertEqual(search._gear_context(), (3, 3, 0, 0))
        search._record_gear_threats(board, BLACK)
        search._record_gear_threats(board, BLACK)
        self.assertEqual(search._gear_context(), (3, 3, 0, 0))

    def test_threshold_remainder_opponent_and_no_accumulation_at_limits(self):
        _, search = build([])
        search._advance_gear_progress(BLACK, 80)
        search._advance_gear_progress(BLACK, 60)
        self.assertEqual(search._gear_context(), (4, 2, 40, -40))
        search._advance_gear_progress(WHITE, 80)
        self.assertEqual(search._gear_context(), (4, 2, -40, 40))
        search._advance_gear_progress(WHITE, 60)
        self.assertEqual(search._gear_context(), (3, 3, 0, 0))
        search._advance_gear_progress(BLACK, 500)
        self.assertEqual(search._gear_context(), (5, 1, 0, 0))
        search._advance_gear_progress(WHITE, 500)
        self.assertEqual(search._gear_context(), (1, 5, 0, 0))

    def test_real_move_only_credits_mover_new_threat(self):
        board, search = build([(7, 6, BLACK), (7, 7, BLACK)])
        board[7][8] = BLACK
        search.on_move(board, 7, 8, BLACK)
        self.assertEqual(search._gear_context(), (3, 3, 40, -40))
        board[0][0] = WHITE
        search.on_move(board, 0, 0, WHITE)
        self.assertEqual(search._gear_context(), (3, 3, 40, -40))
        self.assertEqual(search._ordinary_gear(BLACK), 3)

    def test_nested_journal_restores_siblings_and_all_state(self):
        board, search = build([(7, 6, BLACK), (7, 7, BLACK)])
        before = copy.deepcopy(search.__dict__)
        first = {}
        board[7][8] = BLACK
        search.on_move(board, 7, 8, BLACK, journal=first, record_history=False)
        context = search._gear_context()
        second = {}
        board[0][0] = WHITE
        search.on_move(board, 0, 0, WHITE, journal=second, record_history=False)
        search.restore(second)
        board[0][0] = 0
        self.assertEqual(search._gear_context(), context)
        search.restore(first)
        board[7][8] = 0
        for key in ('sb', 'sw', '_direction_levels', 'gear', 'gear_progress', '_gear_active',
                    'zero_history', 'zero_active'):
            self.assertEqual(search.__dict__[key], before[key], key)

    def test_all_five_quotas_both_sides(self):
        board, search = build([])
        for gear, quota in ORDINARY_GEAR_PROFILE.items():
            search.gear = {BLACK: gear, WHITE: 6-gear}
            with patch.object(search, '_fallback_by_count', return_value=[]) as fallback:
                search._ordinary_candidates(board, BLACK, True)
                fallback.assert_called_with(board, *quota, BLACK, True)
                search._ordinary_candidates(board, WHITE, True)
                fallback.assert_called_with(board, *ORDINARY_GEAR_PROFILE[6-gear], WHITE, True)

    def test_zero_quota_really_has_no_points(self):
        board, search = build([(6, 6, BLACK), (8, 7, WHITE), (7, 9, BLACK), (9, 8, WHITE)])
        self.assertEqual(search._fallback_by_count(board, 0, 0, BLACK, True), [])
        for quota in ((0, 4), (4, 0)):
            points = search._fallback_by_count(board, *quota, BLACK, True)
            self.assertLessEqual(len(points), 4)
            self.assertEqual(len({(r, c) for r, c, w in points}), len(points))
            self.assertTrue(all(board[r][c] == 0 for r, c, w in points))

    def test_context_separates_same_board_candidate_and_result_caches(self):
        board, search = build([])
        deep = DeepSearch(search)
        deep._begin_search_cache(board, False)
        try:
            with patch.object(deep, '_tactical_result', return_value=(None, {})), \
                    patch.object(search, '_fallback_by_count', side_effect=lambda b, a, d, p, w:
                                 [(a, d, 50.0)]):
                old_key = deep._result_key(BLACK, BLACK, 10, 1)
                first = deep._candidate_points(board, BLACK)
                search._advance_gear_progress(BLACK, 100)
                second = deep._candidate_points(board, BLACK)
                self.assertNotEqual(first, second)
                self.assertNotEqual(old_key, deep._result_key(BLACK, BLACK, 10, 1))
                self.assertEqual(deep._candidate_points(board, BLACK), second)
                self.assertEqual(len(deep._tree_root['candidates']), 2)
        finally:
            deep._reuse_active = False

    def test_real_reroot_inherits_simulated_progress_and_rejects_mismatch(self):
        engine = Engine()
        board, _ = build([(7, 6, BLACK), (7, 7, BLACK)])
        engine.search.rebuild_all(board)
        deep = engine.deep_search
        deep._begin_search_cache(board, False)
        journal = {}
        board[7][8] = BLACK
        engine.search.on_move(board, 7, 8, BLACK, journal=journal, record_history=False)
        child = deep._remember_position(board, [(7, 8, BLACK, 50)])
        predicted = child['gear_context']
        engine.search.restore(journal)
        board[7][8] = 0
        deep._reuse_active = False
        board[7][8] = BLACK
        engine.on_move(board, 7, 8, BLACK)
        self.assertIs(deep._tree_root, child)
        self.assertEqual(engine.search._gear_context(), predicted)
        child['gear_context'] = (1, 5, 0, 0)
        deep._begin_search_cache(board, False)
        self.assertIsNot(deep._tree_root, child)

    def test_policy_points_invalidate_signature_and_leaf_formula_stays_separate(self):
        board, search = build([(6, 6, BLACK), (8, 7, WHITE)])
        deep = DeepSearch(search)
        signature = deep._reuse_signature()
        score = deep._stale_result_score(board, BLACK)
        search._advance_gear_progress(BLACK, 100)
        self.assertEqual(deep._stale_result_score(board, BLACK), score)
        with patch.dict(GEAR_THREAT_POINTS, {2: 101}):
            self.assertNotEqual(deep._reuse_signature(), signature)


if __name__ == '__main__':
    unittest.main()
