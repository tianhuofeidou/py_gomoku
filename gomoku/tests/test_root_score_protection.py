"""硬态早停只能淘汰硬败；入选比较与全败兜底必须保持完整原始分。"""
import copy
import json
import unittest
from unittest.mock import patch

from gomoku.core.engine import Engine
from gomoku.core.utils import BLACK, WHITE, empty_board


class RootScoreProtectionTest(unittest.TestCase):
    def run_tree(self, player=BLACK, all_lose=False, complete=False, factor=1., engine=None):
        engine = Engine() if engine is None else engine
        deep = engine.deep_search
        deep.T0 = 1.
        board = empty_board()
        roots = {(7, 7): 100., (7, 9): 100.}
        before = copy.deepcopy((board, engine.search.sb, engine.search.sw,
                                engine.search._direction_levels, engine.search.zero_history))

        def candidates(b, turn, n=5, plies=0):
            return [(r, c, w) for (r, c), w in roots.items()] if turn == player else [
                (6, 6, 100.), (6, 8, 100.)]

        def won(b, r, c, turn):
            return turn != player and (r, c) == (6, 6) and (all_lose or b[7][9] == player)

        def leaf(b, branch, status, turn, contexts):
            if all_lose:
                # 仅看首个已证明的败路会选左边；完整累计分应选右边。
                return factor * ((-5. if b[7][7] else -20.) if status == 'lose'
                                 else (-100. if b[7][7] else 100.))
            return factor * (999. if status == 'lose' else -10.)

        with patch.object(deep, '_candidate_points', side_effect=candidates), \
                patch.object(deep, '_check_win', side_effect=won), \
                patch.object(deep, '_leaf_score_impl', side_effect=leaf) as score:
            rows = deep.rank_candidates(board, roots, player, complete_scores=complete)
        self.assertEqual(before, (board, engine.search.sb, engine.search.sw,
                                  engine.search._direction_levels, engine.search.zero_history))
        self.assertIsNone(deep._root_pass)
        self.assertEqual(engine.moves, [])
        for row in rows:
            self.assertEqual(row['state_name'], deep._state_name(row['state']))
            self.assertEqual(row['child_state'], deep._state_child_state(row['state']))
            self.assertEqual(row['forced'], deep._state_forced(row['state']))
        json.dumps(rows, allow_nan=False)
        return rows, score.call_count

    def test_discarded_hard_loss_has_no_invented_score(self):
        for player in (BLACK, WHITE):
            rows, calls = self.run_tree(player)
            self.assertEqual((rows[0]['r'], rows[0]['c']), (7, 7))
            self.assertEqual(rows[0]['score'], -20.)
            self.assertTrue(rows[0]['score_complete'])
            self.assertEqual(rows[1]['state'], -1)
            self.assertIsNone(rows[1]['score'])
            self.assertFalse(rows[1]['score_complete'])
            self.assertEqual(calls, 3)  # 不为已淘汰的硬败展开第二个应手。

    def test_all_losing_fallback_restores_complete_scores_and_order(self):
        for player in (BLACK, WHITE):
            rows, calls = self.run_tree(player, all_lose=True)
            full, _ = self.run_tree(player, all_lose=True, complete=True)
            self.assertEqual(rows, full)
            self.assertEqual((rows[0]['r'], rows[0]['c']), (7, 9))
            self.assertEqual(rows[0]['score'], 80.)
            self.assertEqual(rows[1]['score'], -105.)
            self.assertTrue(all(row['score_complete'] for row in rows))
            self.assertEqual(calls, 4)  # 补分复用第一阶段叶子，不重复执行叶子函数。

    def test_explicit_complete_scores_retains_numeric_consumers(self):
        rows, calls = self.run_tree(complete=True)
        self.assertTrue(all(row['score_complete'] for row in rows))
        self.assertTrue(all(isinstance(row['score'], float) for row in rows))
        self.assertEqual(calls, 4)

    def test_empty_or_occupied_candidates_release_pass(self):
        engine = Engine()
        board = empty_board()
        for points in ([], {}, 'invalid'):
            self.assertEqual(engine.deep_search.rank_candidates(board, points, BLACK), [])
            self.assertIsNone(engine.deep_search._root_pass)
        board[7][7] = BLACK
        engine.on_move(board, 7, 7, BLACK)
        self.assertEqual(engine.deep_search.rank_candidates(board, {(7, 7): 100}, WHITE), [])

    def test_none_scores_sort_after_complete_scores_in_same_state(self):
        deep = Engine().deep_search
        self.assertGreater(deep._pick_key_state(-1, -1000.), deep._pick_key_state(-1, None))
        self.assertGreater(deep._pick_key_state(0, -1000.), deep._pick_key_state(-1, None))

    def test_list_coordinates_keep_existing_callers_compatible(self):
        engine = Engine()
        engine.deep_search.T0 = 1.
        board = empty_board()
        rows = engine.deep_search.rank_candidates(board, [[7, 7], [7, 9]], BLACK)
        self.assertEqual({(row['r'], row['c']) for row in rows}, {(7, 7), (7, 9)})
        self.assertEqual(board, empty_board())
        self.assertIsNone(engine.deep_search._root_pass)

    def test_pass_and_leaf_memos_do_not_leak_between_calls(self):
        engine = Engine()
        left, _ = self.run_tree(all_lose=True, engine=engine)
        right, _ = self.run_tree(all_lose=True, factor=2., engine=engine)
        self.assertEqual(right[0]['score'], 2 * left[0]['score'])

    def test_decision_network_requests_complete_scores(self):
        engine = Engine(use_decision_net=True)
        board = empty_board()
        # 本方前三手现在由开局模块接管；完整评分契约在第四手及之后验证。
        for r, c, player in ((0, 0, BLACK), (14, 14, WHITE), (1, 2, BLACK),
                             (13, 11, WHITE), (2, 4, BLACK), (12, 9, WHITE)):
            board[r][c] = player
            engine.on_move(board, r, c, player)
        row = engine.deep_search._candidate_result(7, 7, 1.)
        with patch.object(engine.search, 'tactical_candidates', return_value=('', {})), \
                patch.object(engine.search, '_fallback', return_value=[(7, 7, 80.), (7, 9, 70.)]), \
                patch.object(engine, '_memory_bonus', return_value={}), \
                patch.object(engine, '_decision_pick', return_value=(7, 7)), \
                patch.object(engine.deep_search, 'rank_candidates', return_value=[row]) as rank:
            engine.analyze_turn(board, BLACK)
        self.assertTrue(rank.call_args.kwargs['complete_scores'])

    def test_temperature_change_applies_to_next_decision(self):
        engine = Engine()
        deep = engine.deep_search
        deep.T0 = 20.
        roots = {(7, 7): 100., (7, 9): 100.}

        def candidates(board, player, n=5, plies=0):
            deep.T0 = 0.  # 模拟主线程在后台思考时切换下一手档位。
            return [(r, c, w) for (r, c), w in roots.items()] if player == BLACK else [(6, 6, 100.)]

        with patch.object(deep, '_candidate_points', side_effect=candidates), \
                patch.object(deep, '_check_win', return_value=False), \
                patch.object(deep, '_recursive', return_value=(0., 0, 0, 0, 1, 0, ())) as recursive:
            deep.rank_candidates(empty_board(), roots, BLACK)
            self.assertEqual(recursive.call_count, 2)
            self.assertTrue(all(call.args[4] == 14. for call in recursive.call_args_list))
            recursive.reset_mock()
            deep.rank_candidates(empty_board(), roots, BLACK)
            recursive.assert_not_called()

    def test_display_does_not_turn_missing_score_into_zero(self):
        try:
            from play import GomokuApp
        except ImportError as exc:
            self.skipTest(str(exc))
        self.assertEqual(GomokuApp._fmt_score(None), '未补算')
        self.assertEqual(GomokuApp._fmt_score(1e9), '直取')
        self.assertEqual(GomokuApp._fmt_score(-123.), '-123')


if __name__ == '__main__':
    unittest.main()
