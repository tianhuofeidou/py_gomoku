# -*- coding: utf-8 -*-
"""最优应手硬回传与子/孙节点局部软证据的回归。"""
import copy
import itertools
import unittest
from unittest.mock import patch

from gomoku.core.deep_search import DeepSearch as D
from gomoku.core.engine import Engine
from gomoku.core.utils import BLACK, WHITE, empty_board


class StateBackpropTest(unittest.TestCase):
    def run_tree(self, scenario, player):
        """在真实模拟/回滚流程中控制分支及终局，不修改候选和评分算法。"""
        engine = Engine()
        deep = engine.deep_search
        deep.T0 = 20.0
        board = empty_board()
        before = copy.deepcopy((board, engine.search.sb, engine.search.sw,
                                engine.search.gear, engine.search.zero_history))
        root = (7, 7)
        first, second = (6, 6), (6, 8)
        good, bad, unclear = (5, 5), (5, 7), (5, 9)
        replies = [first] if scenario in ('risk', 'hard_win_risk') else [first, second]

        def candidates(b, turn, n=5, plies=0):
            count = sum(v != 0 for row in b for v in row)
            if count == 0:
                points = [root]
            elif count == 1:
                points = replies
            elif count == 2:
                if scenario == 'win_unknown':
                    points = [unclear]
                elif scenario == 'win_lose':
                    points = [good]
                elif scenario == 'risk':
                    points = [bad, unclear]
                elif b[first[0]][first[1]]:
                    points = [good, bad] if scenario in ('both', 'hard_win_risk') else [good]
                else:
                    points = [unclear]
            elif count == 3:
                points = [(4, 4)] if b[bad[0]][bad[1]] else [(4, 6)]
            else:
                points = []
            return [(r, c, 100.0) for r, c in points]

        def won(b, r, c, turn):
            if turn == player:
                return (r, c) == good
            return ((r, c) == (4, 4) or
                    (scenario in ('win_unknown', 'win_lose') and (r, c) == first))

        with patch.object(deep, '_candidate_points', side_effect=candidates), \
                patch.object(deep, '_check_win', side_effect=won), \
                patch.object(deep, '_leaf_score', side_effect=lambda b, br, status, p, ctx: {
                    'win': 1000.0, 'lose': -1000.0, 'stale': 17.0}[status]):
            ranked = deep.rank_candidates(board, {root: 100.0}, player)
        self.assertEqual(before, (board, engine.search.sb, engine.search.sw,
                                  engine.search.gear, engine.search.zero_history))
        self.assertEqual(engine.moves, [])
        result = ranked[0]
        self.assertEqual(result['forced'], deep._state_forced(result['state']))
        self.assertEqual(result['state_name'], deep._state_name(result['state']))
        self.assertEqual(result['child_state'], deep._state_child_state(result['state']))
        return result

    def test_opponent_can_select_win_among_unknown_or_losing_replies(self):
        for player, scenario in itertools.product((BLACK, WHITE), ('win_unknown', 'win_lose')):
            with self.subTest(player=player, scenario=scenario):
                self.assertEqual(self.run_tree(scenario, player)['state'], D.STATE_LOSE)

    def test_local_opportunity_risk_and_risk_priority(self):
        expected = {'opportunity': D.STATE_EXISTS_WIN, 'risk': D.STATE_EXISTS_LOSE,
                    'both': D.STATE_EXISTS_LOSE, 'hard_win_risk': D.STATE_WIN}
        for player, scenario in itertools.product((BLACK, WHITE), expected):
            with self.subTest(player=player, scenario=scenario):
                self.assertEqual(self.run_tree(scenario, player)['state'], expected[scenario])

    def test_terminal_win_ignores_earlier_nonforcing_move_and_restores_context(self):
        for player in (BLACK, WHITE):
            engine = Engine()
            board = empty_board()
            for r, c in [(7, 3), (7, 4), (7, 5), (7, 6), (1, 1)]:
                board[r][c] = player
                engine.on_move(board, r, c, player)
            branch = [(1, 1, player, 60.0)]
            contexts = [{'camp': 1, 'fatal': 0}]
            before = copy.deepcopy((board, engine.search.sb, engine.search.sw, branch, contexts))
            cache = {}
            with patch.object(engine.deep_search, '_candidate_points', return_value=[(7, 7, 100.0)]) as candidates:
                result = engine.deep_search._recursive(
                    board, player, 3 - player, player, 160, branch, contexts, cache)
                cached = engine.deep_search._recursive(
                    board, player, 3 - player, player, 160, branch, contexts, cache)
            self.assertEqual(result[0], 1000.0)
            self.assertEqual(result[1], D.STATE_WIN)
            self.assertEqual(result[-1], (D.STATE_WIN,))
            self.assertEqual(cached, result)
            candidates.assert_called_once()
            self.assertEqual(before, (board, engine.search.sb, engine.search.sw, branch, contexts))

    def test_hard_loss_cannot_win_by_score_and_all_loss_still_has_legal_fallback(self):
        for player, all_lose in itertools.product((BLACK, WHITE), (False, True)):
            engine = Engine()
            deep = engine.deep_search
            deep.T0 = 1.0
            board = empty_board()
            roots = {(7, 7): 100.0, (7, 9): 100.0}
            def candidates(b, turn, n=5, plies=0):
                return [(r, c, w) for (r, c), w in roots.items()] if turn == player else [(6, 6, 100.0)]
            def won(b, r, c, turn):
                return turn == 3-player and (all_lose or b[7][9] == player)
            def score(b, branch, status, turn, contexts):
                if not all_lose:
                    return 999.0 if status == 'lose' else -100.0
                return -5.0 if b[7][9] == player else -20.0
            with patch.object(deep, '_candidate_points', side_effect=candidates), \
                    patch.object(deep, '_check_win', side_effect=won), \
                    patch.object(deep, '_leaf_score', side_effect=score):
                ranked = deep.rank_candidates(board, roots, player)
                move = deep.deep_search(board, roots, player)
            self.assertEqual(move, (7, 9) if all_lose else (7, 7))
            self.assertEqual(ranked[0]['state'], D.STATE_LOSE if all_lose else D.STATE_NONE)
            self.assertEqual(board, empty_board())

    def test_direct_five_and_unsearched_block_have_consistent_state_fields(self):
        for player in (BLACK, WHITE):
            engine = Engine()
            board = empty_board()
            for r, c in [(7, 3), (7, 4), (7, 5), (7, 6)]:
                board[r][c] = player
                engine.on_move(board, r, c, player)
            for turn, expected in ((player, D.STATE_WIN), (3-player, D.STATE_NONE)):
                result = engine.analyze_turn(board, turn)['part4_result']['ranked'][0]
                self.assertEqual(result['state'], expected)
                self.assertEqual(result['forced'], D._state_forced(expected))
                self.assertEqual(result['child_state'], 'none')
                self.assertEqual(result['score'], 1e9)

    def test_depth_five_hard_results_match_independent_minimax(self):
        def combine_tree(leaves):
            if len(leaves) == 1:
                return leaves[0], []
            mid = len(leaves) // 2
            children = [combine_tree(leaves[:mid]), combine_tree(leaves[mid:])]
            states = [c[0] for c in children]
            grandchildren = [s for c in children for s in c[1]]
            return D._merge_child_states(states, grandchildren), states

        def bounds(leaves, depth=0):
            if len(leaves) == 1:
                if leaves[0] == 0:
                    return -1, 1
                value = leaves[0] * (-1 if depth % 2 else 1)
                return value, value
            mid = len(leaves) // 2
            a, b = bounds(leaves[:mid], depth+1), bounds(leaves[mid:], depth+1)
            select = min if depth % 2 == 0 else max
            return select(a[0], b[0]), select(a[1], b[1])

        cases = [[0]*32, [1]*32, [-1]*32, [1]+[0]*31,
                 [-1, -1]+[0]*30, [-1, -1, 1, 0]*8,
                 [1, -1, 1, 0]*8, [1, -1, 0, 0]*8]
        # 同时穷举较小树的全部叶子配置，避免只锁定几个手选例子。
        cases.extend(itertools.product((-1, 0, 1), repeat=8))
        for leaves in cases:
            state, _ = combine_tree(leaves)
            interval = bounds(leaves)
            expected = 1 if interval == (1, 1) else -1 if interval == (-1, -1) else 0
            self.assertEqual(D._state_forced(state), expected, leaves)


if __name__ == '__main__':
    unittest.main()
