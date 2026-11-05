# -*- coding: utf-8 -*-
"""零耗温续推回归：禁止普通补步，覆盖黑白、两层入口、当前盘面评分与缓存相位。"""
from contextlib import ExitStack
from copy import deepcopy
import unittest
from unittest.mock import patch

from gomoku.core.deep_search import DeepSearch
from gomoku.core.pattern import PatternAnalyzer
from gomoku.core.search import Search
from gomoku.core.utils import BLACK, WHITE, EMPTY, empty_board

A, O, B = (7, 7), (8, 7), (9, 7)


def snapshot(board, search, branch=(), contexts=()):
    return deepcopy((board, search.sb, search.sw, search._direction_levels,
                     search.gear, search.zero_history, search.zero_active,
                     branch, contexts))


def position(moves, swap=False):
    board = empty_board()
    search = Search(PatternAnalyzer())
    for index, label in enumerate(moves.split()):
        r, c = int(label[1:])-1, ord(label[0])-ord('A')
        side = BLACK if index % 2 == 0 else WHITE
        if swap:
            side = 3-side
        board[r][c] = side
        search.on_move(board, r, c, side)
    return board, search


class ZeroCostContinuationTest(unittest.TestCase):
    def assert_updated(self, search, board):
        reference = Search(PatternAnalyzer())
        reference.rebuild_all(board)
        self.assertEqual(search.sb, reference.sb)
        self.assertEqual(search.sw, reference.sw)
        for r, row in enumerate(board):
            for c, value in enumerate(row):
                if value == EMPTY:
                    for side in (BLACK, WHITE):
                        self.assertEqual(search._direction_levels[side][r*15+c],
                                         reference._direction_levels[side][r*15+c])

    def controlled(self, side, stage, temperature, mine, reply, *, weight=100.0,
                   cached=True, complete=True, fatal=False):
        board = empty_board()
        search = Search(PatternAnalyzer())
        deep = DeepSearch(search)
        deep.T0 = temperature
        branch, contexts = [], []
        if stage == 'recursive':
            for r, c, p in ((1, 1, side), (1, 2, 3-side)):
                board[r][c] = p
                search.on_move(board, r, c, p)
                branch.append((r, c, p, 60.0))
                contexts.append({'fatal': 0, 'temp': temperature})
        base = sum(v != EMPTY for row in board for v in row)
        before = snapshot(board, search, branch, contexts)
        leaves, moves = [], []
        score, on_move = deep._leaf_score, search.on_move

        def delta(b):
            return sum(v != EMPTY for row in b for v in row)-base

        def tactical(b, player):
            self.assert_updated(search, b)
            n = delta(b)
            self.assertLessEqual(n, 2, '温度耗尽后落了额外普通步')
            if n == 0 and player == side and mine:
                return mine, {A: weight}
            if n == 1 and player == 3-side and reply:
                return reply, {O: weight}
            return None, {}

        def fallback(b, *args, **kwargs):
            n = delta(b)
            self.assertLess(n, 2, '对方刚落完后不得生成普通补手')
            return [(*(A if n == 0 else O), weight)]

        def observe(b, path, status, player, ctx_branch=None):
            self.assert_updated(search, b)
            value = score(b, path, status, player, ctx_branch)
            node = deep._position_node(b)
            leaves.append(dict(delta=delta(b), path=deepcopy(path), status=status,
                               player=player, value=value, contexts=deepcopy(ctx_branch),
                               frontier=deepcopy(node['frontier']) if node else None))
            return value

        def record(b, r, c, p, **kwargs):
            moves.append((r, c, p))
            return on_move(b, r, c, p, **kwargs)

        with ExitStack() as stack:
            stack.enter_context(patch.object(search, 'tactical_candidates', side_effect=tactical))
            stack.enter_context(patch.object(search, '_fallback', side_effect=fallback))
            stack.enter_context(patch.object(search, '_fallback_by_count', side_effect=fallback))
            stack.enter_context(patch.object(search, 'on_move', side_effect=record))
            stack.enter_context(patch.object(deep, '_leaf_score', side_effect=observe))
            stack.enter_context(patch.object(deep, '_has_kill_points',
                                side_effect=AssertionError('原始0～4扫描不得控制续推')))
            stack.enter_context(patch.object(deep, '_has_forcing_threat', return_value=fatal))
            stack.enter_context(patch.object(deep, '_sub_at', return_value=17))
            if not cached:
                stack.enter_context(patch.object(deep, '_begin_search_cache', return_value=None))
            if stage == 'root':
                rows = deep.rank_candidates(board, {A: weight}, side,
                                            complete_scores=complete)
                self.assertEqual(rows[0]['state'], 0)
                value = rows[0]['score']
            else:
                value = deep._recursive_impl(board, side, 3-side, side, temperature,
                                              branch, contexts, {})[0]
        self.assertEqual(snapshot(board, search, branch, contexts), before)
        self.assertIsNone(deep._root_pass)
        self.assertEqual(len(leaves), 1)
        self.assertEqual(value, leaves[0]['value'])
        self.assertEqual(leaves[0]['player'], side)
        self.assertEqual(leaves[0]['status'], 'stale')
        self.assertEqual(len(moves), leaves[0]['delta'])
        return deep, leaves[0]

    def test_exhausted_ordinary_never_moves_even_weight_100(self):
        for side in (BLACK, WHITE):
            for stage in ('root', 'recursive'):
                for t in (0.0, -45.0):
                    for weight in (60.0, 100.0):
                        with self.subTest(side=side, stage=stage, t=t, weight=weight):
                            deep, leaf = self.controlled(side, stage, t, None, None,
                                                         weight=weight, fatal=True)
                            self.assertEqual(leaf['delta'], 0)
                            self.assertGreater(deep._temperature_cost(weight, False), 0)

    def test_exhausted_zero_cost_chain_stops_before_ordinary_return(self):
        for side in (BLACK, WHITE):
            for stage in ('root', 'recursive'):
                for reason in ('my-five', 'opp-five', 'my-vcf', 'opp-vcf', 'my-d33', 'opp-d33'):
                    for t in (0.0, -45.0):
                        with self.subTest(side=side, stage=stage, reason=reason, t=t):
                            deep, leaf = self.controlled(side, stage, t, reason,
                                                         'opp-vcf', weight=80.0)
                            self.assertEqual(leaf['delta'], 2)
                            self.assertEqual(leaf['contexts'][-1]['temp'], t)
                            if stage == 'root':
                                self.assertTrue(leaf['frontier']['before_move'])
                                self.assertEqual(leaf['frontier']['temperature'], t)

    def test_positive_budget_keeps_original_cost_and_can_cross_zero(self):
        for side in (BLACK, WHITE):
            for stage in ('root', 'recursive'):
                for mine, reply in ((None, None), (None, 'opp-vcf'), ('my-vcf', None)):
                    with self.subTest(side=side, stage=stage, mine=mine, reply=reply):
                        deep, leaf = self.controlled(side, stage, 1.0, mine, reply,
                                                     weight=100.0)
                        expected = 1 if mine is None and reply is None else 2
                        self.assertEqual(leaf['delta'], expected)
                        self.assertEqual(deep._dt(100.0), 3.0)
                        if expected == 2:
                            self.assertEqual(leaf['contexts'][-1]['temp'], 1.0-deep._dt(0.0))

    def test_cache_and_score_modes_use_current_board_when_opponent_just_moved(self):
        for side in (BLACK, WHITE):
            for cached in (False, True):
                for complete in (False, True):
                    with self.subTest(side=side, cached=cached, complete=complete):
                        _, leaf = self.controlled(side, 'root', 1.0, None, 'opp-vcf',
                                                  cached=cached, complete=complete)
                        self.assertEqual(leaf['delta'], 2)
                        self.assertEqual(leaf['path'][-1][2], 3-side)

    def test_real_j4_stops_with_i7_empty_for_both_colors_and_layers(self):
        for swap in (False, True):
            side = WHITE if swap else BLACK
            for stage in ('root', 'recursive'):
                board, search = position('H8 G8 G7 H7 I6 F9 F6 E5 H6 G6', swap)
                deep = DeepSearch(search)
                deep.T0 = 1.0
                before = snapshot(board, search)
                original, leaf_score = deep._candidate_points, deep._leaf_score
                leaves = []
                i5, j4 = (4, 8), (3, 9)

                def candidates(b, p, n=5, plies=0):
                    actual = original(b, p, n, plies)
                    if b[i5[0]][i5[1]] == EMPTY:
                        return [(*i5, 60.0)]
                    if b[j4[0]][j4[1]] == EMPTY:
                        self.assertEqual(deep._tactical_result(b, 3-side)[0], 'opp-vcf')
                        choices = [point for point in actual if point[:2] == j4]
                        self.assertEqual(len(choices), 1)
                        return choices
                    self.fail('J4 后不应再生成黑方普通候选')

                def observe(b, branch, status, p, ctx_branch=None):
                    self.assert_updated(search, b)
                    self.assertEqual(b[j4[0]][j4[1]], 3-side)
                    self.assertEqual(b[6][8], EMPTY, 'I7 普通补手必须为空')
                    self.assertIsNone(search.tactical_candidates(b, side)[0])
                    self.assertEqual(p, side)
                    self.assertEqual(status, 'stale')
                    leaves.append((len(branch), leaf_score(b, branch, status, p, ctx_branch)))
                    return leaves[-1][1]

                with patch.object(deep, '_candidate_points', side_effect=candidates), \
                     patch.object(deep, '_leaf_score', side_effect=observe):
                    if stage == 'root':
                        result = deep.rank_candidates(board, {i5: 60.0}, side,
                                                      complete_scores=True)[0]['score']
                    else:
                        result = deep._recursive_impl(board, side, 3-side, side, 1.0,
                                                      [], [], {})[0]
                self.assertEqual(leaves, [(2, result)])
                self.assertEqual(snapshot(board, search), before)

    def test_replay_position_after_l9_stops_before_black_j8(self):
        """历史 95 层链的第 40 手后，黑方无 reason 时不得补走 J8。"""
        board, search = position(
            'H8 G8 G7 H7 I6 F9 F6 E5 H6 G6 I5 J4 I7 I8 '
            'F5 F4 K5 J6 H5 L5 J5 G5 I4 I3 H3 G2 G3 G4 H4 '
            'H2 K6 L7 K8 K4 F2 E1 K9 K7 J7 L9'
        )
        deep = DeepSearch(search)
        before = snapshot(board, search)
        original_leaf = deep._leaf_score
        leaf_calls = []

        def observe(b, branch, status, player, ctx_branch=None):
            leaf_calls.append((status, player, tuple(branch)))
            return original_leaf(b, branch, status, player, ctx_branch)

        with patch.object(deep, '_candidate_points',
                          side_effect=AssertionError('黑方无 reason 时不得生成 J8 普通候选')), \
             patch.object(deep, '_leaf_score', side_effect=observe):
            reason, points = deep._tactical_result(board, BLACK)
            self.assertIsNone(reason)
            self.assertEqual(points, {})
            value = deep._recursive(
                board, BLACK, WHITE, BLACK, -1.0, [], [], {}
            )

        self.assertEqual(len(leaf_calls), 1)
        self.assertEqual(leaf_calls[0][0:2], ('stale', BLACK))
        self.assertEqual(value[4], 1)
        self.assertEqual(board[7][9], EMPTY, 'J8 不得被错误补出')
        self.assertEqual(snapshot(board, search), before)

    def test_actual_five_precedes_next_turn_and_negative_budget(self):
        for side in (BLACK, WHITE):
            for stage in ('root', 'recursive'):
                board = empty_board()
                for c in range(3, 7):
                    board[7][c] = side
                search = Search(PatternAnalyzer())
                search.rebuild_all(board)
                deep = DeepSearch(search)
                deep.T0 = -45.0
                before = snapshot(board, search)
                tactical = search.tactical_candidates

                def check_before(b, p):
                    self.assertEqual(b[A[0]][A[1]], EMPTY, '已成五不得再查下一手')
                    return tactical(b, p)

                with patch.object(search, 'tactical_candidates', side_effect=check_before):
                    if stage == 'root':
                        row = deep.rank_candidates(board, {A: 100.0}, side,
                                                   complete_scores=True)[0]
                        self.assertEqual((row['score'], row['state']), (1000.0, 1))
                    else:
                        with patch.object(deep, '_candidate_points', return_value=[(*A, 100.0)]):
                            row = deep._recursive_impl(board, side, 3-side, side, -45.0,
                                                       [], [], {})
                        self.assertEqual(row[:2], (1000.0, 1))
                self.assertEqual(snapshot(board, search), before)

    def test_frontier_phase_and_epoch_do_not_mix_or_replenish_twice(self):
        deep = DeepSearch(Search(PatternAnalyzer()))
        deep._cache_credit = 160.0
        for side in (BLACK, WHITE):
            for phase in (False, True):
                node = deep._new_tree_node(bytes(225))
                deep._tree_epoch = 0
                deep._remember_frontier(node, side, -61.22, 1.0, before_move=phase)
                self.assertEqual(deep._resume_temperature(node, side, -7.0,
                                                         before_move=phase, depth=0), -7.0)
                deep._tree_epoch = 1
                self.assertEqual(deep._resume_temperature(node, side, -7.0,
                                                         before_move=not phase, depth=0), -7.0)
                self.assertEqual(deep._resume_temperature(node, 3-side, -7.0,
                                                         before_move=phase, depth=0), -7.0)
                self.assertAlmostEqual(deep._resume_temperature(node, side, -7.0,
                                                               before_move=phase, depth=0), 2.78)
                self.assertAlmostEqual(deep._resume_temperature(node, side, 999.0,
                                                               before_move=phase, depth=0), 2.78)
        self.assertEqual(deep.cache_stats['frontier_resumes'], 4)

    def test_real_reroot_resumes_before_move_frontier_with_debt(self):
        for side in (BLACK, WHITE):
            board = empty_board()
            search = Search(PatternAnalyzer())
            deep = DeepSearch(search)
            deep.T0 = 1.0  # 建立仍欠温超过64的真实末端，验证拨温不会抹掉欠温。

            def tactical(b, p):
                if b[A[0]][A[1]] == side and b[O[0]][O[1]] == EMPTY and p == 3-side:
                    return 'opp-vcf', {O: 80.0}
                return None, {}

            with patch.object(search, 'tactical_candidates', side_effect=tactical), \
                 patch.object(search, '_fallback', return_value=[(*A, 1.0)]):
                deep.rank_candidates(board, {A: 1.0}, side, complete_scores=True)
                debt = 1.0-deep._dt(0.0)
                for r, c, p in ((*A, side), (*O, 3-side)):
                    board[r][c] = p
                    search.on_move(board, r, c, p)
                    deep.advance_search_cache(board, r, c, p)
                self.assertTrue(deep._tree_root['frontier']['before_move'])
                self.assertEqual(deep._tree_root['frontier']['temperature'], debt)
                # 换根后按当前深度0拨入64，旧深度不构成硬停止条件。
                before = snapshot(board, search)
                with patch.object(search, 'on_move', side_effect=AssertionError('欠温未清不得补普通步')):
                    deep.rank_candidates(board, {B: 100.0}, side, complete_scores=True)
                    self.assertEqual(deep.cache_stats['frontier_resumes'], 1)
                    self.assertEqual(deep._tree_root['frontier']['temperature'], debt+64.)
                    self.assertEqual(deep._tree_root['resume_values'][(side, True)], debt+64.)
                    self.assertFalse(deep._tree_root['resume_blocked'])
                    deep.rank_candidates(board, {B: 100.0}, side, complete_scores=True)
                    self.assertEqual(deep.cache_stats['frontier_resumes'], 0)
                    self.assertEqual(deep._tree_root['frontier']['temperature'], debt+64.)
                    self.assertEqual(deep._tree_root['resume_values'][(side, True)], debt+64.)
                    self.assertFalse(deep._tree_root['resume_blocked'])
                self.assertEqual(snapshot(board, search), before)




# 本轮发布默认改为8/3/120；以下旧机制回归显式固定旧口径。
from gomoku.tests._legacy_defaults import enter_legacy as _enter_legacy, leave_legacy as _leave_legacy

def setUpModule():
    _enter_legacy()

def tearDownModule():
    _leave_legacy()
if __name__ == '__main__':
    unittest.main()
