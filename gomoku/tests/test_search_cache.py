# -*- coding: utf-8 -*-
"""跨回合搜索缓存回归：候选上限、冷/热一致、命中、失效、隔离与欠温续算。

被测实现位于 gomoku/core/deep_search.py 与 gomoku/core/engine.py：
`engine.on_move` 推进真实落子树，`engine.reset` 清空；缓存身份由类常量、search
模块权重与方法对象决定；旧叶子续推温度由欠温、权重、评分和深度共同计算。
"""
import copy
import os
import tempfile
import unittest
from unittest.mock import patch

from gomoku import memory
from gomoku.core.deep_search import DeepSearch
from gomoku.core.engine import Engine
from gomoku.core.pattern import PatternAnalyzer
from gomoku.core.search import Search
from gomoku.core.utils import BLACK, WHITE, empty_board, place

GAME = 'H8 G8 G7 H7 I6 F9 F6 I9 E5 D4 H6 G6 G9 J9 F5 I10'.split()
TEST_T0 = 30.0    # 回归用入口温度；只压运行时间，缓存/续温语义与默认 T0 相同。
HIT_KEYS = ('tactical_hits', 'candidate_hits', 'leaf_hits', 'feature_hits', 'subtree_hits')


def rc(token):
    return (int(token[1:]) - 1, ord(token[0]) - 65)


def cd(r, c):
    return chr(65 + c) + str(r + 1)


def replay(engine, board, tokens):
    for index, token in enumerate(tokens):
        r, c = rc(token)
        player = BLACK if index % 2 == 0 else WHITE
        place(board, r, c, player)
        engine.on_move(board, r, c, player)
    return board


def build(tokens=GAME):
    engine = Engine()
    board = empty_board()
    replay(engine, board, tokens)
    engine.deep_search.T0 = TEST_T0
    engine.deep_search.OPENING_COLD_T0 = TEST_T0
    engine.deep_search.MIDDLE_COLD_T0 = TEST_T0
    return engine, board


def moves_of(tokens):
    out = []
    for index, token in enumerate(tokens):
        r, c = rc(token)
        out.append(dict(r=r, c=c, player=BLACK if index % 2 == 0 else WHITE))
    return out


def signature(rows):
    return [(x['r'], x['c'], None if x['score'] is None else round(float(x['score']), 6),
             x['state'], bool(x['score_complete'])) for x in rows]


def hits(stats):
    return sum(stats.get(name, 0) for name in HIT_KEYS)


class SearchCacheTest(unittest.TestCase):

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        for context in (
                patch.dict(os.environ, {'GOMOKU_HOME': tmp.name, 'DSH_GOMOKU_GA': '0'}),
                patch.object(memory, '_global_memory', None)):
            context.start()
            self.addCleanup(context.stop)

    # ---------- 候选上限 ----------

    def test_root_and_internal_two_attack_two_defense_at_all_depths(self):
        engine, board = build(GAME[:4])
        deep = engine.deep_search
        for player in (BLACK, WHITE):
            self.assertFalse(deep._tactical_result(board, player)[0])
            root = deep._candidate_points(board, player, 5, plies=0)
            expected_root = engine.search._fallback_by_count(board, 2, 2, player=player, with_score=True)
            expected_internal = expected_root
            # 普通点现在按评分顺序搜索；配额仍先取2攻2防，不截旧5点表。
            clamp = lambda rows: sorted([(r, c, min(max(float(w), 1.0), 100.0))
                                  for r, c, w in rows], key=lambda row: -row[2])
            self.assertEqual(root, clamp(expected_root))
            self.assertEqual(len(root), 4)
            for plies in (1, 8, 16, 20):
                with self.subTest(player=player, plies=plies):
                    points = deep._candidate_points(board, player, 5, plies=plies)
                    self.assertEqual(points, clamp(expected_internal))
                    self.assertEqual(len(points), 4)

    # ---------- 冷算 vs 缓存 ----------

    def test_cold_and_warm_match_after_real_move(self):
        engine, board = build()
        deep = engine.deep_search
        self.assertEqual(deep._tree_root, dict(deep._new_tree_node(deep._position_key(board)), mover=engine.moves[-1][2]))
        self.assertEqual(hits(deep.cache_stats), 0)  # 开始前没有旧搜索结果；本轮内部允许命中。
        cold = engine.analyze_turn(board, BLACK)['part4_result']['ranked']
        move = (cold[0]['r'], cold[0]['c'])
        inherited = deep._tree_root['children'][(*move, BLACK)]
        self.assertTrue(any(inherited[name] for name in ('tactical', 'candidates', 'values', 'children')))
        place(board, move[0], move[1], BLACK)
        engine.on_move(board, move[0], move[1], BLACK)
        self.assertIs(deep._tree_root, inherited, '真实落子必须换到已搜索的同一子树')
        warm = engine.analyze_turn(board, WHITE)['part4_result']['ranked']
        self.assertGreater(hits(engine.deep_search.cache_stats), 0)

        fresh, fresh_board = build()
        place(fresh_board, move[0], move[1], BLACK)
        fresh.on_move(fresh_board, move[0], move[1], BLACK)
        reference = fresh.analyze_turn(fresh_board, WHITE)['part4_result']['ranked']
        self.assertEqual(signature(warm), signature(reference))

    def test_repeated_analysis_is_stable_and_does_not_rewarm(self):
        engine, board = build()
        first = engine.analyze_turn(board, BLACK)['part4_result']['ranked']
        second = engine.analyze_turn(board, BLACK)['part4_result']['ranked']
        self.assertEqual(signature(first), signature(second))
        self.assertEqual(engine.deep_search.cache_stats['frontier_resumes'], 0)

    # ---------- 失效 ----------

    def test_reset_clears_tree(self):
        engine, board = build()
        engine.analyze_turn(board, BLACK)
        deep = engine.deep_search
        old_root = deep._tree_root
        self.assertTrue(old_root['children'])
        engine.reset(moves_of(GAME))
        self.assertIsNot(deep._tree_root, old_root)
        self.assertEqual(deep._tree_root, dict(deep._new_tree_node(deep._position_key(board)), mover=engine.moves[-1][2]))
        self.assertIsNone(deep._tree_signature)
        self.assertEqual(deep._cache_positions, {})
        self.assertEqual(deep._cache_paths, {})
        self.assertEqual(hits(deep.cache_stats), 0)
        fresh, fresh_board = build()
        after = engine.analyze_turn(board, BLACK)['part4_result']['ranked']
        reference = fresh.analyze_turn(fresh_board, BLACK)['part4_result']['ranked']
        self.assertEqual(signature(after), signature(reference))
        engine.reset()
        self.assertIsNone(deep._tree_root)
        self.assertEqual(engine.moves, [])

    def test_parameter_change_invalidates_tree(self):
        engine, board = build()
        engine.analyze_turn(board, BLACK)
        deep = engine.deep_search
        old_root, old_signature = deep._tree_root, deep._tree_signature
        deep.T0 = TEST_T0 + 10.0
        begin = deep._begin_search_cache
        observed = []

        def begin_and_check(current_board, trace):
            begin(current_board, trace)
            self.assertIsNot(deep._tree_root, old_root)
            self.assertNotEqual(deep._tree_signature, old_signature)
            self.assertEqual(deep._tree_root,
                             deep._new_tree_node(deep._position_key(current_board)))
            self.assertEqual(hits(deep.cache_stats), 0)  # 只在展开前检查旧树未继承。
            observed.append(deep._tree_root)

        with patch.object(deep, '_begin_search_cache', side_effect=begin_and_check):
            passed = engine.analyze_turn(board, BLACK)['part4_result']['ranked']
        self.assertTrue(observed)
        fresh, fresh_board = build()
        fresh.deep_search.T0 = TEST_T0 + 10.0
        reference = fresh.analyze_turn(fresh_board, BLACK)['part4_result']['ranked']
        self.assertEqual(signature(passed), signature(reference))

    # ---------- 叶子评分隔离 ----------

    def test_terminal_and_stale_values_are_isolated(self):
        engine, board = build()
        deep = engine.deep_search
        deep._begin_search_cache(board, False)
        try:
            node = deep._tree_root
            stale = deep._leaf_score(board, [], 'stale', BLACK)
            win = deep._leaf_score(board, [], 'win', BLACK)
            stale_again = deep._leaf_score(board, [], 'stale', BLACK)
            self.assertEqual(win, deep.WIN_SCORE)
            self.assertEqual(stale, stale_again)
            self.assertNotEqual(stale, win)
            statuses = {key[0] for key in node['values']}
            self.assertIn('stale', statuses)
            self.assertIn('win', statuses)
            self.assertEqual(node['terminal'], BLACK)
        finally:
            deep._reuse_active = False

    # ---------- 欠温续算 ----------

    def test_frontier_resume_uses_credit_plus_negative_temperature(self):
        engine, board = build()
        deep = engine.deep_search
        deep.T0 = 80.0
        deep._begin_search_cache(board, False)
        try:
            deep._tree_epoch = 1
            node = deep._new_tree_node(deep._position_key(board))
            node['frontier'] = dict(player=BLACK, temperature=-15.0,
                                    path_weight=1.0, epoch=0)
            self.assertAlmostEqual(deep._resume_temperature(node, BLACK, 999.0), 40.0)
            self.assertEqual(deep.cache_stats['frontier_resumes'], 1)
            # 同方同局重复调用：返回值复用，不重复拨温/计数。
            self.assertAlmostEqual(deep._resume_temperature(node, BLACK, 999.0), 40.0)
            self.assertEqual(deep.cache_stats['frontier_resumes'], 1)
            # 无旧末端、或旧末端不是同一方：保持本轮入口温度。
            empty_node = deep._new_tree_node(deep._position_key(board))
            self.assertEqual(deep._resume_temperature(empty_node, BLACK, 123.0), 123.0)
            wrong = deep._new_tree_node(deep._position_key(board))
            wrong['frontier'] = dict(player=WHITE, temperature=-15.0,
                                     path_weight=1.0, epoch=0)
            self.assertEqual(deep._resume_temperature(wrong, BLACK, 123.0), 123.0)
        finally:
            deep._reuse_active = False

    def test_path_state_is_not_shared(self):
        engine, board = build()
        deep = engine.deep_search
        deep._begin_search_cache(board, False)
        try:
            key = deep._position_key(board)
            left = deep._new_tree_node(key)
            right = deep._new_tree_node(key)
            left['frontier'] = dict(player=BLACK, temperature=-15.0,
                                    path_weight=1.0, epoch=0)
            deep._tree_epoch = 1
            self.assertAlmostEqual(deep._resume_temperature(left, BLACK, 999.0), 23.0)
            self.assertEqual(deep._resume_temperature(right, BLACK, 123.0), 123.0)
            self.assertIsNot(left['resume_values'], right['resume_values'])
            # 聚合分数键包含入口温度/路径权重，不同路径不共用结果。
            a = deep._result_key(BLACK, BLACK, 160.0, 1.0)
            b = deep._result_key(BLACK, BLACK, 80.0, 1.0)
            c = deep._result_key(BLACK, BLACK, 160.0, 0.5)
            self.assertNotEqual(a, b)
            self.assertNotEqual(a, c)
        finally:
            deep._reuse_active = False

    def test_leaf_continuation_policy_uses_weight_score_depth_and_soft_states(self):
        deep = DeepSearch(Search(PatternAnalyzer()))
        # 关闭基础信用后，避免续推上限截平高分与普通叶子的差异。
        deep._cache_credit = 0.0

        def frontier(**kwargs):
            data = dict(temperature=-15.0, path_weight=0.8, depth=4,
                        score=0.0, state=0, terminal=False)
            data.update(kwargs)
            return data

        normal, blocked = deep._continuation_temperature(frontier())
        self.assertFalse(blocked)
        high_score, _ = deep._continuation_temperature(frontier(score=900.0))
        self.assertGreater(high_score, normal)
        low_weight, low_weight_blocked = deep._continuation_temperature(
            frontier(path_weight=deep.CONTINUE_W_MIN / 2))
        self.assertTrue(low_weight_blocked)
        self.assertEqual(low_weight, 0.0)
        deep_leaf, deep_blocked = deep._continuation_temperature(
            frontier(depth=deep.CONTINUE_DEPTH_STOP))
        self.assertTrue(deep_blocked)
        self.assertEqual(deep_leaf, 0.0)

        soft_win, soft_win_blocked = deep._continuation_temperature(frontier(state=2))
        soft_lose, soft_lose_blocked = deep._continuation_temperature(frontier(state=-2))
        self.assertFalse(soft_win_blocked)
        self.assertFalse(soft_lose_blocked)
        self.assertEqual(soft_win, normal)
        self.assertEqual(soft_lose, normal)
        for hard_state in (1, -1):
            value, hard_blocked = deep._continuation_temperature(frontier(state=hard_state))
            self.assertTrue(hard_blocked)
            self.assertEqual(value, 0.0)

    def test_frontier_records_leaf_weight_depth_and_score(self):
        deep = DeepSearch(Search(PatternAnalyzer()))
        node = deep._new_tree_node(bytes(225))
        deep._remember_frontier(node, BLACK, -12.0, 0.35,
                                before_move=False, depth=7, score=625.0, state=-2)
        self.assertEqual(node['frontier']['path_weight'], 0.35)
        self.assertEqual(node['frontier']['depth'], 7)
        self.assertEqual(node['frontier']['score'], 625.0)
        self.assertEqual(node['frontier']['state'], -2)

    # ---------- 判态早停不能冒充完整评分 ----------

    def test_proven_hard_loss_reuses_fixed_complete_score(self):
        engine = Engine()
        deep = engine.deep_search
        deep.T0 = 4.
        board = empty_board()
        roots = {(7, 7): 100., (7, 9): 100.}
        calls = []

        def candidates(b, turn, n=5, plies=0):
            return [(r, c, w) for (r, c), w in roots.items()] if turn == BLACK else [
                (6, 6, 100.), (6, 8, 100.)]

        def won(b, r, c, turn):
            return turn != BLACK and (r, c) == (6, 6) and b[7][9] == BLACK

        def leaf(b, branch, status, turn, contexts):
            calls.append(status)
            return 999. if status == 'lose' else -10.

        def run():
            calls.clear()
            with patch.object(deep, '_candidate_points', side_effect=candidates), \
                    patch.object(deep, '_check_win', side_effect=won), \
                    patch.object(deep, '_leaf_score_impl', side_effect=leaf):
                return deep.rank_candidates(board, roots, BLACK)

        first = run()
        second = run()
        self.assertEqual(signature(first), signature(second))
        self.assertEqual(second[1]['state'], deep.STATE_LOSE)
        self.assertEqual(second[1]['score'], -1000.)
        self.assertTrue(second[1]['score_complete'])
        self.assertTrue(second[0]['score_complete'])

    # ---------- 黑白双方与模拟回滚 ----------

    def test_both_colors_and_state_rollback(self):
        engine, board = build()
        before = (copy.deepcopy(board), copy.deepcopy(engine.search.sb),
                  copy.deepcopy(engine.search.sw), list(engine.moves))
        black_rows = engine.analyze_turn(board, BLACK)['part4_result']['ranked']
        white_rows = engine.analyze_turn(board, WHITE)['part4_result']['ranked']
        self.assertEqual(before[0], board)
        self.assertEqual(before[1], engine.search.sb)
        self.assertEqual(before[2], engine.search.sw)
        self.assertEqual(before[3], engine.moves)

        fresh_b, board_b = build()
        fresh_w, board_w = build()
        self.assertEqual(signature(black_rows),
                         signature(fresh_b.analyze_turn(board_b, BLACK)['part4_result']['ranked']))
        self.assertEqual(signature(white_rows),
                         signature(fresh_w.analyze_turn(board_w, WHITE)['part4_result']['ranked']))



# 本轮发布默认改为8/3/120；以下旧机制回归显式固定旧口径。
from gomoku.tests._legacy_defaults import enter_legacy as _enter_legacy, leave_legacy as _leave_legacy

def setUpModule():
    _enter_legacy()

def tearDownModule():
    _leave_legacy()
if __name__ == '__main__':
    unittest.main(verbosity=2)
