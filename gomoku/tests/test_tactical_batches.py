"""战术分批：完整候选保留、部分分标记、真实换根补推和回滚。"""
import copy
import unittest
from contextlib import ExitStack
from unittest.mock import patch

from gomoku.core.deep_search import DeepSearch
from gomoku.core.engine import Engine
from gomoku.core.utils import BLACK, WHITE, empty_board


class TacticalBatchTest(unittest.TestCase):
    A = (7, 7)
    B = (6, 7)
    POINTS = [(2, c) for c in (1, 3, 5, 7, 9, 11, 13)]

    def snapshot(self, e, board):
        s = e.search
        return copy.deepcopy((board, e.moves, s.sb, s.sw, s._direction_levels,
                              s.zero_history, s.zero_active, s.gear))

    def fixture(self, player=BLACK, root_many=False):
        e = Engine()
        d = e.deep_search
        d.T0 = 0.
        d.TACTICAL_BATCH_SIZE = 3  # 仅专项实验模式；正式默认不分批。
        board = empty_board()
        calls = []

        def tactical(b, turn):
            count = sum(cell != 0 for row in b for cell in row)
            if count == 0:
                pts = self.POINTS if root_many else [self.A]
            elif not root_many and count == 1:
                pts = [self.B]
            elif not root_many and count == 2:
                pts = self.POINTS
            else:
                return None, {}
            return 'my-vcf', {p: 100.-i for i, p in enumerate(pts)}

        def candidates(b, turn, n=5, plies=0):
            if (getattr(e, '_batch_test_fail_candidates', False)
                    and sum(cell != 0 for row in b for cell in row) == 2):
                raise AssertionError('缓存内部节点不能重取候选')
            return [(*point, weight) for point, weight in tactical(b, turn)[1].items()]

        def leaf(scorer, b, branch, status, main_player, contexts=None):
            calls.append(tuple(move[:3] for move in branch))
            return float(10+sum(i+1 for i, p in enumerate(self.POINTS) if b[p[0]][p[1]]))

        stack = ExitStack()
        stack.enter_context(patch.object(d, '_tactical_result', side_effect=tactical))
        stack.enter_context(patch.object(d, '_candidate_points', side_effect=candidates))
        stack.enter_context(patch.object(d, '_check_win', return_value=False))
        stack.enter_context(patch.object(d, '_active_threat_points', return_value=frozenset()))
        stack.enter_context(patch.object(DeepSearch, '_leaf_score_impl', new=leaf))
        self.addCleanup(stack.close)
        return e, board, calls

    def test_internal_keeps_seven_expands_three_and_marks_score_incomplete(self):
        for player in (BLACK, WHITE):
            e, board, calls = self.fixture(player)
            before = self.snapshot(e, board)
            rows = e.deep_search.rank_candidates(board, {self.A: 100.}, player)
            d = e.deep_search
            node = d._tree_root['children'][(*self.A, player)]['children'][(*self.B, 3-player)]
            self.assertEqual(len(node['edges']), 7)
            self.assertEqual(len(node['pending_edges']), 4)
            self.assertEqual([edge[:2] for edge in node['pending_edges']], self.POINTS[3:])
            self.assertEqual(len(node['children']), 3)
            self.assertFalse(node['move_result']['complete'])
            self.assertFalse(rows[0]['score_complete'])
            self.assertFalse(d.root_contribution_report['complete'])
            self.assertIsNone(d.root_contribution_report['total'])
            self.assertEqual(node['results'], {})
            self.assertEqual(self.snapshot(e, board), before)

    def test_hot_read_does_not_expand_again_or_issue_temperature(self):
        e, board, calls = self.fixture()
        d = e.deep_search
        first = d.rank_candidates(board, {self.A: 100.}, BLACK)
        n = len(calls)
        second = d.rank_candidates(board, {self.A: 100.}, BLACK)
        self.assertEqual(first, second)
        self.assertEqual(len(calls), n)
        self.assertEqual(d.cache_stats['pending_expansions'], 0)
        self.assertEqual(d.cache_stats['frontier_resumes'], 0)

    def test_real_two_moves_reroot_then_six_branches_without_regeneration(self):
        for player in (BLACK, WHITE):
            e, board, calls = self.fixture(player)
            d = e.deep_search
            d.rank_candidates(board, {self.A: 100.}, player)
            old = d._tree_root['children'][(*self.A, player)]['children'][(*self.B, 3-player)]
            first_values = {key: copy.deepcopy(child['move_result'])
                            for key, child in old['children'].items()}
            for point, turn in ((self.A, player), (self.B, 3-player)):
                board[point[0]][point[1]] = turn
                e.on_move(board, *point, turn)
            self.assertIs(d._tree_root, old)
            before = self.snapshot(e, board)
            # 不替换方法本身，避免改变正式缓存身份而人为造成未命中。
            e._batch_test_fail_candidates = True
            try:
                points, reason, rows = d.cached_turn(board, player)
            finally:
                e._batch_test_fail_candidates = False
            self.assertEqual(len(points), 7)
            self.assertEqual(len(rows), 6)
            self.assertEqual(len(old['children']), 6)
            self.assertEqual(len(old['pending_edges']), 1)
            self.assertEqual(d.cache_stats['pending_expansions'], 3)
            for key, value in first_values.items():
                self.assertEqual(old['children'][key]['move_result'], value)
            self.assertEqual(self.snapshot(e, board), before)
            self.assertFalse(old['move_result']['complete'])
            self.assertFalse(d.root_contribution_report['complete'])

    def test_root_seven_is_batched_and_third_epoch_finishes(self):
        e, board, calls = self.fixture(root_many=True)
        d = e.deep_search
        points = {p: 100.-i for i, p in enumerate(self.POINTS)}
        rows = d.rank_candidates(board, points, BLACK)
        root = d._tree_root
        self.assertEqual(len(rows), 3)
        self.assertEqual(len(root['edges']), 7)
        self.assertEqual(len(root['pending_edges']), 4)
        # 纯调度单测推进真实回合代号；真实落子换根由上一项另行覆盖。
        for expected in (6, 7):
            d._tree_epoch += 1
            _points, _reason, rows = d.cached_turn(board, BLACK)
            self.assertEqual(len(rows), expected)
        self.assertEqual(root['pending_edges'], ())
        self.assertTrue(root['move_result']['complete'])
        self.assertTrue(d.root_contribution_report['complete'])
        expected = sum((100.-i)*(11.+i) for i in range(7))/sum(100.-i for i in range(7))
        self.assertAlmostEqual(d.root_contribution_report['total'], expected)
        reference, ref_board, _ = self.fixture(root_many=True)
        reference.deep_search.TACTICAL_BATCH_SIZE = 7
        full = reference.deep_search.rank_candidates(ref_board, points, BLACK)
        self.assertEqual(rows, full)

    def aggregate(self, states):
        d = Engine().deep_search
        d.TACTICAL_BATCH_SIZE = 3
        parent = d._new_tree_node(bytes(225))
        parent['mover'] = BLACK
        parent['edges'] = tuple((2, i, WHITE, 100.-i, 0., 1.) for i in range(7))
        parent['pending_edges'] = parent['edges'][len(states):]
        for i, state in enumerate(states):
            child = d._new_tree_node(bytes(225))
            child['mover'] = WHITE
            d._store_cached_move(child, float(i+1), state, 0, 0, 1, BLACK)
            parent['children'][(2, i, WHITE)] = child
        return d, parent, d._aggregate_cached_move(parent, BLACK)

    def test_three_losing_replies_do_not_prove_all_seven_losing(self):
        d, node, value = self.aggregate([-1, -1, -1])
        self.assertNotIn(value['state'], (-1, 1))
        self.assertFalse(value['complete'])
        self.assertIsNone(node['contribution_unit'])
        self.assertFalse(node['contribution_complete'])

    def test_existential_win_completes_score_without_expanding_all_replies(self):
        d, node, value = self.aggregate([1, -1, -1])
        self.assertEqual(value['state'], -1)
        self.assertEqual(value['score'], -1000.)
        self.assertTrue(value['complete'])
        self.assertEqual(len(node['pending_edges']), 4)
        self.assertEqual(node['contribution_leaves'], 1)

    def test_all_seven_losing_replies_can_prove_win(self):
        d, node, value = self.aggregate([-1]*7)
        self.assertEqual(value['state'], 1)
        self.assertTrue(value['complete'])

    def test_ordinary_four_not_batched(self):
        e = Engine()
        d = e.deep_search
        board = empty_board()
        node = d._new_tree_node(d._position_key(board))
        items = [(1., 1, c, 50.) for c in (1, 3, 5, 7)]
        with patch.object(d, '_tactical_result', return_value=(None, {})):
            selected = d._cache_edges(board, node, BLACK, items, 160., BLACK)
        self.assertEqual(selected, items)
        self.assertEqual(node['pending_edges'], ())

    def test_original_weights_temperatures_and_copy_are_preserved(self):
        e, board, calls = self.fixture(root_many=True)
        d = e.deep_search
        d.rank_candidates(board, {p: 100.-i for i, p in enumerate(self.POINTS)}, BLACK)
        root = d._tree_root
        self.assertEqual([edge[3] for edge in root['edges']], [100.-i for i in range(7)])
        self.assertEqual([edge[4] for edge in root['edges']], [0.]*7)
        copied = d._new_tree_node(root['board'])
        d._copy_cold_subtree(copied, root)
        copied['pending_edges'] = copied['pending_edges'][1:]
        self.assertEqual(len(root['pending_edges']), 4)
        self.assertEqual(len(copied['pending_edges']), 3)
        self.assertIsNot(root['children'], copied['children'])

    def test_pending_set_does_not_replace_no_common_defense_proof(self):
        e = Engine()
        d = e.deep_search
        board = empty_board()
        before = self.snapshot(e, board)
        with patch.object(d, '_tactical_result', return_value=('opp-vcf', {})):
            rows = d._no_defense_turn(board, BLACK, 'opp-vcf')
        self.assertEqual(rows[0]['state'], -1)
        self.assertEqual(d._tree_root['pending_edges'], ())
        self.assertTrue(d._tree_root['move_result']['complete'])
        self.assertEqual(self.snapshot(e, board), before)

    def test_partial_descendant_invalidates_parent_full_cache_and_contribution(self):
        d, node, _value = self.aggregate([0]*7)
        child = node['children'][node['edges'][0][:3]]
        child['move_result']['complete'] = False
        value = d._aggregate_cached_move(node, BLACK)
        self.assertFalse(value['complete'])
        key = d._result_key(WHITE, BLACK, 0., 1.)
        d._save_tree_result(node, key, (1., 0, 0, 0, 1, 0, ()))
        self.assertEqual(node['results'], {})
        self.assertIsNone(node['contribution_unit'])

    def test_pending_edge_charges_positive_to_negative_only_once(self):
        e, board, _calls = self.fixture(root_many=True)
        d = e.deep_search
        d.T0 = 1.
        node = d._new_tree_node(d._position_key(board))
        node['tactical_count'] = 8  # 下一手扣2，正温跨零，不应算入额外8手。
        d._tree_root = node
        d._tree_signature = d._reuse_signature()
        d._begin_search_cache(board, False)
        d._cache_edges(board, node, BLACK, [(1., *p, 100.) for p in self.POINTS], 1., BLACK)
        node['mover'] = WHITE
        d._walk_cached_edges(board, node, BLACK, [], [])
        self.assertEqual(len(node['children']), 3)
        d._tree_epoch += 1
        d._walk_cached_edges(board, node, BLACK, [], [])
        self.assertEqual(len(node['children']), 6)
        for child in node['children'].values():
            self.assertEqual(child['tactical_count'], 9)
            self.assertEqual(child['extension_state'], (0, 0, 0))
        self.assertEqual([edge[4] for edge in node['edges']], [-1.]*7)

    def test_old_batch_early_stop_does_not_allow_six_new_edges_next_round(self):
        d, node, _value = self.aggregate([0, 0])
        node['selection_reason'] = 'my-vcf'
        node['batch_epoch'] = d._tree_epoch
        node['batch_moves'] = tuple(edge[:3] for edge in node['edges'][:3])
        # 第3条已获准但还没展开；原pending仅存第4..7条。
        node['pending_edges'] = node['edges'][3:]
        d._tree_epoch += 1
        d._activate_tactical_batch(node)
        missing_activated = [move for move in node['batch_moves'] if move not in node['children']]
        self.assertEqual(missing_activated, [edge[:3] for edge in node['edges'][2:5]])
        self.assertEqual(node['pending_edges'], node['edges'][5:])


if __name__ == '__main__':
    unittest.main()
