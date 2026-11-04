# -*- coding: utf-8 -*-
"""根分整合：继承份额与贡献分的定向手算测试。"""
import copy
import unittest

from gomoku.core.deep_search import DeepSearch, BLACK, WHITE
from gomoku.core.pattern import PatternAnalyzer
from gomoku.core.search import Search


def edge(r, c, player, weight):
    return (r, c, player, float(weight), 0.0, 1.0)


def leaf(score, complete=True):
    return {
        'edges': (),
        'children': {},
        'move_result': {'score': score, 'complete': complete},
        'frontier': None,
        'results': {},
        'resume_values': {},
        'resume_source': None,
    }


def inner(edges, children):
    return {
        'edges': tuple(edges),
        'children': dict(children),
        'move_result': None,
        'frontier': None,
        'results': {},
        'resume_values': {},
        'resume_source': None,
    }


class GuardedNode(dict):
    """只允许读取已保存汇总，访问后代字段即失败。"""
    def __getitem__(self, key):
        if key in ('edges', 'children'):
            raise AssertionError('读取了后代结构: ' + key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key in ('edges', 'children'):
            raise AssertionError('读取了后代结构: ' + key)
        return super().get(key, default)



class RootContributionTest(unittest.TestCase):
    def setUp(self):
        self.d = DeepSearch(Search(PatternAnalyzer()))

    def _prime(self, node):
        for child in node.get('children', {}).values():
            self._prime(child)
        self.d._refresh_node_contribution(node)

    def test_parent_40_splits_100_60_40(self):
        parent = inner(
            [edge(0, 0, BLACK, 100), edge(0, 1, BLACK, 60), edge(0, 2, BLACK, 40)],
            {(0, 0, BLACK): leaf(100), (0, 1, BLACK): leaf(100), (0, 2, BLACK): leaf(100)})
        self._prime(parent)
        value, complete, leaves = self.d._subtree_contribution(parent, 40.0)
        self.assertTrue(complete)
        self.assertAlmostEqual(value, 40.0, places=9)
        self.assertEqual(leaves, 3)
        shares = []
        for e in parent['edges']:
            shares.append(40.0 * e[3] / 200.0)
        self.assertEqual([round(x, 9) for x in shares], [20.0, 12.0, 8.0])

    def test_equal_binary_layers_match(self):
        root = inner(
            [edge(0, 0, BLACK, 1), edge(0, 1, BLACK, 1)],
            {
                (0, 0, BLACK): inner(
                    [edge(1, 0, BLACK, 1), edge(1, 1, BLACK, 1)],
                    {(1, 0, BLACK): leaf(100), (1, 1, BLACK): leaf(100)}),
                (0, 1, BLACK): inner(
                    [edge(2, 0, BLACK, 1), edge(2, 1, BLACK, 1)],
                    {(2, 0, BLACK): leaf(100), (2, 1, BLACK): leaf(-100)}),
            })
        self._prime(root)
        value, complete, leaves = self.d._subtree_contribution(root, 100.0)
        self.assertTrue(complete)
        self.assertAlmostEqual(value, 50.0, places=9)
        self.assertEqual(leaves, 4)
        # 根层直接相加与叶子公式同值
        left, _, _ = self.d._subtree_contribution(root['children'][(0, 0, BLACK)], 50.0)
        right, _, _ = self.d._subtree_contribution(root['children'][(0, 1, BLACK)], 50.0)
        self.assertAlmostEqual(left + right, value, places=9)

    def test_uneven_depth_includes_shallow_leaf(self):
        root = inner(
            [edge(0, 0, BLACK, 1), edge(0, 1, BLACK, 1)],
            {
                (0, 0, BLACK): leaf(100),
                (0, 1, BLACK): inner(
                    [edge(1, 0, BLACK, 1), edge(1, 1, BLACK, 1)],
                    {(1, 0, BLACK): leaf(100), (1, 1, BLACK): leaf(100)}),
            })
        self._prime(root)
        value, complete, leaves = self.d._subtree_contribution(root, 100.0)
        self.assertTrue(complete)
        self.assertAlmostEqual(value, 100.0, places=9)
        self.assertEqual(leaves, 3)

    def test_single_child_inherits_all(self):
        parent = inner([edge(0, 0, BLACK, 7)], {(0, 0, BLACK): leaf(100)})
        self._prime(parent)
        value, complete, leaves = self.d._subtree_contribution(parent, 100.0)
        self.assertTrue(complete)
        self.assertAlmostEqual(value, 100.0, places=9)
        self.assertEqual(leaves, 1)

    def test_positive_negative_zero(self):
        parent = inner(
            [edge(0, 0, BLACK, 1), edge(0, 1, BLACK, 1), edge(0, 2, BLACK, 1)],
            {(0, 0, BLACK): leaf(100), (0, 1, BLACK): leaf(-100), (0, 2, BLACK): leaf(0)})
        self._prime(parent)
        value, complete, leaves = self.d._subtree_contribution(parent, 100.0)
        self.assertTrue(complete)
        self.assertAlmostEqual(value, 0.0, places=9)
        self.assertEqual(leaves, 3)

    def test_pruned_edges_only(self):
        parent = inner(
            [edge(0, 0, BLACK, 100), edge(0, 1, BLACK, 40)],
            {(0, 0, BLACK): leaf(100), (0, 1, BLACK): leaf(100)})
        self._prime(parent)
        value, complete, _ = self.d._subtree_contribution(parent, 100.0)
        self.assertTrue(complete)
        self.assertAlmostEqual(value, 100.0, places=9)
        # 已剔除的 60 权重不参与直接兄弟归一，path_weight 不参与份额。
        self.assertAlmostEqual(parent['edges'][0][3] / 140.0, 100.0 / 140.0, places=9)

    def test_unknown_branch_stays_unknown(self):
        parent = inner(
            [edge(0, 0, BLACK, 1), edge(0, 1, BLACK, 1)],
            {(0, 0, BLACK): leaf(100), (0, 1, BLACK): leaf(None, complete=False)})
        self._prime(parent)
        value, complete, leaves = self.d._subtree_contribution(parent, 100.0)
        self.assertIsNone(value)
        self.assertFalse(complete)
        self.assertEqual(leaves, 2)

    def test_root_report_and_ranked_not_mutated(self):
        root = inner(
            [edge(0, 0, BLACK, 100), edge(0, 1, BLACK, 60), edge(0, 2, BLACK, 40)],
            {(0, 0, BLACK): leaf(100), (0, 1, BLACK): leaf(-100), (0, 2, BLACK): leaf(0)})
        self.d._tree_root = root
        ranked = [
            {'r': 0, 'c': 0, 'score': 7.0, 'state': 0, 'state_name': '无'},
            {'r': 0, 'c': 1, 'score': -3.0, 'state': -1, 'state_name': '必败'},
            {'r': 0, 'c': 2, 'score': 1.0, 'state': 0, 'state_name': '无'},
        ]
        self._prime(root)
        before = copy.deepcopy((root, ranked))
        self.d._refresh_root_contribution_report(ranked)
        report = self.d.root_contribution_report
        self.assertTrue(report['complete'])
        self.assertEqual(report['root_share'], 100.0)
        self.assertEqual([round(x['share'], 9) for x in report['roots']],
                         [50.0, 30.0, 20.0])
        self.assertEqual([round(x['contribution'], 9) for x in report['roots']],
                         [50.0, -30.0, 0.0])
        self.assertAlmostEqual(report['total'], 20.0, places=9)
        self.assertEqual([x['score'] for x in report['roots']], [7.0, -3.0, 1.0])
        self.assertEqual((root, ranked), before)

    def test_report_recomputed_not_accumulated_and_new_root(self):
        root = inner([edge(0, 0, BLACK, 1)], {(0, 0, BLACK): leaf(100)})
        self._prime(root)
        self.d._tree_root = root
        ranked = [{'r': 0, 'c': 0, 'score': 1.0, 'state': 0, 'state_name': '无'}]
        self.d._refresh_root_contribution_report(ranked)
        first = self.d.root_contribution_report['total']
        self.d._refresh_root_contribution_report(ranked)
        self.assertEqual(self.d.root_contribution_report['total'], first)
        new_root = inner([edge(0, 0, WHITE, 1)], {(0, 0, WHITE): leaf(50)})
        self.d._tree_root = new_root
        self._prime(new_root)
        self.d._refresh_root_contribution_report(
            [{'r': 0, 'c': 0, 'score': 2.0, 'state': 0, 'state_name': '无'}])
        self.assertEqual(self.d.root_contribution_report['total'], 50.0)

    def test_no_root_edges_unknown(self):
        self.d._tree_root = None
        self.d._refresh_root_contribution_report([])
        self.assertFalse(self.d.root_contribution_report['complete'])
        self.assertIsNone(self.d.root_contribution_report['total'])

    def test_stored_parent_summary_does_not_read_grandchildren(self):
        child = GuardedNode(contribution_unit=100.0,
                            contribution_complete=True,
                            contribution_leaves=1)
        root = inner([edge(0, 0, BLACK, 1)], {(0, 0, BLACK): child})
        self.d._tree_root = root
        self.d._refresh_root_contribution_report(
            [{'r': 0, 'c': 0, 'score': 1.0, 'state': 0}])
        self.assertEqual(self.d.root_contribution_report['total'], 100.0)
        self.assertEqual(self.d.root_contribution_report['leaf_count'], 1)

    def test_hot_report_reads_only_immediate_summary(self):
        child = GuardedNode(contribution_unit=100.0,
                            contribution_complete=True,
                            contribution_leaves=1)
        root = inner([edge(0, 0, BLACK, 1)], {(0, 0, BLACK): child})
        self.d._tree_root = root
        ranked = [{'r': 0, 'c': 0, 'score': 1.0, 'state': 0}]
        self.d._refresh_root_contribution_report(ranked)
        first = self.d.root_contribution_report['total']
        self.d._refresh_root_contribution_report(ranked)
        self.assertEqual(self.d.root_contribution_report['total'], first)

    def test_root_leaf_no_edges_counts_100(self):
        root = leaf(-700.0)
        self.d._tree_root = root
        self.d._refresh_node_contribution(root)
        self.d._refresh_root_contribution_report([])
        r = self.d.root_contribution_report
        self.assertTrue(r['complete'])
        self.assertEqual(r['total'], -700.0)
        self.assertEqual(r['leaf_count'], 1)
        self.assertEqual(r['roots'], [])
        self.assertEqual(r['reason'], 'root_leaf')

    def test_incomplete_move_result_ignores_old_frontier(self):
        node = leaf(100.0, complete=False)
        node['frontier'] = {'score': 50.0}
        self.d._refresh_node_contribution(node)
        self.assertIsNone(node['contribution_unit'])
        self.assertFalse(node['contribution_complete'])
        self.d._tree_root = node
        self.d._refresh_root_contribution_report([])
        self.assertIsNone(self.d.root_contribution_report['total'])
        self.assertFalse(self.d.root_contribution_report['complete'])

    def test_child_update_propagates_and_does_not_accumulate(self):
        child = leaf(100.0)
        parent = inner([edge(0, 0, BLACK, 1)], {(0, 0, BLACK): child})
        self._prime(parent)
        self.assertEqual(parent['contribution_unit'], 100.0)
        child['move_result'] = {'score': 200.0, 'complete': True}
        self.d._refresh_node_contribution(child)
        self.d._refresh_node_contribution(parent)
        self.assertEqual(parent['contribution_unit'], 200.0)
        self.d._refresh_node_contribution(parent)
        self.assertEqual(parent['contribution_unit'], 200.0)

    def test_shared_subtree_copy_has_no_absolute_share(self):
        shared = leaf(100.0)
        a = inner([edge(0, 0, BLACK, 1)], {(0, 0, BLACK): shared})
        b = inner([edge(0, 1, BLACK, 1)], {(0, 1, BLACK): shared})
        self._prime(a)
        self._prime(b)
        self.assertEqual(self.d._subtree_contribution(shared, 50.0)[0], 50.0)
        self.assertEqual(self.d._subtree_contribution(shared, 100.0)[0], 100.0)
        self.assertEqual(shared['contribution_unit'], 100.0)
        copied = {}
        self.d._copy_cold_subtree(copied, a)
        self.assertEqual(copied['contribution_unit'], 100.0)
        self.assertEqual(copied['children'][(0, 0, BLACK)]['contribution_unit'], 100.0)

    def test_direct_path_clears_old_report(self):
        from gomoku.core.engine import Engine
        from gomoku.core.utils import empty_board, place
        board = empty_board()
        for c in range(3, 7):
            place(board, 7, c, BLACK)
        place(board, 0, 0, WHITE)
        eng = Engine()
        eng.search.rebuild_all(board)
        eng.deep_search.root_contribution_report = {'stale': True}
        result = eng.analyze_turn(board, BLACK)
        self.assertIsNone(eng.deep_search.root_contribution_report)
        self.assertEqual(result['part4_result']['type'], 'direct')

    def _real_engine(self, board):
        from gomoku.core.engine import Engine
        eng = Engine()
        eng.search.rebuild_all(board)
        eng.deep_search.T0 = 160.0
        eng.deep_search.use_nn = False
        eng.deep_search.collect_samples = False
        eng.opening_book.lookup = lambda b, p: None
        eng._memory_bonus = lambda: {}
        eng._opening_symmetry = 0
        eng.search.gear = {BLACK: 3, WHITE: 3}
        return eng

    def test_real_analysis_report_hot_and_ranking_unchanged(self):
        from gomoku.tests.test_compound_defense import picture
        _, board = picture('two43')
        board_before = copy.deepcopy(board)
        eng = self._real_engine(board)
        first = eng.analyze_turn(board, BLACK)
        ranked = copy.deepcopy(first['part4_result']['ranked'])
        report = copy.deepcopy(eng.deep_search.root_contribution_report)
        self.assertIsNotNone(report)
        self.assertTrue(report['roots'])
        eng.deep_search.root_contribution_report = {'stale': True}
        second = eng.analyze_turn(board, BLACK)
        self.assertEqual(second['part4_result']['ranked'], ranked)
        self.assertNotEqual(eng.deep_search.root_contribution_report, {'stale': True})
        self.assertEqual(eng.deep_search.root_contribution_report['total'],
                         report['total'])
        self.assertEqual(board, board_before)
        self.assertEqual(eng.search.gear, {BLACK: 3, WHITE: 3})

    def test_report_refresh_does_not_change_analysis_result(self):
        from gomoku.tests.test_compound_defense import picture
        _, board_a = picture('two43')
        _, board_b = picture('two43')
        eng_no = self._real_engine(board_a)
        eng_no.deep_search._refresh_root_contribution_report = lambda ranked: None
        result_no = eng_no.analyze_turn(board_a, BLACK)
        eng = self._real_engine(board_b)
        result = eng.analyze_turn(board_b, BLACK)
        self.assertEqual(result_no['part4_result'], result['part4_result'])
        self.assertEqual(result_no['part3_candidates'], result['part3_candidates'])

    def test_real_analysis_white_report_and_rollback(self):
        from gomoku.tests.test_compound_defense import picture
        _, board = picture('disjoint')
        eng = self._real_engine(board)
        before = copy.deepcopy((board, eng.moves, eng.search.sb, eng.search.sw,
                                eng.search._direction_levels, eng.search.zero_history,
                                eng.search.zero_active))
        result = eng.analyze_turn(board, WHITE)
        report = eng.deep_search.root_contribution_report
        self.assertIsNotNone(report)
        self.assertEqual((board, eng.moves, eng.search.sb, eng.search.sw,
                          eng.search._direction_levels, eng.search.zero_history,
                          eng.search.zero_active), before)
        self.assertIn('ranked', result['part4_result'])

    def test_real_no_common_root_leaf(self):
        from gomoku.tests.test_compound_defense import picture
        _, board = picture('disjoint')
        eng = self._real_engine(board)
        result = eng.analyze_turn(board, BLACK)
        report = eng.deep_search.root_contribution_report
        self.assertIsNotNone(report)
        self.assertTrue(report['complete'])
        self.assertEqual(report['leaf_count'], 1)
        self.assertEqual(report['roots'], [])
        self.assertEqual(report['total'], result['part4_result']['ranked'][0]['score'])

    def test_real_on_move_reroot_gets_new_100_share_report(self):
        from gomoku.core.utils import place
        from gomoku.tests.test_compound_defense import picture
        _, board = picture('two43')
        eng = self._real_engine(board)
        first = eng.analyze_turn(board, BLACK)
        move = first['part4_result']['move']
        r, c = move
        place(board, r, c, BLACK)
        eng.on_move(board, r, c, BLACK)
        result = eng.analyze_turn(board, WHITE)
        report = eng.deep_search.root_contribution_report
        self.assertIsNotNone(report)
        self.assertEqual(report['root_share'], 100.0)
        self.assertEqual(eng.deep_search._tree_root['board'],
                         eng.deep_search._position_key(board))
        self.assertIn('ranked', result['part4_result'])


if __name__ == '__main__':
    unittest.main()
