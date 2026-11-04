# -*- coding: utf-8 -*-
"""纯算法清理契约：拒绝旧网络接口、不导入网络包、保留服务和评分语义。"""
import builtins
import unittest
from unittest.mock import patch

from gomoku import service
from gomoku.core.engine import Engine
from gomoku.core.utils import BLACK, WHITE, empty_board


class PureAlgorithmTest(unittest.TestCase):
    def test_network_configuration_is_removed(self):
        engine = Engine()
        for name in ('use_decision_net', 'decision_net', '_decision_pick', '_ensure_decision_net'):
            self.assertFalse(hasattr(engine, name), name)
        for name in ('use_nn', 'model_path', '_nn_score', 'collect_samples', 'samples'):
            self.assertFalse(hasattr(engine.deep_search, name), name)
        with self.assertRaises(TypeError):
            Engine(use_decision_net=True)
        with self.assertRaises(TypeError):
            service.create_engine(use_decision_net=True)

    def test_direct_moves_without_optional_imports_both_colors(self):
        original_import = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name == 'gomoku.nn' or name.startswith('gomoku.nn.') or name.split('.')[0] in ('torch', 'numpy'):
                raise AssertionError('纯算法入口不应导入 ' + name)
            return original_import(name, *args, **kwargs)
        for player in (BLACK, WHITE):
            with self.subTest(player=player), patch('builtins.__import__', side_effect=guarded):
                board = empty_board()
                for c in (4, 5, 6, 7):
                    board[7][c] = player
                engine = service.create_engine()
                engine.search.rebuild_all(board)
                original = [row[:] for row in board]
                move, ranked, legacy_flag = service.ai_move(board, [], player, engine=engine)
                self.assertFalse(legacy_flag)
                self.assertTrue(ranked)
                self.assertEqual(board, original)
                board[move[0]][move[1]] = player
                self.assertTrue(engine.deep_search._check_win(board, *move, player))

    def test_leaf_values_and_status_remain_algorithmic(self):
        deep = Engine().deep_search
        board = empty_board()
        for player in (BLACK, WHITE):
            self.assertEqual(deep._leaf_score_impl(board, [], 'win', player), deep.WIN_SCORE)
            self.assertEqual(deep._leaf_score_impl(board, [], 'lose', player), -deep.WIN_SCORE)
            deep.reset_leaf_eval_stats()
            expected = deep._stale_result_score(board, player)
            self.assertEqual(deep._leaf_score_impl(board, [], 'stale', player), expected)
            status = deep.leaf_eval_status()
            self.assertEqual(status['mode'], 'heuristic')
            self.assertEqual(status['total'], 1)
            self.assertNotIn('nn', status)
