# -*- coding: utf-8 -*-
"""方向增量缓存：与完整四方向更新等价，模拟后缓存及状态表全部恢复。"""
import copy
import random
import unittest
from unittest.mock import patch

from gomoku.core.deep_search import DeepSearch
from gomoku.core.pattern import PatternAnalyzer
from gomoku.core.search import Search
from gomoku.core.utils import BLACK, WHITE, EMPTY, SIZE, DIRECTIONS, empty_board


class FullDirectionSearch(Search):
    """优化前的更新方式：棋段区内每点、每方完整扫描四方向，不读方向缓存。"""

    def on_move(self, board, r, c, player, journal=None, record_history=True):
        self._write_state(self.sb, r, c, 17, journal)
        self._write_state(self.sw, r, c, 17, journal)
        region = self.scope._both_segment_scope(board, r, c)
        for side, state in ((BLACK, self.sb), (WHITE, self.sw)):
            for rr, cc in region:
                value = self._eval(board, rr, cc, side) if board[rr][cc] == EMPTY else 17
                self._write_state(state, rr, cc, value, journal)
        if record_history:
            self._record_zero_event(board, BLACK)
            self._record_zero_event(board, WHITE)


def snapshot(search, board):
    return copy.deepcopy((board, search.sb, search.sw, search._direction_levels,
                          search.gear, search.zero_history, search.zero_active))


def play(search, board, moves):
    for r, c, player in moves:
        board[r][c] = player
        search.on_move(board, r, c, player)


class DirectionCacheTest(unittest.TestCase):
    def assert_full_rebuild_matches(self, search, board):
        reference = Search(PatternAnalyzer())
        reference.rebuild_all(board)
        self.assertEqual(search.sb, reference.sb)
        self.assertEqual(search.sw, reference.sw)
        for r in range(SIZE):
            for c in range(SIZE):
                if board[r][c] != EMPTY:
                    continue  # 占点缓存不可读，允许保留落子前的值以供回滚。
                index = r * SIZE + c
                for player in (BLACK, WHITE):
                    self.assertEqual(search._direction_levels[player][index],
                                     reference._direction_levels[player][index],
                                     (r, c, player))

    def test_random_moves_match_full_direction_updates(self):
        rng = random.Random(20261019)
        cells = [(r, c) for r in range(SIZE) for c in range(SIZE)]
        for _ in range(4):
            search = Search(PatternAnalyzer())
            reference = FullDirectionSearch(PatternAnalyzer())
            board = empty_board()
            for turn, (r, c) in enumerate(rng.sample(cells, 60)):
                player = BLACK if turn % 2 == 0 else WHITE
                board[r][c] = player
                search.on_move(board, r, c, player)
                reference.on_move(board, r, c, player)
                self.assertEqual(search.sb, reference.sb)
                self.assertEqual(search.sw, reference.sw)
                self.assertEqual(search.zero_history, reference.zero_history)
                self.assertEqual(search.zero_active, reference.zero_active)
                if turn % 10 == 0:
                    self.assert_full_rebuild_matches(search, board)

    def test_only_affected_direction_is_scanned(self):
        search = Search(PatternAnalyzer())
        board = empty_board()
        play(search, board, [(7, 7, BLACK), (6, 6, WHITE)])
        r, c, player = 8, 7, BLACK
        board[r][c] = player
        region = search.scope._both_segment_scope(board, r, c)
        empty_count = sum(board[rr][cc] == EMPTY for rr, cc in region)
        with patch.object(search, '_direction_level', wraps=search._direction_level) as scan:
            search.on_move(board, r, c, player)
        self.assertGreater(empty_count, 0)
        self.assertEqual(scan.call_count, 2 * empty_count)  # 每空点黑白各一个方向。
        for call in scan.call_args_list:
            _, rr, cc, _, dr, dc = call.args
            self.assertEqual((r - rr) * dc, (c - cc) * dr)
        self.assert_full_rebuild_matches(search, board)

    def test_journal_restores_changed_direction_when_subclass_is_unchanged(self):
        search = Search(PatternAnalyzer())
        board = empty_board()
        play(search, board, [(7, c, BLACK) for c in (3, 4, 5, 6)])
        before = snapshot(search, board)
        index = 7 * SIZE + 7
        self.assertEqual(search.sb[7][7], 0)
        old_level = search._direction_levels[BLACK][index][1]
        board[5][7] = BLACK
        journal = {}
        search.on_move(board, 5, 7, BLACK, journal=journal, record_history=False)
        self.assertEqual(search.sb[7][7], 0)
        self.assertNotEqual(search._direction_levels[BLACK][index][1], old_level)
        self.assertIn((id(search._direction_levels[BLACK]), index, 1), journal)
        self.assertNotIn((id(search.sb), 7, 7), journal)
        search.restore(journal)
        board[5][7] = EMPTY
        self.assertEqual(snapshot(search, board), before)

    def test_nested_journals_and_shared_journal_restore_all_state(self):
        search = Search(PatternAnalyzer())
        board = empty_board()
        play(search, board, [(7, 7, BLACK), (6, 6, WHITE), (8, 8, BLACK)])
        moves = [(6, 8, WHITE), (7, 8, BLACK), (5, 7, WHITE), (8, 7, BLACK)]
        for shared in (False, True):
            before = snapshot(search, board)
            journal = {}
            steps = []
            for r, c, player in moves:
                if not shared:
                    journal = {}
                board[r][c] = player
                search.on_move(board, r, c, player, journal=journal, record_history=False)
                steps.append((r, c, journal))
                self.assert_full_rebuild_matches(search, board)
            if shared:
                search.restore(journal)
                for r, c, _ in steps:
                    board[r][c] = EMPTY
            else:
                for r, c, journal in reversed(steps):
                    search.restore(journal)
                    board[r][c] = EMPTY
                    self.assert_full_rebuild_matches(search, board)
            self.assertEqual(snapshot(search, board), before)

    def test_rebuild_replaces_stale_cache_and_supports_next_move(self):
        search = Search(PatternAnalyzer())
        board = empty_board()
        play(search, board, [(7, 7, BLACK), (6, 6, WHITE), (8, 8, BLACK)])
        board[8][8] = EMPTY  # 模拟悔棋，旧缓存必须作废。
        for levels in search._direction_levels.values():
            for point in levels:
                point[:] = [0] * len(DIRECTIONS)
        search.rebuild_all(board)
        self.assert_full_rebuild_matches(search, board)
        self.assertFalse(any(search.zero_history.values()))
        play(search, board, [(6, 8, BLACK)])
        self.assert_full_rebuild_matches(search, board)

    def test_independent_eval_ignores_and_preserves_cached_directions(self):
        search = Search(PatternAnalyzer())
        board = empty_board()
        play(search, board, [(7, 7, BLACK), (6, 6, WHITE)])
        before = snapshot(search, board)
        temporary = copy.deepcopy(board)
        temporary[7][8] = BLACK  # 临时棋盘没有经过 on_move，不得读取方向缓存。
        reference = Search(PatternAnalyzer())
        self.assertEqual(search._eval(temporary, 7, 9, BLACK),
                         reference._eval(temporary, 7, 9, BLACK))
        self.assertEqual(snapshot(search, board), before)
        empty_before = copy.deepcopy(temporary)
        with patch.object(search, '_direction_level', side_effect=RuntimeError('scan failed')):
            with self.assertRaisesRegex(RuntimeError, 'scan failed'):
                search._eval(temporary, 7, 9, BLACK)
        self.assertEqual(temporary, empty_before)
        self.assertEqual(snapshot(search, board), before)

    def test_ranked_and_leaf_counts_match_original_update_for_both_colors(self):
        positions = [
            [(7, 7, BLACK), (6, 6, WHITE), (5, 5, BLACK), (6, 8, WHITE),
             (5, 6, BLACK), (7, 6, WHITE), (4, 6, BLACK), (8, 8, WHITE)],
            [(7, 7, BLACK), (7, 8, WHITE), (10, 8, BLACK), (6, 9, WHITE),
             (8, 7, BLACK), (6, 7, WHITE), (9, 7, BLACK), (8, 6, WHITE),
             (10, 7, BLACK), (11, 7, WHITE), (9, 8, BLACK)],
        ]
        for moves in positions:
            for player in (BLACK, WHITE):
                outputs = []
                for cls in (Search, FullDirectionSearch):
                    search = cls(PatternAnalyzer())
                    board = empty_board()
                    play(search, board, moves)
                    before = snapshot(search, board)
                    reason, points = search.tactical_candidates(board, player)
                    if not reason:
                        points = {(r, c): w for r, c, w in search._fallback(
                            board, player=player, with_score=True)}
                    deep = DeepSearch(search)
                    deep.T0 = 40
                    outputs.append((deep.rank_candidates(board, points, player),
                                    deep.leaf_eval_status()))
                    self.assertEqual(snapshot(search, board), before)
                self.assertEqual(outputs[0], outputs[1])


if __name__ == '__main__':
    unittest.main()
