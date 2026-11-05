# -*- coding: utf-8 -*-
"""整盘前5手稀疏开局库的定向回归：数据完整性、对称查谱、边界与接管语义。"""
import collections
import copy
import unittest
from unittest.mock import patch

from gomoku.core.deep_search import DeepSearch
from gomoku.core.engine import Engine
from gomoku.core.opening_book import (FRONT_STONES_BEFORE, OpeningBook, _opening_move,
                                      canonical_position, inverse_point, transform_point)
from gomoku.core.opening_book_front import PROVENANCE, RECORDS
from gomoku.core.utils import BLACK, WHITE, EMPTY, SIZE, empty_board

FRONT = 'teacher:black-five'
BLACK_THREE_SOURCES = ('teacher:black-three', 'teacher:black-three-root', 'human:black-three')
WHITE_FOUR_SOURCES = ('teacher:white-four', 'human:white-four-merge')


def board_of(black, white):
    """按记录里的（已是规范坐标系的）棋子集合重建盘面。"""
    board = empty_board()
    for cell in black:
        board[cell // SIZE][cell % SIZE] = BLACK
    for cell in white:
        board[cell // SIZE][cell % SIZE] = WHITE
    return board


def rotate_board(board, symmetry):
    changed = empty_board()
    for r in range(SIZE):
        for c in range(SIZE):
            if board[r][c] != EMPTY:
                rr, cc = transform_point(r, c, symmetry)
                changed[rr][cc] = board[r][c]
    return changed


def group_records():
    groups = collections.OrderedDict()
    for row in RECORDS:
        groups.setdefault((row[0], row[1], row[2]), []).append(row)
    return groups


def neutral_rank(board, points, player, **kwargs):
    """接管阶段的搜索替身：只确认谱着被当作唯一根候选送入搜索。"""
    if len(points) != 1:
        raise AssertionError('第1-5手谱着必须固定一个根候选')
    r, c = next(iter(points))
    return [DeepSearch._candidate_result(r, c, 0.)]


class FrontBookDataTest(unittest.TestCase):
    def test_records_are_legal_and_carry_source(self):
        self.assertGreater(len(RECORDS), 0)
        seen = set()
        for row in RECORDS:
            player, black, white, r, c, kind, score, score_kind, view, source = row
            self.assertIn(player, (BLACK, WHITE))
            self.assertTrue(0 <= r < SIZE and 0 <= c < SIZE)
            self.assertNotIn(r * SIZE + c, black)
            self.assertNotIn(r * SIZE + c, white)
            self.assertFalse(set(black) & set(white))
            self.assertEqual(len(black), len(white) + (player == WHITE))
            self.assertIn(kind, ('primary', 'policy', 'coverage', 'unrated'))
            self.assertTrue(source)
            if kind == 'coverage':
                self.assertIsNotNone(score, '已评分覆盖必须带分数')
            if kind == 'unrated':
                self.assertIsNone(score, '待评候选不得伪造分数')
            if score is not None:
                self.assertIn(score_kind, ('cp', 'mate'))
                self.assertIn(view, ('black', 'white'))
            key = (player, black, white, r, c)
            self.assertNotIn(key, seen, '同一盘面同一落点不得重复记录')
            seen.add(key)

    def test_each_position_has_one_primary_and_no_duplicate_points(self):
        for key, rows in group_records().items():
            points = [(row[3], row[4]) for row in rows]
            self.assertEqual(len(points), len(set(points)), key)
            primaries = [row for row in rows if row[5] == 'primary']
            policies = [row for row in rows if row[5] == 'policy']
            self.assertEqual(len(primaries) + (1 if policies else 0), 1, key)
            if policies:
                # 白二只保留正邻/斜邻两种可选方向，其余三类不进本局面。
                self.assertEqual(len(policies), 2, key)
                self.assertEqual(len(rows), 2, key)

    def test_primary_is_inside_the_retained_set(self):
        """没有人工指定时，暂定主着取同局面、同口径的最高分；人工指定不受此限。"""
        human = ('human:black-three', 'human:white-four-merge', 'rule:h8-first',
                 'human:adjacent-two')
        for key, rows in group_records().items():
            primaries = [row for row in rows if row[5] == 'primary']
            if not primaries:
                self.assertTrue([row for row in rows if row[5] == 'policy'], key)
                continue
            primary = primaries[0]
            self.assertIn((primary[3], primary[4]), {(row[3], row[4]) for row in rows})
            if primary[9] in human:
                continue
            rated = [row for row in rows if row[6] is not None]
            if not rated:
                self.assertIsNone(primary[6], key)
                continue
            best = min(rated, key=lambda row: (0 if row[7] == 'mate' else 1,
                                               row[6] if row[7] == 'mate' else -row[6]))
            self.assertEqual((primary[6], primary[7]), (best[6], best[7]), key)

    def test_provenance_matches_records(self):
        kinds = collections.Counter(row[5] for row in RECORDS)
        groups = group_records()
        by_stones = collections.Counter(len(key[1]) + len(key[2]) for key in groups)
        self.assertEqual(PROVENANCE['positions'], len(groups))
        self.assertEqual(PROVENANCE['candidate_moves'], len(RECORDS))
        self.assertEqual(PROVENANCE['primary_moves'], kinds['primary'])
        self.assertEqual(PROVENANCE['policy_moves'], kinds['policy'])
        self.assertEqual(PROVENANCE['unrated_moves'], kinds['unrated'])
        self.assertEqual(PROVENANCE['positions_by_stones'], dict(sorted(by_stones.items())))
        self.assertEqual(PROVENANCE['rule'], 'freestyle15')
        self.assertFalse(PROVENANCE['proven'], '查谱数据不能写成五态硬证明')
        self.assertEqual(PROVENANCE['black_three_confirmed'], 27)
        self.assertEqual(PROVENANCE['white_four_confirmed'], 74)
        self.assertEqual(PROVENANCE['black_five_confirmed'], 429)

    def test_confirmed_input_counts_and_dedup_accounting(self):
        three = sum(1 for row in RECORDS if row[9] in BLACK_THREE_SOURCES)
        four = sum(1 for row in RECORDS if row[9] in WHITE_FOUR_SOURCES)
        five = sum(1 for row in RECORDS if row[9] == FRONT)
        self.assertEqual(three, 27)
        self.assertEqual(four, 74)
        self.assertEqual(five, PROVENANCE['black_five_unique'])
        # 429 条黑五候选在八种旋转/镜像下归并为更少的唯一规范候选：
        # 归并账目必须闭合，不得静默丢弃，也不得后写覆盖。
        self.assertEqual(
            PROVENANCE['black_five_confirmed'],
            PROVENANCE['black_five_unique'] + PROVENANCE['black_five_duplicates']
            + len(PROVENANCE['black_five_score_conflicts']))
        self.assertEqual(len(PROVENANCE['black_five_merged_positions']), 5)

    def test_black_five_retention_rule_holds_per_position(self):
        """每组保底前三，其余普通分严格>400；带杀棋标签的候选不因分数被删。"""
        for key, rows in group_records().items():
            five = [row for row in rows if row[9] == FRONT]
            if not five:
                continue
            low = [row for row in five if row[7] == 'cp' and row[6] <= 400]
            self.assertLessEqual(len(low), 3, key)
            self.assertFalse([row for row in five if row[7] == 'cp' and row[6] == 400], key)
        mates = [row for row in RECORDS if row[9] == FRONT and row[7] == 'mate']
        self.assertGreater(len(mates), 0, '带正杀棋标签的候选必须保留')

    def test_merged_white_four_branch_is_replaced_and_h11_removed(self):
        def branch(white_two, black_three):
            rows = [row for row in RECORDS
                    if row[9] in WHITE_FOUR_SOURCES and row[0] == WHITE]
            base = board_of(*self._pair_stones(white_two, black_three))
            key, _symmetry = canonical_position(base, WHITE)
            return {self._label(*inverse_point(row[3], row[4], _symmetry))
                    for row in rows
                    if (row[0], row[1], row[2]) == key}

        self.assertEqual(branch('I9', 'G8'), {'I7', 'I8', 'F8'})
        self.assertEqual(branch('I10', 'H9'), {'G10', 'H10', 'H7'})
        self.assertNotIn('J9', branch('I10', 'H9'))
        self.assertNotIn('H11', branch('I10', 'H9'))
        # 全候选 75 条中删除的 (I10, G9, H11) 不得重新出现。
        self.assertEqual(PROVENANCE['white_four_full_candidates'], 75)
        self.assertEqual(PROVENANCE['white_four_dropped'], 'I10/G9/H11')
        dropped = self._pair_stones('I10', 'G9')
        self.assertNotIn('H11', branch('I10', 'G9'))

    def test_black_three_confirmed_list_and_unrated_points(self):
        expected = {
            'H9': {'I7', 'I8', 'I9', 'H10', 'I10'},
            'I9': {'I7', 'I8', 'G8', 'J9'},
            'H10': {'G9', 'G10', 'G8', 'F8', 'G7'},
            'I10': {'H9', 'H10', 'I8', 'I9', 'J9', 'G9'},
            'J10': {'I8', 'J8', 'J9', 'K9', 'I7', 'H7', 'H6'},
        }
        self.assertEqual(sum(len(moves) for moves in expected.values()), 27)
        unrated = set()
        for white_two, moves in expected.items():
            rows = self._black_three_rows(white_two)
            self.assertEqual({self._label(*self._absolute(row, white_two)) for row in rows}, moves)
            self.assertNotEqual([row for row in rows if row[5] == 'primary'], [],
                                '每个黑三局面必须有暂定主着')
            for row in rows:
                if row[5] == 'unrated':
                    self.assertIsNone(row[6], '待评点不得伪造分数')
                    unrated.add(self._label(*self._absolute(row, white_two)))
        self.assertEqual(unrated, {'G7', 'G9', 'H6'})
        # 低于 400 已移出的三项不得重新进入正式库。
        removed = {('I9', 'J10'), ('H10', 'F10'), ('I10', 'J10')}
        for white_two, move in removed:
            rows = self._black_three_rows(white_two)
            self.assertNotIn(move, {self._label(*self._absolute(row, white_two))
                                    for row in rows})

    # ---- 辅助 ----

    @staticmethod
    def _label(r, c):
        return '%s%d' % ('ABCDEFGHIJKLMNO'[c], r + 1)

    @staticmethod
    def _pair_stones(white_two, black_three):
        columns = 'ABCDEFGHIJKLMNO'
        offsets = {'H9': (8, 7), 'I9': (8, 8), 'H10': (9, 7), 'I10': (9, 8), 'J10': (9, 9)}

        def point(name):
            return int(name[1:]) - 1, columns.index(name[0])

        black = (7 * SIZE + 7, point(black_three)[0] * SIZE + point(black_three)[1])
        white = (offsets[white_two][0] * SIZE + offsets[white_two][1],)
        return tuple(sorted(black)), white

    def _black_three_rows(self, white_two):
        black, white = self._pair_stones(white_two, 'H9')
        order = 'H9 I9 H10 I10 J10'.split()
        point = {'H9': (8, 7), 'I9': (8, 8), 'H10': (9, 7), 'I10': (9, 8), 'J10': (9, 9)}
        board = board_of((7 * SIZE + 7,), (point[white_two][0] * SIZE + point[white_two][1],))
        key, _symmetry = canonical_position(board, BLACK)
        return [row for row in RECORDS
                if row[0] == BLACK and row[9] in BLACK_THREE_SOURCES
                and (row[0], row[1], row[2]) == key]

    def _absolute(self, row, white_two):
        point = {'H9': (8, 7), 'I9': (8, 8), 'H10': (9, 7), 'I10': (9, 8), 'J10': (9, 9)}
        board = board_of((7 * SIZE + 7,), (point[white_two][0] * SIZE + point[white_two][1],))
        _key, symmetry = canonical_position(board, BLACK)
        return inverse_point(row[3], row[4], symmetry)


class FrontBookLookupTest(unittest.TestCase):
    def test_all_positions_round_trip_under_eight_symmetries(self):
        book = OpeningBook()
        for key, rows in group_records().items():
            player, black, white = key
            board = board_of(black, white)
            stored = {(row[3], row[4]) for row in rows}
            self.assertEqual({choice[0] for choice in book.front_choices(board, player)},
                             stored)
            symmetric = [s for s in range(8) if rotate_board(board, s) == board]
            for symmetry in range(8):
                view = rotate_board(board, symmetry)
                view_points = {choice[0] for choice in book.front_choices(view, player)}
                if len(symmetric) == 1 and view != board:
                    self.assertEqual(view_points,
                                     {transform_point(r, c, symmetry) for r, c in stored},
                                     (key, symmetry))
                else:
                    # 盘面自身存在对称时，命中同一局面的朝向可不唯一；只要求
                    # 拉回规范坐标系后与保存的候选一致，且落点全部合法。
                    _canonical_key, canonical = canonical_position(view, player)
                    self.assertEqual({transform_point(r, c, canonical) for r, c in view_points},
                                     stored, (key, symmetry))
                view_move, _source = book.front_move(view, player, symmetry)
                self.assertIn(view_move, view_points)
                board_move, source = _opening_move(book, board, player, (), symmetry)
                self.assertEqual(board_move, inverse_point(*view_move, symmetry))
                self.assertTrue(source)
                self.assertEqual(board[board_move[0]][board_move[1]], EMPTY)

    def test_stone_insertion_order_does_not_affect_lookup(self):
        book = OpeningBook()
        for key, rows in group_records().items():
            player, black, white = key
            forward = board_of(black, white)
            reversed_board = empty_board()
            for cell in list(black)[::-1]:
                reversed_board[cell // SIZE][cell % SIZE] = BLACK
            for cell in list(white)[::-1]:
                reversed_board[cell // SIZE][cell % SIZE] = WHITE
            self.assertEqual(book.front_choices(forward, player),
                             book.front_choices(reversed_board, player))

    def test_translation_is_not_a_hit(self):
        book = OpeningBook()
        board = empty_board()
        board[7][7] = BLACK
        board[8][7] = WHITE
        board[6][8] = BLACK
        self.assertTrue(book.front_choices(board, WHITE))
        shifted = empty_board()
        for r in range(15):
            for c in range(15):
                if board[r][c] != EMPTY and r + 1 <= 14 and c + 1 <= 14:
                    shifted[r + 1][c + 1] = board[r][c]
        self.assertEqual(book.front_choices(shifted, WHITE), (), '平移不是完整盘面命中')

    def test_white_two_only_offers_the_two_adjacent_families(self):
        book = OpeningBook()
        board = empty_board()
        board[7][7] = BLACK
        moves = set()
        for symmetry in range(8):
            move, _source = _opening_move(book, board, WHITE, (), symmetry)
            moves.add(move)
            self.assertEqual(max(abs(move[0] - 7), abs(move[1] - 7)), 1)
        for point, kind, _score, _kind, _view, _source in book.front_choices(board, WHITE):
            self.assertEqual(max(abs(point[0] - 7), abs(point[1] - 7)), 1,
                             '不得把其他三类白二当成我方可选着')
            self.assertEqual(kind, 'policy')
        expected = {(r, c) for r in range(6, 9) for c in range(6, 9) if (r, c) != (7, 7)}
        self.assertEqual(moves, expected, '正邻、斜邻各四个方向，不扩大到其他三类')

    def test_primary_selection_does_not_depend_on_symmetry_split(self):
        """只有白二用视角高位选择正邻/斜邻；其他局面任何视角都给同一主着。"""
        book = OpeningBook()
        for key, rows in group_records().items():
            player, black, white = key
            if any(row[5] == 'policy' for row in rows):
                continue
            board = board_of(black, white)
            picks = {book.front_move(board, player, symmetry)[0] for symmetry in range(8)}
            self.assertEqual(len(picks), 1, key)


class FrontBookBoundaryTest(unittest.TestCase):
    def test_first_five_plies_take_over_for_both_colors(self):
        for symmetry in (0, 3):
            engine = Engine()
            engine._opening_symmetry = symmetry
            board = empty_board()
            taken = []
            for ply in range(1, 6):
                player = BLACK if ply % 2 else WHITE
                with patch.object(engine.deep_search, 'rank_candidates',
                                  side_effect=neutral_rank) as rank:
                    result = engine.analyze_turn(board, player)
                move = result['part4_result']['move']
                rank.assert_called_once()
                self.assertEqual(list(rank.call_args.args[1]), [move],
                                 '谱着只能作为唯一根候选交给深推')
                self.assertEqual(engine.opening_source, 'book')
                self.assertEqual(result['part4_result']['type'], 'searched')
                self.assertEqual(board[move[0]][move[1]], EMPTY)
                self.assertEqual(result['part4_result']['ranked'][0]['state'], 0,
                                 '查谱不得伪造五态结论')
                taken.append((ply, move))
                board[move[0]][move[1]] = player
            self.assertEqual(len({move for _ply, move in taken}), 5, '前5手不得重复占点')
            # 第6手：盘面已有5子，新库退出，不再接管。
            engine2 = Engine()
            engine2._opening_symmetry = symmetry
            engine2.reset([{'r': r, 'c': c, 'player': BLACK if ply % 2 else WHITE}
                           for ply, (r, c) in taken])
            with patch.object(engine2.deep_search, 'rank_candidates',
                              return_value=[DeepSearch._candidate_result(0, 0, 0.)]):
                result = engine2.analyze_turn(board, WHITE)
            self.assertIsNone(engine2.opening_detail)
            self.assertEqual(engine2.opening_book.front_choices(board, WHITE), ())
            self.assertEqual(result['part4_result']['type'], 'searched')

    def test_fifth_ply_requires_full_black_five_position(self):
        engine = Engine()
        board = empty_board()
        for ply, (r, c) in enumerate(((7, 7), (8, 7), (6, 8), (8, 6)), start=1):
            board[r][c] = BLACK if ply % 2 else WHITE
        self.assertEqual(len(engine.opening_book.front_choices(board, BLACK)) > 0, True)
        board[0][0] = BLACK
        board[0][1] = WHITE
        self.assertEqual(engine.opening_book.front_choices(board, BLACK), (),
                         '第5手只认完整前四手盘面')

    def test_early_out_of_book_returns_to_original_search(self):
        engine = Engine()
        board = empty_board()
        board[7][7] = BLACK
        board[9][9] = WHITE          # 白二 J10 在库内
        board[0][0] = BLACK          # 但黑三不在确认清单
        self.assertEqual(engine.opening_book.front_choices(board, WHITE), ())
        with patch.object(engine.deep_search, 'rank_candidates',
                          return_value=[DeepSearch._candidate_result(5, 5, 0.)]) as rank:
            result = engine.analyze_turn(board, WHITE)
        self.assertTrue(rank.called)
        self.assertEqual(result['part4_result']['type'], 'searched')
        self.assertIsNone(engine.opening_detail)

    def test_tactical_priority_still_wins_over_front_book(self):
        for owner in (BLACK, WHITE):
            engine = Engine()
            board = empty_board()
            for c in range(4):
                board[7][c] = owner
            engine.search.rebuild_all(board)
            result = engine.analyze_turn(board, 3 - owner if owner == BLACK else BLACK)
            self.assertEqual(result['part4_result']['move'], (7, 4))

    def test_public_return_shape_and_custom_book_entry_stay_compatible(self):
        engine = Engine()
        board = empty_board()
        self.assertIsInstance(engine.compute_move(board, BLACK), list)
        self.assertEqual(engine.compute_move(board, BLACK), [(7, 7)])
        custom = OpeningBook([(BLACK, (), (), 7, 7)])
        self.assertEqual(custom.lookup(board, BLACK), (7, 7))
        move, source = custom.front_move(board, BLACK, 0)
        self.assertEqual(move, (7, 7))
        self.assertTrue(source)

    def test_real_move_keeps_root_switch_working_across_the_front_book(self):
        """真实落子推进状态表并换根后，第2手仍是库内应手且不破坏状态。"""
        engine = Engine()
        board = empty_board()
        engine.search.rebuild_all(board)
        with patch.object(engine.deep_search, 'rank_candidates', side_effect=neutral_rank):
            first = engine.analyze_turn(board, BLACK)['part4_result']['move']
        self.assertEqual(first, (7, 7))
        board[7][7] = BLACK
        engine.on_move(board, 7, 7, BLACK)
        before = copy.deepcopy((board, engine.search.sb, engine.search.sw,
                                engine.search._direction_levels, list(engine.moves)))
        with patch.object(engine.deep_search, 'rank_candidates',
                          return_value=[DeepSearch._candidate_result(8, 7, 0.)]):
            result = engine.analyze_turn(board, WHITE)
        move = result['part4_result']['move']
        self.assertEqual(board[move[0]][move[1]], EMPTY)
        # 白二只允许正邻/斜邻两类：仍是一格相邻，其他三类白二不得被我方采用。
        self.assertEqual(max(abs(move[0] - 7), abs(move[1] - 7)), 1)
        self.assertEqual(len(engine.opening_book.front_choices(board, WHITE)), 2)
        self.assertEqual((board, engine.search.sb, engine.search.sw,
                          engine.search._direction_levels, list(engine.moves)), before)

    def test_simulated_search_state_is_restored_around_takeover(self):
        engine = Engine()
        board = empty_board()
        board[7][7] = BLACK
        board[8][7] = WHITE
        engine.search.rebuild_all(board)
        before = copy.deepcopy((board, engine.search.sb, engine.search.sw,
                                engine.search._direction_levels, engine.moves))
        with patch.object(engine.deep_search, 'rank_candidates',
                          return_value=[DeepSearch._candidate_result(6, 8, 0.)]):
            engine.analyze_turn(board, BLACK)
        self.assertEqual((board, engine.search.sb, engine.search.sw,
                          engine.search._direction_levels, engine.moves), before)


if __name__ == '__main__':
    unittest.main()
