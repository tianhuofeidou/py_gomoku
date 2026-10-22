# -*- coding: utf-8 -*-
"""按本方落子次数接管开局，随后谱点只参与根候选的回归。"""
import copy
import unittest
from unittest.mock import patch

from gomoku import service
from gomoku.core.engine import Engine
from gomoku.core.opening_book import (OpeningBook, canonical_position, transform_point,
                                      inverse_point, opening_fallback)
from gomoku.core.opening_book_data import RECORDS, PROVENANCE
from gomoku.core.utils import BLACK, WHITE, empty_board
from gomoku.session import PlaySession


def record(board, player, move):
    key, symmetry = canonical_position(board, player)
    return (*key, *transform_point(*move, symmetry))


def setup_position(count):
    board = empty_board()
    moves = [(7, 7, BLACK), (6, 7, WHITE), (5, 9, BLACK), (8, 5, WHITE),
             (9, 8, BLACK), (4, 6, WHITE), (3, 3, BLACK), (10, 10, WHITE)]
    engine = Engine()
    engine._opening_symmetry = 0  # 固定方向以验证原始谱着；随机方向另有专项覆盖。
    for r, c, p in moves[:count]:
        board[r][c] = p
        engine.on_move(board, r, c, p)
    return engine, board, [{'r': r, 'c': c, 'player': p} for r, c, p in moves[:count]]


class OpeningBookTest(unittest.TestCase):
    def test_coordinate_transform_round_trip(self):
        for symmetry in range(8):
            for r in range(15):
                for c in range(15):
                    self.assertEqual(inverse_point(*transform_point(r, c, symmetry), symmetry), (r, c))

    def test_lookup_matches_whole_board_and_color_under_all_symmetries(self):
        _engine, board, _moves = setup_position(6)
        move = (1, 2)
        book = OpeningBook([record(board, BLACK, move)])
        for symmetry in range(8):
            changed = empty_board()
            for r in range(15):
                for c in range(15):
                    rr, cc = transform_point(r, c, symmetry)
                    changed[rr][cc] = board[r][c]
            self.assertEqual(book.lookup(changed, BLACK), transform_point(*move, symmetry))
            self.assertIsNone(book.lookup(changed, WHITE))
        board[14][14] = WHITE
        self.assertIsNone(book.lookup(board, BLACK), '不能把局部形状匹配当成完整局面命中')

    def test_packaged_records_have_legal_consistent_moves(self):
        self.assertEqual(PROVENANCE['rule'], 'freestyle15')
        self.assertFalse(PROVENANCE['proven'])
        self.assertEqual(len(RECORDS), PROVENANCE['positions'])
        self.assertGreater(len(RECORDS), 0)
        book = OpeningBook()
        for player, black, white, r, c in RECORDS:
            board = empty_board()
            for color, stones in ((BLACK, black), (WHITE, white)):
                for cell in stones:
                    board[cell // 15][cell % 15] = color
            self.assertEqual(book.lookup(board, player), (r, c))
            self.assertEqual(board[r][c], 0)
            self.assertEqual(len(black), len(white) + (player == WHITE))

    def test_both_colors_first_three_moves_skip_deep_search(self):
        for player, counts in ((BLACK, (0, 2, 4)), (WHITE, (1, 3, 5))):
            for count in counts:
                with self.subTest(player=player, count=count):
                    engine, board, moves = setup_position(count)
                    desired = (2, 1)
                    engine.opening_book = OpeningBook([record(board, player, desired)])
                    before = copy.deepcopy((board, engine.search.sb, engine.search.sw,
                                            engine.search._direction_levels, engine.moves))
                    engine.deep_search._leaf_score(board, [], 'stale', player, [])
                    with patch.object(engine.deep_search, 'rank_candidates', side_effect=AssertionError('开局不能深推')):
                        move, ranked, net = service.ai_move(board, moves, player, engine)
                    self.assertEqual(move, desired)
                    self.assertEqual(engine.opening_source, 'book')
                    self.assertEqual(ranked[0]['state'], 0, '谱着不能伪造必胜')
                    self.assertFalse(net)
                    self.assertEqual(engine.deep_search.leaf_eval_status()['total'], 0)
                    self.assertEqual((board, engine.search.sb, engine.search.sw,
                                      engine.search._direction_levels, engine.moves), before)

    def test_random_orientation_only_chooses_board_preserving_equivalents(self):
        for player, counts in ((BLACK, (0, 2, 4)), (WHITE, (1, 3, 5))):
            for count in counts:
                engine, board, _moves = setup_position(count)
                desired = (2, 1)
                engine.opening_book = OpeningBook([record(board, player, desired)])
                allowed = set()
                for symmetry in range(8):
                    changed = empty_board()
                    for r in range(15):
                        for c in range(15):
                            rr, cc = transform_point(r, c, symmetry)
                            changed[rr][cc] = board[r][c]
                    if changed == board:
                        allowed.add(transform_point(*desired, symmetry))
                before = copy.deepcopy((board, engine.search.sb, engine.search.sw,
                                        engine.search._direction_levels, engine.moves))
                actual = []
                for symmetry in range(8):
                    with self.subTest(player=player, count=count, symmetry=symmetry):
                        engine._opening_symmetry = symmetry
                        with patch.object(engine.deep_search, 'rank_candidates', side_effect=AssertionError('开局不能深推')):
                            result = engine.analyze_turn(board, player)
                            move = result['part4_result']['move']
                            self.assertEqual(engine.compute_move(board, player), [move])
                            self.assertEqual(engine.analyze_turn(board, player)['part4_result']['move'], move)
                        actual.append(move)
                        self.assertIn(move, allowed)
                        self.assertEqual(board[move[0]][move[1]], 0)
                        self.assertEqual(result['part4_result']['ranked'][0]['state'], 0)
                        self.assertEqual((board, engine.search.sb, engine.search.sw,
                                          engine.search._direction_levels, engine.moves), before)
                self.assertEqual(set(actual), allowed)

    def test_packaged_white_first_move_varies_in_four_equal_directions(self):
        engine, board, _moves = setup_position(1)
        moves = []
        for symmetry in range(8):
            engine._opening_symmetry = symmetry
            moves.append(engine.analyze_turn(board, WHITE)['part4_result']['move'])
        expected = {(5, 7), (9, 7), (7, 5), (7, 9)}
        self.assertEqual(set(moves), expected)
        for move in expected:
            self.assertEqual(moves.count(move), 2, '八个等概率方向不能因镜像重复而偏向某个落点')

    def test_random_orientation_keeps_rules_and_tactical_weights(self):
        for player in (BLACK, WHITE):
            engine = Engine()
            engine.opening_book = OpeningBook(())
            board = empty_board()
            board[7][7] = 3 - player
            engine.search.rebuild_all(board)
            tactical = {(5, 7): 100.0, (9, 7): 80.0}
            for symmetry in range(8):
                engine._opening_symmetry = symmetry
                with patch.object(engine.search, 'tactical_candidates', return_value=('opp-vcf', tactical)):
                    result = engine.analyze_turn(board, player)
                self.assertEqual(result['part4_result']['move'], (5, 7))
                self.assertEqual(engine.opening_source, 'rules')
                self.assertEqual(tactical, {(5, 7): 100.0, (9, 7): 80.0})
        engine, board, _moves = setup_position(1)
        engine.opening_book = OpeningBook(())
        moves = set()
        for symmetry in range(8):
            engine._opening_symmetry = symmetry
            moves.add(engine.analyze_turn(board, WHITE)['part4_result']['move'])
        self.assertEqual(moves, {opening_fallback(board, WHITE)})

    def test_new_game_redraws_orientation_but_undo_and_rebuild_preserve_it(self):
        with patch('gomoku.core.engine.SystemRandom') as random_source:
            draw = random_source.return_value.randrange
            draw.side_effect = [0, 1, 2]
            session = PlaySession('opening-direction-test')
            self.assertEqual(session.engine._opening_symmetry, 0)
            session.reset(mode='pvp')
            self.assertEqual(session.engine._opening_symmetry, 1)
            session.place(7, 7, BLACK)
            hint = session.analyze(WHITE)[0]
            self.assertEqual(session.ai_play(WHITE)[0], hint)
            session.undo('last')
            self.assertEqual(session.engine._opening_symmetry, 1)
            self.assertEqual(session.analyze(WHITE)[0], hint)
            session.undo('last')
            self.assertEqual(session.engine._opening_symmetry, 1, '悔棋到空盘仍是同一局')
            session.place(7, 7, BLACK)
            self.assertEqual(session.analyze(WHITE)[0], hint)
            session.engine.reset(session.game.moves)
            self.assertEqual(session.engine._opening_symmetry, 1)
            book = session.engine.opening_book
            temperature = session.engine.deep_search.T0
            session.reset(mode='pvp')
            self.assertEqual(session.engine._opening_symmetry, 2)
            self.assertIs(session.engine.opening_book, book)
            self.assertEqual(session.engine.deep_search.T0, temperature)
            self.assertEqual(draw.call_count, 3)

    def test_fourth_move_keeps_original_book_candidate_for_all_orientations(self):
        engine = Engine()
        board = empty_board()
        for r, color in ((7, BLACK), (6, BLACK), (8, BLACK),
                         (5, WHITE), (9, WHITE), (4, WHITE)):
            board[r][7] = color
        engine.search.rebuild_all(board)
        desired = (7, 6)
        engine.opening_book = OpeningBook([record(board, BLACK, desired)])
        points = [(0, 0, 10.0), (0, 1, 9.0)]
        for symmetry in range(8):
            engine._opening_symmetry = symmetry
            with (patch.object(engine.search, 'tactical_candidates', return_value=(None, {})),
                  patch.object(engine.search, '_fallback', return_value=points),
                  patch.object(engine.deep_search, 'rank_candidates', return_value=[]) as rank):
                engine.analyze_turn(board, BLACK)
            self.assertIn(desired, rank.call_args.args[1])
            self.assertNotIn((7, 8), rank.call_args.args[1])

    def test_opening_takeover_also_precedes_optional_decision_network(self):
        engine = Engine(use_decision_net=True)
        with patch.object(engine, '_decision_pick', side_effect=AssertionError('开局不能被网络改选')):
            result = engine.analyze_turn(empty_board(), BLACK)
        self.assertEqual(result['part4_result']['move'], (7, 7))
        self.assertFalse(engine.decision_net_used)

    def test_fourth_move_book_point_is_searched_with_original_weight(self):
        for player, count in ((BLACK, 6), (WHITE, 7)):
            with self.subTest(player=player):
                engine, board, _moves = setup_position(count)
                desired = (0, 0)
                engine.opening_book = OpeningBook([record(board, player, desired)])
                with patch.object(engine.deep_search, 'rank_candidates', wraps=engine.deep_search.rank_candidates) as rank:
                    before = copy.deepcopy((board, engine.search.sb, engine.search.sw,
                                            engine.search._direction_levels))
                    result = engine.analyze_turn(board, player)
                self.assertTrue(rank.called, '本方第四手不能再直取谱着')
                self.assertIn(desired, rank.call_args.args[1])
                score = engine.search._candidate_score(board, *desired, player, 'attack')
                self.assertEqual(rank.call_args.args[1][desired], min(max(score, 1.0), 100.0))
                self.assertEqual(result['part4_result']['type'], 'searched')
                self.assertEqual(engine.opening_source, 'candidate')
                self.assertEqual((board, engine.search.sb, engine.search.sw,
                                  engine.search._direction_levels), before)

    def test_missing_book_first_three_uses_rules_without_deep_search(self):
        engine = Engine()
        engine.opening_book = OpeningBook(())
        board = empty_board()
        board[0][0] = BLACK
        with patch.object(engine.deep_search, 'rank_candidates', side_effect=AssertionError('谱外也不能深推')):
            result = engine.analyze_turn(board, WHITE)
        r, c = result['part4_result']['move']
        self.assertEqual(engine.opening_source, 'rules')
        self.assertEqual(board[r][c], 0)
        self.assertLessEqual(max(r, c), 1, '远离中心的首手应就近应对，不能固定走天元邻点')

    def test_book_cannot_override_immediate_tactics(self):
        for player in (BLACK, WHITE):
            for owner in (player, 3 - player):
                engine = Engine()
                board = empty_board()
                for c in range(4):
                    board[7][c] = owner
                engine.search.rebuild_all(board)
                engine.opening_book = OpeningBook([record(board, player, (0, 0))])
                result = engine.analyze_turn(board, player)
                self.assertEqual(result['part4_result']['move'], (7, 4))
                self.assertIsNone(engine.opening_source)

    def test_book_point_cannot_replace_tactical_defense_candidates(self):
        engine = Engine()
        board = empty_board()
        board[7][7] = BLACK
        board[7][8] = WHITE
        board[5][5] = BLACK
        board[8][8] = WHITE
        board[9][10] = BLACK
        board[4][8] = WHITE
        engine.search.rebuild_all(board)
        engine.opening_book = OpeningBook([record(board, BLACK, (0, 0))])
        tactical = {(6, 6): 100.0, (6, 7): 80.0}
        with patch.object(engine.search, 'tactical_candidates', return_value=('opp-vcf', tactical)), \
             patch.object(engine.deep_search, 'rank_candidates', return_value=[]) as rank:
            engine.analyze_turn(board, BLACK)
        self.assertEqual(rank.call_args.args[1], tactical)
        self.assertIsNone(engine.opening_source)

    def test_session_analysis_move_and_undo_share_opening_policy(self):
        session = PlaySession('opening-test')
        session.reset(mode='pvp')
        session.place(7, 7, BLACK)
        before = copy.deepcopy(session.game.moves)
        hint = session.analyze(WHITE)[0]
        self.assertEqual(session.game.moves, before)
        played = session.ai_play(WHITE)[0]
        self.assertEqual(hint, played)
        session.undo('last')
        self.assertIsNone(session.engine.opening_source)
        self.assertEqual(session.analyze(WHITE)[0], hint)

    def test_invalid_book_entry_is_not_silently_used(self):
        with self.assertRaises(ValueError):
            OpeningBook([(BLACK, (0,), (), 0, 0)])
        with self.assertRaises(ValueError):
            OpeningBook([(BLACK, (), (), 0, 0), (BLACK, (), (), 1, 1)])

    def test_full_board_fallback_has_no_move(self):
        self.assertIsNone(opening_fallback([[BLACK] * 15 for _ in range(15)], WHITE))


if __name__ == '__main__':
    unittest.main()
