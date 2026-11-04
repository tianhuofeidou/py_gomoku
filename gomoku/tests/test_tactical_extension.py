"""显式开启8手实验截断：独立威胁差计分、未知状态、路径/缓存/回滚。"""
import copy
import unittest
from contextlib import ExitStack
from unittest.mock import patch

from gomoku.core.engine import Engine
from gomoku.core.deep_search import DeepSearch
from gomoku.core.utils import BLACK, WHITE, empty_board


def snapshot(engine, board):
    s = engine.search
    return copy.deepcopy((board, engine.moves, s.sb, s.sw, s._direction_levels,
                          s.zero_history, s.zero_active, s.gear))


class Observer:
    def __init__(self):
        self.events = []

    def record(self, kind, node=None, **values):
        self.events.append((kind, values))


class TacticalExtensionTest(unittest.TestCase):
    def setUp(self):
        # 正式默认关闭；本模块专门检验显式开启的旧实验机制。
        for name in ('TACTICAL_FREE_PLIES','TACTICAL_EXTENSION_PLIES'):
            setting=patch.object(DeepSearch,name,8)
            setting.start()
            self.addCleanup(setting.stop)

    def test_score_range_perspective_and_more_than_one_threat_per_move(self):
        d = Engine().deep_search
        for black, white, expected in ((2, 1, 200), (4, 0, 800), (8, 0, 800),
                                       (0, 7, -800), (3, 3, 0)):
            context = [{'extension_state': (8, black, white)}]
            self.assertEqual(d._extension_score(context, None, BLACK), expected)
            self.assertEqual(d._extension_score(context, None, WHITE), -expected)

    def test_only_exhausted_tactical_steps_consume_extension(self):
        d = Engine().deep_search
        board = empty_board()
        for temperature, tactical, consumed in ((1, True, False), (0, False, False),
                                                 (0, True, True), (-20, True, True)):
            context = {}
            with patch.object(d, '_active_threat_points', side_effect=[frozenset(), frozenset({(1, 1), (2, 2)})]):
                d._prepare_extension_step(context, board, BLACK, temperature, tactical, (0, 0, 0))
                d._finish_extension_step(context, board, BLACK)
            self.assertEqual(context['extension_state'], (1, 2, 0) if consumed else (0, 0, 0))

    def test_old_threat_does_not_repeat_and_branch_ledgers_are_independent(self):
        d = Engine().deep_search
        previous = (3, 2, 1)
        context = {}
        with patch.object(d, '_active_threat_points', side_effect=[frozenset({(1, 1)}),
                                                                  frozenset({(1, 1), (2, 2), (3, 3)})]):
            d._prepare_extension_step(context, empty_board(), WHITE, -1, True, previous)
            d._finish_extension_step(context, empty_board(), WHITE)
        self.assertEqual(context['extension_state'], (4, 2, 3))
        self.assertEqual(previous, (3, 2, 1))
        defender = {}
        with patch.object(d, '_active_threat_points', return_value=frozenset({(1, 1)})):
            d._prepare_extension_step(defender, empty_board(), BLACK, -1, True, previous)
            d._finish_extension_step(defender, empty_board(), BLACK)
        self.assertEqual(defender['extension_state'], (4, 2, 1))

    def play_step(self, engine, board, point, player, previous):
        context = {}
        d = engine.deep_search
        d._prepare_extension_step(context, board, player, 0, True, previous)
        journal = {}
        board[point[0]][point[1]] = player
        engine.search.on_move(board, *point, player, journal=journal, record_history=False)
        d._finish_extension_step(context, board, player)
        return context, journal

    def test_real_live_three_creates_two_distinct_threat_points(self):
        for player in (BLACK, WHITE):
            engine = Engine()
            board = empty_board()
            board[7][5] = board[7][6] = player
            engine.search.rebuild_all(board)
            before = snapshot(engine, board)
            context, journal = self.play_step(engine, board, (7, 7), player, (0, 0, 0))
            self.assertEqual(context['extension_state'], (1, 2, 0) if player == BLACK else (1, 0, 2))
            engine.search.restore(journal)
            board[7][7] = 0
            self.assertEqual(snapshot(engine, board), before)

    def test_real_pure_block_counts_zero_but_counter_four_counts_new_threat(self):
        for counter in (False, True):
            engine = Engine()
            board = empty_board()
            for c in (5, 6, 7, 8):
                board[7][c] = WHITE
            if counter:
                for r in (4, 5, 6):
                    board[r][9] = BLACK
            engine.search.rebuild_all(board)
            before = snapshot(engine, board)
            context, journal = self.play_step(engine, board, (7, 9), BLACK, (0, 0, 0))
            self.assertEqual(context['extension_state'], (1, 1 if counter else 0, 0))
            engine.search.restore(journal)
            board[7][9] = 0
            self.assertEqual(snapshot(engine, board), before)

    def chain(self, player, temperature=0, win_at=None, revisit=False):
        engine = Engine()
        d = engine.deep_search
        d.T0 = temperature
        board = empty_board()
        points = [(1+2*(i//5), 1+2*(i % 5)) for i in range(20)]
        observer = Observer()
        d._debug_observer = observer

        def count(b):
            return sum(value != 0 for row in b for value in row)

        def tactical(b, side):
            index = count(b)
            return ('my-vcf', {points[index]: 100.}) if index < len(points) else (None, {})

        def candidates(b, side, n=5, plies=0):
            return [(r, c, weight) for (r, c), weight in tactical(b, side)[1].items()]

        before = snapshot(engine, board)
        with ExitStack() as stack:
            stack.enter_context(patch.object(d, '_tactical_result', side_effect=tactical))
            stack.enter_context(patch.object(d, '_candidate_points', side_effect=candidates))
            stack.enter_context(patch.object(d, '_active_threat_points', return_value=frozenset()))
            stack.enter_context(patch.object(d, '_check_win', side_effect=lambda b, r, c, p: count(b) == win_at))
            rows = d.rank_candidates(board, {points[0]: 100.}, player, complete_scores=True)
            self.assertEqual(snapshot(engine, board), before)
            steps = [values for kind, values in observer.events if kind == 'temperature']
            if revisit:
                for r, c, side in ((*points[0], player), (*points[1], 3-player)):
                    board[r][c] = side
                    engine.on_move(board, r, c, side)
                reroot_before = snapshot(engine, board)
                first = d.cached_turn(board, player)
                second = d.cached_turn(board, player)
                self.assertIsNotNone(first)
                self.assertEqual(first[2], second[2])
                self.assertEqual(snapshot(engine, board), reroot_before)
                self.assertEqual(len([1 for kind, _ in observer.events if kind == 'temperature']), len(steps))
                pending = [d._tree_root]
                while pending:
                    node = pending.pop()
                    self.assertLessEqual(node['extension_state'][0], 8)
                    pending.extend(node['children'].values())
            return d, rows, steps, observer

    def test_actual_recursive_chain_stops_at_eight_both_colors_and_does_not_seed_state(self):
        for player in (BLACK, WHITE):
            d, rows, steps, observer = self.chain(player)
            self.assertEqual(len(steps), 8)
            self.assertEqual([step['extension_state'][0] for step in steps], list(range(1, 9)))
            self.assertEqual(rows[0]['score'], 0)
            self.assertEqual(rows[0]['state'], 0)
            cuts = [e for kind, e in observer.events if kind == 'cutoff']
            self.assertEqual(cuts[-1]['leaf_end_reason'], 'tactical_extension')
            self.assertEqual(d.cache_stats['tactical_probe_moves'], 0)
            self.assertTrue(d.root_contribution_report['complete'])

    def test_positive_temperature_does_not_spend_eight_step_quota(self):
        _, _, steps, _ = self.chain(BLACK, temperature=160)
        self.assertEqual(len(steps), 20)
        self.assertTrue(all(step['extension_state'] == (0, 0, 0) for step in steps))

    def test_actual_five_on_eighth_move_takes_priority(self):
        for player in (BLACK, WHITE):
            _, rows, steps, observer = self.chain(player, win_at=8)
            self.assertEqual(len(steps), 8)
            self.assertEqual(rows[0]['state'], -1)
            self.assertEqual(rows[0]['score'], -1000)
            self.assertFalse(any(kind == 'cutoff' and values['leaf_end_reason'] == 'tactical_extension'
                                 for kind, values in observer.events))

    def test_real_reroot_and_repeated_analysis_do_not_refill_quota(self):
        self.chain(BLACK, revisit=True)

    def test_no_common_defense_precedes_budget_unknown(self):
        d = Engine().deep_search
        with patch.object(d, '_tactical_result', return_value=('opp-d33', {})):
            self.assertFalse(d._temperature_exhausted(empty_board(), BLACK, -1, 0, extension_count=8))
        with patch.object(d, '_tactical_result', return_value=('my-vcf', {(1, 1): 100.})):
            self.assertTrue(d._temperature_exhausted(empty_board(), BLACK, -1, 0, extension_count=8))

    def test_keys_and_leaf_value_are_isolated_by_count_and_both_threat_counts(self):
        d = Engine().deep_search
        b = empty_board()
        d._begin_search_cache(b, False)
        first, second = (8, 4, 0), (8, 0, 4)
        self.assertNotEqual(d._result_key(1, 1, 0, 1, extension_state=first),
                            d._result_key(1, 1, 0, 1, extension_state=second))
        self.assertNotEqual(d._cold_result_key(b, 1, 2, 1, 0, 1, 8, extension_state=first),
                            d._cold_result_key(b, 1, 2, 1, 0, 1, 8, extension_state=second))
        self.assertEqual(d._leaf_score(b, [], 'tactical_extension', BLACK, [{'extension_state': first}]), 800)
        self.assertEqual(d._leaf_score(b, [], 'tactical_extension', BLACK, [{'extension_state': second}]), -800)
        self.assertEqual(d._leaf_score(b, [], 'tactical_extension', BLACK, [{'extension_state': first}]), 800)
        self.assertEqual(d.cache_stats['leaf_hits'], 1)

    def test_cap_frontier_never_receives_cache_temperature_and_setting_invalidates_tree(self):
        d = Engine().deep_search
        d._cache_credit = 160
        frontier = dict(state=0, terminal=False, extension_state=(8, 2, 1),
                        path_weight=1, depth=1, temperature=-1, score=200)
        self.assertEqual(d._continuation_temperature(frontier), (0, True))
        b = empty_board()
        d._begin_search_cache(b, False)
        root = d._tree_root
        d.TACTICAL_EXTENSION_PLIES = 16
        d._begin_search_cache(b, False)
        self.assertIsNot(d._tree_root, root)


if __name__ == '__main__':
    unittest.main()
