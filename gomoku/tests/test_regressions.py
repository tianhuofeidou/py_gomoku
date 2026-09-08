"""Runtime regressions; all persistence is redirected to temporary storage."""
import copy
import json
import os
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gomoku import game, memory, service
from gomoku.adapter import server, ai_move
from gomoku.core.engine import Engine
from gomoku.core.search import DEFAULT_GEAR, GEAR_ZERO_WINDOW
from gomoku.core.deep_search import DeepSearch
from gomoku.core.utils import BLACK, WHITE, empty_board
from gomoku.session import PlaySession


class RegressionTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = tmp.name
        for context in (
                patch.dict(os.environ, {'GOMOKU_HOME': self.home, 'DSH_GOMOKU_GA': '0',
                                        'DSH_GOMOKU_USE_NET': '0'}),
                patch.object(game, 'games', {}),
                patch.object(memory, '_global_memory', None),
                patch.object(server, 'sessions', {}),
                patch.object(service, '_engine', None)):
            context.start()
            self.addCleanup(context.stop)

    def board_engine(self, stones):
        board = empty_board()
        e = Engine()
        e.deep_search.use_nn = False
        for r, c, p in stones:
            board[r][c] = p
            e.on_move(board, r, c, p)
        return board, e

    def test_recursive_forced_block_for_both_colors(self):
        for attacker in (BLACK, WHITE):
            defender = 3 - attacker
            board, e = self.board_engine(
                [(7, c, attacker) for c in (4, 5, 6, 7)] +
                [(7, 3, defender), (2, 10, defender), (12, 12, defender)] +
                [(1, c, defender) for c in (2, 3, 4)])
            self.assertEqual(e.deep_search._candidate_points(board, defender), [(7, 8, 100.0)])
            self.assertEqual(e.analyze_turn(board, defender)['part4_result']['move'], (7, 8))

    def test_own_win_precedes_block(self):
        board, e = self.board_engine([(7, c, BLACK) for c in (4, 5, 6, 7)] +
                                     [(1, c, WHITE) for c in (4, 5, 6, 7)])
        for p, row in ((BLACK, 7), (WHITE, 1)):
            self.assertTrue(all(r == row for r, c, w in e.deep_search._candidate_points(board, p)))

    def test_recursive_defense_preserves_all_weighted_points(self):
        board, e = self.board_engine([(7, c, BLACK) for c in (5, 6, 7)])
        reason, points = e.search.tactical_candidates(board, WHITE)
        self.assertEqual(reason, 'opp-vcf')
        self.assertEqual({(r, c): w for r, c, w in e.deep_search._candidate_points(board, WHITE, n=1)}, points)

    def test_incremental_matches_full_rebuild_and_journal(self):
        rng = random.Random(7)
        board, e = self.board_engine([])
        for i, (r, c) in enumerate([(2, 4)] + rng.sample([(r, c) for r in range(15)
                                                        for c in range(15) if (r, c) != (2, 4)], 29)):
            p = 1 + i % 2
            board[r][c] = p
            before = copy.deepcopy((e.search.sb, e.search.sw))
            journal = {}
            e.search.on_move(board, r, c, p, journal=journal, record_history=False)
            reference = Engine()
            reference.search.rebuild_all(board)
            self.assertEqual((e.search.sb, e.search.sw), (reference.search.sb, reference.search.sw))
            e.search.restore(journal)
            self.assertEqual((e.search.sb, e.search.sw), before)
            e.on_move(board, r, c, p)

    def test_memory_bonus_respects_all_symmetries(self):
        gm = memory.global_memory()
        gm['goodLines'] = [{'ai': [{'r': 8, 'c': 9}]}]
        bonus = memory.move_memory_bonus()
        for transform in memory.SYMMS:
            self.assertEqual(bonus[transform(8, 9)], memory.MEM_W_WIN)

    def test_server_replaces_equal_length_or_changed_prefix(self):
        histories = [[(7, 7, 1)], [(0, 0, 1)],
                     [(7, 7, 1), (6, 6, 2), (8, 8, 1)], [(7, 7, 1)]]
        with patch.object(server, 'compute_move', return_value=(1, 1, [], False)):
            for history in histories:
                moves = [dict(r=r, c=c, player=p) for r, c, p in history]
                out = server.handle({'id': 9, 'type': 'move', 'session': 's', 'moves': moves})
                self.assertTrue(out['ok'])
                expected = empty_board()
                for r, c, p in history:
                    expected[r][c] = p
                self.assertEqual(server.sessions['s'].board, expected)
                self.assertEqual(server.sessions['s'].engine.moves, history)

    def test_server_appends_without_replacing_engine(self):
        with patch.object(server, 'compute_move', return_value=(1, 1, [], False)):
            first = [{'r': 7, 'c': 7, 'player': 1}]
            server.handle({'type': 'move', 'session': 's', 'moves': first})
            engine = server.sessions['s'].engine
            server.handle({'type': 'move', 'session': 's', 'moves': first + [{'r': 6, 'c': 6, 'player': 2}]})
            self.assertIs(server.sessions['s'].engine, engine)

    def test_invalid_request_preserves_session_and_id(self):
        with patch.object(server, 'compute_move', return_value=(1, 1, [], False)):
            first = [{'r': 7, 'c': 7, 'player': 1}]
            server.handle({'type': 'move', 'session': 's', 'moves': first})
            for bad in ({'r': -1, 'c': 0, 'player': 2}, {'r': 7, 'c': 7, 'player': 2}):
                result = server.handle({'id': 42, 'type': 'move', 'session': 's', 'moves': first + [bad]})
                self.assertFalse(result['ok'])
                self.assertEqual(result['id'], 42)
                self.assertEqual(server.sessions['s'].count, 1)

    def test_sessions_are_independent_and_reset_all_state(self):
        one, two = PlaySession('one'), PlaySession('two')
        self.assertIsNot(one.engine, two.engine)
        one.place(7, 7, BLACK)
        one.place(6, 6, WHITE)
        one.engine.search.gear[BLACK] = 1
        one.engine.search.zero_active[BLACK] = True
        one.undo('last')
        self.assertEqual(one.engine.moves, [(7, 7, BLACK)])
        self.assertFalse(one.engine.search.zero_active[BLACK])
        self.assertEqual(one.engine.search.gear[BLACK], DEFAULT_GEAR)
        one.reset()
        self.assertEqual(one.engine.moves, [])
        self.assertEqual(two.engine.moves, [])

    def test_zero_history_expires_by_ply(self):
        board, e = self.board_engine([])
        e.search.sb[0][0] = 0
        e.search._record_zero_event(board, BLACK)
        e.search.sb[0][0] = 17
        for _ in range(GEAR_ZERO_WINDOW):
            e.search._record_zero_event(board, BLACK)
        self.assertEqual(e.search._zero_momentum(BLACK), 0)

    def test_compute_move_honors_black(self):
        board, e = self.board_engine([(7, c, BLACK) for c in range(4, 8)] + [(7, 3, WHITE)])
        self.assertEqual(e.compute_move(board, BLACK), (7, 8))
        with patch.object(e, 'analyze_turn', wraps=e.analyze_turn) as analyze:
            e.compute_move(board, BLACK)
            self.assertEqual(analyze.call_args.args[1], BLACK)

    def finish_game(self, sid):
        g = game.Game()
        g.register(sid)
        for r, c, p in [(7, 0, 1), (0, 0, 2), (7, 1, 1), (0, 2, 2),
                        (7, 2, 1), (0, 4, 2), (7, 3, 1), (0, 6, 2), (7, 4, 1)]:
            self.assertTrue(g.place(r, c, p)['ok'])
        return g

    def test_terminal_undo_reverses_only_its_own_record_after_reload(self):
        self.finish_game('other')
        g = self.finish_game('one')
        self.assertEqual(memory.global_memory()['totals']['losses'], 2)
        game.games.clear()
        game.load_all()
        g = game.games['one']
        g.undo_last_move()
        gm = memory.global_memory()
        self.assertEqual(gm['totals']['losses'], 1)
        self.assertEqual(len(gm['badLines']), 1)
        self.assertEqual(sum(gm['lossByType'].values()), 1)
        self.assertFalse(g.archived)
        self.assertEqual(g.sid, 'one')
        self.assertEqual(g.history, [])
        g.exec('draw', 'human')
        self.assertEqual(gm['totals'], {'wins': 0, 'losses': 1, 'draws': 1})
        self.assertEqual(len(g.history), 1)
        g.undo_last_move()
        self.assertEqual(gm['totals']['draws'], 0)

    def test_legacy_terminal_undo(self):
        g = self.finish_game('legacy')
        g.history[-1].pop('memoryReceipt')
        memory.global_memory()['badLines'][-1].pop('resultId')
        g.undo_last_move()
        self.assertEqual(memory.global_memory()['totals']['losses'], 0)
        self.assertEqual(memory.global_memory()['badLines'], [])
        g.place(7, 4, BLACK)
        self.assertEqual(memory.global_memory()['totals']['losses'], 1)

    def test_new_game_retains_history(self):
        s = PlaySession('local')
        s.game = self.finish_game('local')
        s.reset()
        self.assertEqual(len(s.game.history), 1)

    def test_ga_is_opt_in_and_path_is_shared(self):
        self.assertIs(server.load_ga_params, service.load_ga_params)
        self.assertIs(ai_move.load_ga_params, service.load_ga_params)
        with patch.object(service.G, 'apply_params') as apply:
            for loader in (service.load_ga_params, server.load_ga_params, ai_move.load_ga_params):
                loader()
            apply.assert_not_called()
            with patch.dict(os.environ, {'DSH_GOMOKU_GA': '1'}):
                service.load_ga_params()
            apply.assert_called_once()
            self.assertIn('deep_t0', apply.call_args.args[0])

    def test_opening_is_identical_in_both_adapters(self):
        moves = [{'r': 7, 'c': 7, 'player': 1}, {'r': 6, 'c': 6, 'player': 2}]
        result = server.handle({'type': 'move', 'session': 'opening', 'moves': moves, 'player': 1})
        root = Path(__file__).resolve().parents[2]
        out = subprocess.run([sys.executable, '-m', 'gomoku.adapter.ai_move'],
                             input=json.dumps({'moves': moves, 'player': 1}), text=True,
                             capture_output=True, cwd=root, timeout=15, check=True)
        once = json.loads(out.stdout)
        self.assertEqual(once, {k: result[k] for k in ('row', 'col', 'ranked', 'net_used')})

    def test_network_flag_and_model_path(self):
        e = Engine(use_decision_net=True)
        e.decision_net_used = True
        ranked = [{'r': 2, 'c': 3, 'score': 4}]
        self.assertEqual(e._decision_pick(empty_board(), WHITE, ranked), (2, 3))
        self.assertFalse(e.decision_net_used)
        self.assertTrue(Path(e.deep_search.model_path).is_absolute())


if __name__ == '__main__':
    unittest.main()
