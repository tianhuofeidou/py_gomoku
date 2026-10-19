"""协议盘面、对战时限与有限搜索复盘的回归。"""
import io
import itertools
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from gomoku.core.utils import BLACK, WHITE
from gomoku.protocol import Brain
from gomoku.tools import match, review


class BoardOrderTest(unittest.TestCase):
    def test_board_all_orders_both_colors(self):
        positions = (
            (BLACK, [(7, 7, 1), (7, 8, 2)]),
            (WHITE, [(7, 7, 2), (7, 8, 1), (6, 6, 2)]),
        )
        for color, stones in positions:
            for order in itertools.permutations(stones):
                with self.subTest(color=color, order=order):
                    brain = Brain()
                    data = ''.join('%d,%d,%d\n' % s for s in order) + 'DONE\n'
                    with patch('gomoku.protocol.service.ai_move', return_value=((0, 0), [], None)):
                        reply = brain.read_board(io.StringIO(data))
                    self.assertEqual(reply, '0,0')
                    self.assertEqual(brain.my_color, color)
                    for x, y, field in stones:
                        self.assertEqual(brain.board[y][x], color if field == 1 else 3 - color)
                    self.assertEqual([m['player'] for m in brain.moves],
                                     [BLACK if i % 2 == 0 else WHITE
                                      for i in range(len(stones) + 1)])
                    self.assertEqual(brain.engine.moves,
                                     [(m['r'], m['c'], m['player']) for m in brain.moves])

    def test_invalid_board_is_rejected_before_mutating(self):
        for data in ('1,1,1\nDONE\n', '1,1,3\nDONE\n',
                     '15,0,2\nDONE\n', '1,1,1\n1,1,2\nDONE\n',
                     'bad\nDONE\n'):
            with self.subTest(data=data):
                brain = Brain()
                with patch('gomoku.protocol.service.ai_move') as pick:
                    self.assertTrue(brain.read_board(io.StringIO(data)).startswith('ERROR'))
                pick.assert_not_called()
                self.assertEqual(brain.moves, [])


class FakeEngine:
    ms = 0

    def __init__(self, name, *args, **kwargs):
        self.name = name
        self.last_reply = '7,7'
        self.last_ms = 0
        self.closed = False

    def start_game(self, *args, **kwargs):
        return True

    def send_time_left(self, ms):
        pass

    def ask_move(self, *args, **kwargs):
        return (7, 7, self.ms) if self.name == 'black' else (6, 6, self.ms)

    def close(self):
        self.closed = True


class MatchClockTest(unittest.TestCase):
    def test_initialization_time_is_counted_only_on_own_clock(self):
        class ClockEngine(FakeEngine):
            ms = 20

            def start_game(self, *args, **kwargs):
                self.last_ms = 50 if self.name == 'black' else 0
                return True

        with patch.object(match, 'Engine', ClockEngine):
            winner, moves, log = match.play_one('', '', [], 1, 100, timeout_match=60)
        self.assertEqual(winner, 'W')
        self.assertEqual(moves, [])
        with patch.object(match, 'Engine', ClockEngine):
            winner, moves, log = match.play_one('', '', [(7, 7)], 2, 100, timeout_match=60)
        self.assertEqual(winner, 'draw', '黑方初始化不得占用白方的钟')
        self.assertEqual(log, [])

    def test_move_timeout_is_a_loss_and_closes_both_engines(self):
        created = []

        class HangingEngine(FakeEngine):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                created.append(self)

            def ask_move(self, *args, **kwargs):
                raise TimeoutError('test')

        with patch.object(match, 'Engine', HangingEngine):
            winner, moves, log = match.play_one('', '', [], 1, 100)
        self.assertEqual(winner, 'W')
        self.assertIn('超时判负', log[-1])
        self.assertEqual(moves, [])
        self.assertEqual(len(created), 2)
        self.assertTrue(all(e.closed for e in created))

    def test_either_clock_can_forfeit_both_colors(self):
        for turn, total in ((50, 1000), (100, 60), (50, 0)):
            for black_ms, white_ms, winner in ((70, 0, 'W'), (0, 70, 'B')):
                with self.subTest(turn=turn, total=total, winner=winner):
                    class ClockEngine(FakeEngine):
                        def ask_move(self, *args, **kwargs):
                            self.ms = black_ms if self.name == 'black' else white_ms
                            return super().ask_move(*args, **kwargs)
                    with patch.object(match, 'Engine', ClockEngine):
                        result, moves, log = match.play_one('', '', [], 2, turn,
                                                           timeout_match=total)
                    self.assertEqual(result, winner)
                    self.assertIn('超时判负', log[-1])

    def test_cumulative_clock_and_exact_boundary(self):
        class ClockEngine(FakeEngine):
            ms = 40

            def ask_move(self, *args, **kwargs):
                row = getattr(self, 'row', 0)
                self.row = row + 1
                return row, 0 if self.name == 'black' else 1, self.ms
        with patch.object(match, 'Engine', ClockEngine):
            winner, moves, log = match.play_one('', '', [], 3, 40, timeout_match=60)
        self.assertEqual(winner, 'W')
        self.assertEqual(len(moves), 2)
        self.assertIn('超时判负', log[-1])


class PipeDeadlineTest(unittest.TestCase):
    def test_silent_move_forfeits_and_reaps_processes_both_colors(self):
        with tempfile.TemporaryDirectory(prefix='gomoku-timeout-') as folder:
            path = Path(folder) / 'brain.py'
            path.write_text(
                "import sys, time\n"
                "for line in sys.stdin:\n"
                " if line.startswith('START'): print('OK', flush=True)\n"
                " elif line.strip() == 'END': break\n"
                " elif line.startswith(('BEGIN', 'TURN')): time.sleep(60)\n",
                encoding='utf-8')
            command = '%s -u %s' % (sys.executable, path)
            original = match.Engine
            for opening, expected in (([], 'W'), ([(7, 7)], 'B')):
                with self.subTest(winner=expected):
                    created = []

                    def capture(*args, **kwargs):
                        engine = original(*args, **kwargs)
                        created.append(engine)
                        return engine

                    begin = time.monotonic()
                    with patch.object(match, 'Engine', side_effect=capture):
                        winner, moves, log = match.play_one(
                            command, command, opening, 2, 1000)
                    self.assertEqual(winner, expected)
                    self.assertEqual(len(moves), len(opening))
                    self.assertIn('超时判负', log[-1])
                    self.assertLess(time.monotonic() - begin, 5)
                    self.assertTrue(all(e.proc.poll() is not None for e in created))
                    self.assertTrue(all(not e._reader.is_alive() for e in created))

    def make_engine(self, body, timeout=1000):
        folder = tempfile.TemporaryDirectory(prefix='gomoku-pipe-')
        self.addCleanup(folder.cleanup)
        path = Path(folder.name) / 'brain.py'
        path.write_text('import sys, time\n' + body, encoding='utf-8')
        engine = match.Engine('test', '%s -u %s' % (sys.executable, path), timeout_turn=timeout)
        self.addCleanup(engine.close)
        return engine

    def test_no_start_reply_has_deadline(self):
        engine = self.make_engine('time.sleep(60)\n', timeout=400)
        begin = time.monotonic()
        self.assertFalse(engine.start_game(15, 400))
        engine.close()
        self.assertLess(time.monotonic() - begin, 3)
        self.assertIsNotNone(engine.proc.poll())

    def test_silent_or_debug_spam_move_has_deadline(self):
        for spam in (False, True):
            with self.subTest(spam=spam):
                body = ("for line in sys.stdin:\n"
                        " if line.startswith('START'): print('OK', flush=True)\n"
                        " elif line.startswith('BEGIN'):\n")
                body += ("  while True:\n   print('DEBUG thinking', flush=True)\n   time.sleep(.01)\n"
                         if spam else "  time.sleep(60)\n")
                engine = self.make_engine(body)
                self.assertTrue(engine.start_game(15, 1000))
                engine.send_time_left(80)
                begin = time.monotonic()
                try:
                    engine.ask_move('BEGIN')
                except TimeoutError:
                    pass
                else:
                    self.fail('无回复必须触发时限')
                engine.close()
                self.assertLess(time.monotonic() - begin, 3)
                self.assertIsNotNone(engine.proc.poll())

    def test_eof_returns_and_empty_board_sends_done(self):
        engine = self.make_engine(
            "for line in sys.stdin:\n"
            " if line.startswith('START'): print('OK', flush=True)\n"
            " elif line.strip() == 'DONE': print('7,7', flush=True)\n"
            " elif line.strip() == 'END': break\n")
        self.assertTrue(engine.start_game(15, 1000))
        self.assertEqual(engine.ask_move('BOARD', [])[0:2], (7, 7))
        exited = self.make_engine("print('OK', flush=True)\n")
        self.assertEqual(exited.read_line(), 'OK')
        self.assertIsNone(exited.read_line())


class ReviewEvidenceTest(unittest.TestCase):
    def run_review(self, ranked):
        with patch.object(review, 'Engine') as factory:
            factory.return_value.analyze_turn.return_value = {'part4_result': {'ranked': ranked}}
            return review.review_game({'moves': [{'r': 0, 'c': 0, 'player': BLACK}]}, start_ply=0)

    @staticmethod
    def candidate(r, c, state, name):
        return dict(r=r, c=c, state=state, state_name=name)

    def test_unsearched_move_is_unknown(self):
        rows, first, missed = self.run_review([self.candidate(1, 1, 0, '无')])
        self.assertEqual(rows[0]['played_state'], '未评估（不在候选）')
        self.assertIsNone(first)
        self.assertIsNone(missed)

    def test_soft_loss_is_a_non_hard_loss_alternative(self):
        rows, first, missed = self.run_review([
            self.candidate(1, 1, -2, '存在必败'), self.candidate(0, 0, -1, '必败')])
        self.assertEqual(rows[0]['save'], 'B2')
        self.assertIsNone(first)
        self.assertEqual(missed, (1, 'B2', 'A1'))

    def test_only_evaluated_hard_loss_marks_missed_alternative(self):
        for state, name in ((-2, '存在必败'), (0, '无'), (1, '必胜'), (2, '存在必胜')):
            with self.subTest(state=state):
                _, _, missed = self.run_review([
                    self.candidate(1, 1, 0, '无'), self.candidate(0, 0, state, name)])
                self.assertIsNone(missed)
        rows, first, missed = self.run_review([self.candidate(0, 0, -1, '必败')])
        self.assertEqual(first, 1)
        self.assertIsNone(rows[0]['save'])
        self.assertIsNone(missed)


if __name__ == '__main__':
    unittest.main()
