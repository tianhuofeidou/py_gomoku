"""败局复测的跨轮隔离、不同起点覆盖与最后一手终局回归。"""
import contextlib
import io
import unittest
from unittest.mock import patch

from gomoku import loss_book as lb
from gomoku.tools import loss_book as cli
from gomoku.tests.test_loss_book import LossBookBase, mock_engine_factory


class LossBookBatchTest(LossBookBase):
    def case(self, manual=3):
        case, _ = lb.add_case(self.MOVES, 1, winner=2)
        lb.set_suspects(case, manual_ply=manual)
        return lb.load_store()['cases'][0]

    def run_cli_round(self, case, outcomes, *, stop=False):
        results = iter(outcomes)
        calls = []

        def retest(c, **kwargs):
            outcome = next(results)
            if outcome == 'interrupt':
                raise KeyboardInterrupt('模拟用户中断本轮')
            calls.append(kwargs['start_ply'])
            result = {'startPly': kwargs['start_ply'], 'result': outcome,
                      'fingerprint': lb.current_fingerprint()}
            if kwargs.get('batch_id'):
                result['batchId'] = kwargs['batch_id']
            return lb._record_retest(c, result, None, batch_total=kwargs['batch_total'])

        argv = ['retest', '--id', case['id']]
        if stop:
            argv.append('--stop-after-first')
        with patch.object(lb, 'retest_case', side_effect=retest), contextlib.redirect_stdout(io.StringIO()):
            try:
                cli.main(argv)
            except KeyboardInterrupt:
                pass
        return calls

    def test_cancelled_cli_round_cannot_fill_missing_start_with_previous_round(self):
        case = self.case()
        self.assertEqual(self.run_cli_round(case, ['win', 'win', 'interrupt']), [5, 3])
        self.assertEqual(lb.load_store()['cases'][0]['status'], 'pending')
        self.assertEqual(self.run_cli_round(case, ['win', 'interrupt']), [5])
        current = lb.load_store()['cases'][0]
        self.assertEqual([r['startPly'] for r in current['retests']], [5, 3, 5])
        self.assertEqual(current['status'], 'pending', '旧轮两条记录不能凑齐新轮三个起点')
        self.assertEqual(lb.archive_passed(), [])

    def test_same_version_new_complete_round_can_pass_after_failed_round(self):
        case = self.case()
        self.run_cli_round(case, ['loss', 'win', 'win'])
        self.assertEqual(lb.load_store()['cases'][0]['status'], 'pending')
        self.run_cli_round(case, ['win', 'win', 'win'])
        current = lb.load_store()['cases'][0]
        self.assertEqual(current['status'], 'passed')
        self.assertEqual(len(current['retests']), 6, '保留旧轮失败历史')
        self.assertEqual(len({r['batchId'] for r in current['retests']}), 2)

    def test_repeated_start_in_one_round_does_not_count_as_three_starts(self):
        case = self.case()
        batch = lb.begin_retest_batch(case, lb.retest_starts(case, manual_ply=3))
        for _ in range(3):
            lb._record_retest(case, {'startPly': 5, 'result': 'win',
                                    'batchId': batch['id'], 'fingerprint': batch['fingerprint']}, None)
        self.assertEqual(lb.load_store()['cases'][0]['status'], 'pending')
        self.assertEqual(lb.archive_passed(), [])

    def test_late_old_round_result_does_not_override_current_complete_round(self):
        case = self.case()
        starts = lb.retest_starts(case, manual_ply=3)
        old = lb.begin_retest_batch(case, starts)
        new = lb.begin_retest_batch(case, starts)
        for start in starts:
            lb._record_retest(case, {'startPly': start['startPly'], 'result': 'win',
                                    'batchId': new['id'], 'fingerprint': new['fingerprint']}, None)
        lb._record_retest(case, {'startPly': 5, 'result': 'loss', 'batchId': old['id'],
                                'fingerprint': old['fingerprint']}, None)
        self.assertEqual(lb.load_store()['cases'][0]['status'], 'passed')

    def test_old_version_failure_does_not_block_new_complete_round(self):
        case = self.case()
        starts = lb.retest_starts(case, manual_ply=3)
        old_fp = {'engine': {'digest': 'old'}, 'book': 'old', 'rule': lb.RULE_ID}
        old = lb.begin_retest_batch(case, starts, fingerprint=old_fp)
        lb._record_retest(case, {'startPly': 5, 'result': 'loss', 'batchId': old['id'],
                                'fingerprint': old_fp}, None)
        new = lb.begin_retest_batch(case, starts)
        for start in starts:
            lb._record_retest(case, {'startPly': start['startPly'], 'result': 'draw',
                                    'batchId': new['id'], 'fingerprint': new['fingerprint']}, None)
        current = lb.load_store()['cases'][0]
        self.assertEqual(current['status'], 'passed')
        self.assertEqual(len(current['retests']), 4)

    def test_stop_after_first_does_not_pass_incomplete_round(self):
        case = self.case()
        self.assertEqual(self.run_cli_round(case, ['win'], stop=True), [5])
        self.assertEqual(lb.load_store()['cases'][0]['status'], 'pending')

    def test_stop_after_first_can_pass_if_only_one_start_exists(self):
        case = self.case(manual=1)
        self.assertEqual(self.run_cli_round(case, ['win'], stop=True), [1])
        self.assertEqual(lb.load_store()['cases'][0]['status'], 'passed')


class LossBookBudgetTerminalTest(LossBookBase):
    @staticmethod
    def original_loss(ai_color):
        moves = []
        for c in range(4):
            moves.extend([{'r': 7, 'c': c, 'player': 1}, {'r': 0, 'c': c, 'player': 2}])
        # 原败局：本方错过成五，对手下一手成五。
        moves.extend([{'r': 5, 'c': 5, 'player': 1}, {'r': 0, 'c': 4, 'player': 2}])
        if ai_color == 2:
            moves = [{'r': 14, 'c': 14, 'player': 1}] + [dict(m, player=3 - m['player']) for m in moves]
        return moves

    def test_win_on_final_allowed_ply_is_terminal_for_both_colors(self):
        for color in (1, 2):
            with self.subTest(ai_color=color):
                case, _ = lb.add_case(self.original_loss(color), color, winner=3 - color)
                result = lb.retest_case(case, manual_ply=5, max_plies=1,
                                        ai_move_fn=lambda *a, **k: ((7, 4), [], False),
                                        engine_factory=mock_engine_factory)
                self.assertEqual(result['result'], lb.RETEST_WIN)
                self.assertEqual(result['winner'], color)
                self.assertEqual(result['plies'], 1)

    def test_loss_on_final_allowed_ply_is_terminal_for_both_colors(self):
        for color in (1, 2):
            with self.subTest(ai_color=color):
                case, _ = lb.add_case(self.original_loss(color), color, winner=3 - color)
                result = lb.retest_case(case, manual_ply=5, max_plies=2,
                                        ai_move_fn=lambda *a, **k: ((5, 5), [], False),
                                        engine_factory=mock_engine_factory)
                self.assertEqual(result['result'], lb.RETEST_LOSS)
                self.assertEqual(result['winner'], 3 - color)
                self.assertEqual(result['plies'], 2)


if __name__ == '__main__':
    unittest.main()
