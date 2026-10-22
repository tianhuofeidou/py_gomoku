"""真实 Python 子进程验证独立 CLI、败局库命令和 Gomocup 管道入口。

所有子进程只使用标准库及临时 GOMOKU_HOME，不 mock 引擎或命令处理函数。
"""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from gomoku import loss_book as lb

ROOT = Path(__file__).resolve().parents[2]


class CliEndToEndTest(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory(prefix='gomoku-cli-e2e-')
        self.addCleanup(folder.cleanup)
        self.home = Path(folder.name)
        self.env = dict(os.environ, GOMOKU_HOME=str(self.home), PYTHONIOENCODING='utf-8',
                        PYTHONUTF8='1', DSH_GOMOKU_GA='0')

    def run_cli(self, *args, input=None):
        result = subprocess.run([sys.executable, '-B', '-S', *args], cwd=ROOT,
                                env=self.env, input=input, text=True, encoding='utf-8',
                                capture_output=True, timeout=45)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn('Traceback', result.stderr)
        return result.stdout

    def read_store(self):
        return json.loads((self.home / 'loss-book.json').read_text(encoding='utf-8'))

    def test_interactive_moves_help_invalid_input_undo_numeric_input_and_quit(self):
        output = self.run_cli('play_cli.py', input='h\nbad\na16\n0,0\nh8\nh8\nu\n8,8\nq\n')
        self.assertIn('坐标格式', output)
        self.assertIn('从 1 开始', output)
        self.assertIn('无法识别', output)
        self.assertIn('越界', output)
        self.assertIn('该位置已有棋子', output)
        self.assertIn('已悔棋', output)
        self.assertIn('决策：棋谱开局', output)
        self.assertIn('叶子：未触发', output)
        self.assertIn('分 谱着', output)
        responses = re.findall(r'AI 落子 ([A-O]\d+)', output)
        self.assertEqual(len(responses), 2)
        self.assertEqual(responses[0], responses[1], '悔棋重走保持本局开局视角')
        games = json.loads((self.home / 'games.json').read_text(encoding='utf-8'))
        moves = games['cli']['moves']
        self.assertEqual(len(moves), 2)
        self.assertEqual(moves[0], {'r': 7, 'c': 7, 'player': 1}, '8,8 与 H8 一致')
        self.assertEqual(moves[1]['player'], 2)
        self.assertNotEqual((moves[1]['r'], moves[1]['c']), (7, 7))

    def test_loss_book_import_locate_retest_show_and_archive_with_real_engine(self):
        moves = []
        for c in range(4):
            moves.extend([{'r': 7, 'c': c, 'player': 1}, {'r': 0, 'c': c, 'player': 2}])
        moves.extend([{'r': 5, 'c': 5, 'player': 1}, {'r': 0, 'c': 4, 'player': 2}])
        payload = {'games': [
            {'game': 1, 'a_white': False, 'winner': 'W', 'win_by': 'five', 'moves': moves},
            {'game': 2, 'a_white': False, 'winner': 'B', 'win_by': 'five', 'moves': moves},
            {'game': 3, 'a_white': False, 'winner': 'draw', 'moves': moves},
            {'game': 4, 'a_white': False, 'winner': 'W', 'win_by': 'timeout', 'moves': moves},
        ]}
        source = self.home / 'games-import.json'
        source.write_text(json.dumps(payload), encoding='utf-8')
        command = ('-m', 'gomoku.tools.loss_book')
        output = self.run_cli(*command, 'import', '--json', str(source), '--ai-side', 'a')
        self.assertIn('跳过 2 局', output)
        cases = self.read_store()['cases']
        self.assertEqual({c['kind'] for c in cases}, {'loss', 'timeout'})
        target = next(c for c in cases if c['kind'] == 'loss')
        ident = target['id']
        self.assertIn(ident[:8], self.run_cli(*command, 'list', '--kind', 'loss'))
        self.assertIn('疑似关键败手', self.run_cli(*command, 'locate', '--id', ident))
        located = next(c for c in self.read_store()['cases'] if c['id'] == ident)
        self.assertTrue(any(s['ply'] == 9 and s['reason'] == 'missed-five' for s in located['suspects']))
        self.assertIn('已人工指定', self.run_cli(*command, 'locate', '--id', ident, '--ply', '5'))
        output = self.run_cli(*command, 'retest', '--id', ident, '--max-plies', '1')
        current = next(c for c in self.read_store()['cases'] if c['id'] == ident)
        self.assertEqual(current['retestBatch']['expectedStarts'], [9, 7, 5])
        self.assertEqual([r['startPly'] for r in current['retests']], [9, 7, 5])
        self.assertEqual(current['retests'][0]['result'], 'win', '真实引擎最后允许一手成五')
        self.assertEqual(current['status'], 'pending', '另外两个起点预算耗尽不能通过')
        self.assertEqual(len({r['batchId'] for r in current['retests']}), 1)
        self.assertIn('复测记录', self.run_cli(*command, 'show', '--id', ident))
        self.assertIn('已移入归档：0', self.run_cli(*command, 'archive'))
        self.assertIn('没有需要复测', self.run_cli(*command, 'retest', '--id', ident, '--stale'))
        # 再运行同一命令必须新建轮次，保留两轮实际执行历史。
        first_batch = current['retestBatch']['id']
        self.run_cli(*command, 'retest', '--id', ident, '--max-plies', '1', '--stop-after-first')
        current = next(c for c in self.read_store()['cases'] if c['id'] == ident)
        self.assertNotEqual(current['retestBatch']['id'], first_batch)
        self.assertEqual(len(current['retests']), 4)
        self.assertEqual(current['status'], 'pending')

    def test_off_book_interactive_move_reports_original_algorithm(self):
        output = self.run_cli('play_cli.py', input='a1\nq\n')
        self.assertIn('决策：纯算法', output)
        self.assertIn('AI 落子', output)
        self.assertNotIn('决策：谱外开局规则', output)
        moves = json.loads((self.home / 'games.json').read_text(encoding='utf-8'))['cli']['moves']
        self.assertEqual(moves[0], {'r': 0, 'c': 0, 'player': 1})
        self.assertEqual(len(moves), 2)
        self.assertNotEqual((moves[1]['r'], moves[1]['c']), (0, 0))

    def test_finished_game_can_start_a_new_game_from_cli(self):
        # CLI 每次启动创建新会话，不载入终局存档；真实走完一局再测试 n。
        # 黑方沿外围间隔落子，不形成五连，给中心的白方完成进攻。
        inputs = ['h8'] + [col + str(row) for col in 'ace' for row in range(1, 16, 2)]
        output = self.run_cli('play_cli.py', input='\n'.join(inputs + ['n', 'h8', 'q']) + '\n')
        self.assertIn('AI 获胜', output)
        self.assertIn('输入 n 开新局', output)
        current = json.loads((self.home / 'games.json').read_text(encoding='utf-8'))['cli']
        self.assertIsNone(current['winner'])
        self.assertEqual(len(current['moves']), 2)
        self.assertEqual(current['moves'][0], {'r': 7, 'c': 7, 'player': 1})

    def test_archive_command_preserves_a_completed_round(self):
        # 归档命令的输入是已验证的存储状态；复测实跑另由上面的用例覆盖。
        with patch.dict(os.environ, {'GOMOKU_HOME': str(self.home)}):
            case, _ = lb.add_case([{'r': 7, 'c': 7, 'player': 1}], 1, winner=2)
            batch = lb.begin_retest_batch(case, [{'startPly': 1}])
            lb._record_retest(case, {'startPly': 1, 'result': 'win',
                                    'batchId': batch['id'], 'fingerprint': batch['fingerprint']}, None)
        output = self.run_cli('-m', 'gomoku.tools.loss_book', 'archive')
        self.assertIn('已移入归档：1', output)
        store = self.read_store()
        self.assertEqual(store['cases'], [])
        self.assertEqual(store['archive'][0]['id'], case['id'])
        self.assertEqual(store['archive'][0]['retests'][0]['batchId'], batch['id'])

    def test_gomocup_cli_reports_version_and_returns_a_legal_first_move(self):
        output = self.run_cli('-m', 'gomoku.protocol', input='ABOUT\nSTART 15\nBEGIN\nEND\n')
        self.assertIn('version="1.9.0"', output)
        self.assertIn('OK', output)
        self.assertIn('7,7', output.splitlines())


if __name__ == '__main__':
    unittest.main()
