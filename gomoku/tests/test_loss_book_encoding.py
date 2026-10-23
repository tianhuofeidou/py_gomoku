"""覆盖 Windows CI 的 cp1252 标准流，使用真实败局库 CLI。"""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from gomoku.tools import loss_book as cli


class LossBookEncodingTest(unittest.TestCase):
    def test_import_with_cp1252_streams_emits_utf8(self):
        with tempfile.TemporaryDirectory(prefix='gomoku-cli-encoding-') as tmp:
            path = Path(tmp) / 'match.json'
            path.write_text(json.dumps({'games': [
                {'game': 1, 'a_white': False, 'winner': 'W',
                 'moves': [{'r': 7, 'c': 7, 'player': 1}]}]}), encoding='utf-8')
            with io.BytesIO() as out, io.BytesIO() as err:
                stdout = io.TextIOWrapper(out, encoding='cp1252')
                stderr = io.TextIOWrapper(err, encoding='cp1252')
                try:
                    with patch.dict(os.environ, {'GOMOKU_HOME': tmp}), \
                            patch.object(sys, 'stdout', stdout), patch.object(sys, 'stderr', stderr):
                        self.assertEqual(cli.main(['import', '--json', str(path)]), 0)
                        self.assertEqual(stderr.encoding, 'utf-8')
                    stdout.flush()
                    self.assertIn('导入完成：新增 1 例', out.getvalue().decode('utf-8'))
                finally:
                    stdout.detach()
                    stderr.detach()

    def test_module_entry_with_cp1252_environment_emits_utf8_help(self):
        with tempfile.TemporaryDirectory(prefix='gomoku-cli-encoding-') as tmp:
            env = dict(os.environ, GOMOKU_HOME=tmp, PYTHONIOENCODING='cp1252', PYTHONUTF8='0')
            result = subprocess.run([sys.executable, '-B', '-S', '-m', 'gomoku.tools.loss_book', '--help'],
                                    cwd=Path(__file__).resolve().parents[2], env=env,
                                    capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr.decode('ascii', errors='backslashreplace'))
            self.assertIn('败局回归库', result.stdout.decode('utf-8'))


if __name__ == '__main__':
    unittest.main()
