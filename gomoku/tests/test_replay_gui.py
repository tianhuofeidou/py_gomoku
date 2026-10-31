"""手动复盘的输入边界、视角、只读加载与分支隔离。"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gomoku.tools.replay_gui import (
    ReplayPosition, read_records, scan_directory, _load_last_directory, _save_last_directory,
)


def sample():
    return [dict(r=7, c=7, player=1), dict(r=7, c=8, player=2),
            dict(r=8, c=7, player=1, ms=123.4)]


class ReplayTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)

    def write(self, name, data):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding='utf-8')
        return path

    def record(self):
        path = self.write('game.json', dict(id='a', our_color='white', result='loss', moves=sample()))
        return read_records(path)[0][0]

    def test_dataset_preferred_over_duplicate_defeats_and_results(self):
        row = dict(id='a', result='loss', our_color='white', moves=sample())
        path = self.write('dataset.json', [row, dict(row, id='smoke', scored=False)])
        self.write('defeats.json', dict(games=[row]))
        self.write('results/game.json', dict(games=[row]))
        before = path.read_bytes()
        records, errors = scan_directory(self.root)
        self.assertEqual([r.identity for r in records], ['a'])
        self.assertEqual(records[0].our_color, 'white')
        self.assertFalse(errors)
        self.assertEqual(path.read_bytes(), before)

    def test_match_games_and_sidecar_color_and_outcome(self):
        path = self.write('results.json', dict(games=[dict(moves=sample(), winner='B')]))
        self.write('results.json.meta.json', dict(our_color='white', game_id='test', opponent='foo'))
        records, errors = read_records(path)
        self.assertFalse(errors)
        self.assertEqual((records[0].our_color, records[0].result), ('white', 'loss'))

    def test_dataset_also_loads_prefixed_retests_but_not_duplicate_originals(self):
        original = dict(id='original', result='loss', our_color='white', moves=sample())
        path = self.write('dataset.json', [original])
        before = path.read_bytes()
        self.write('defeats.json', dict(games=[original]))
        self.write('重赛_TPTC_从48手.json', dict(original, id='retest-48', opponent_label='重赛 TPTC'))
        records, errors = scan_directory(self.root)
        self.assertFalse(errors)
        self.assertEqual([r.identity for r in records], ['retest-48', 'original'])
        self.assertEqual(records[0].our_color, 'white')
        self.assertEqual(path.read_bytes(), before)

    def test_a_white_false_is_black_and_no_color_is_unknown(self):
        path = self.write('results.json', dict(games=[dict(moves=sample(), winner='B', a_white=False),
                                                   dict(moves=sample(), winner='B')]))
        records, _ = read_records(path)
        self.assertEqual(records[0].result, 'win')
        self.assertEqual(records[1].our_color, '')
        self.assertEqual(records[1].result, '')

    def test_bad_moves_reported_without_hiding_valid_game(self):
        bad = [dict(r=15, c=0, player=1), dict(r=True, c=0, player=1),
               dict(r=0, c=0, player=2), dict(r=0, c=0, player=1, ms=float('nan'))]
        rows = [dict(moves=[m]) for m in bad]
        rows += [dict(moves=[dict(r=0, c=0, player=1), dict(r=0, c=0, player=2)])]
        rows += [dict(moves=sample())]
        records, errors = read_records(self.write('games.json', rows))
        self.assertEqual(len(records), 1)
        self.assertEqual(len(errors), 5)

    def test_moves_after_five_rejected(self):
        moves = []
        for c in range(5):
            moves.append(dict(r=0, c=c, player=1))
            if c < 4:
                moves.append(dict(r=1, c=c, player=2))
        records, errors = read_records(self.write('win.json', dict(moves=moves)))
        self.assertEqual(len(records), 1)
        records, errors = read_records(self.write('after.json', dict(moves=moves + [dict(r=2, c=0, player=2)])))
        self.assertFalse(records)
        self.assertIn('成五后', errors[0])

    def test_root_and_results_scanned_without_runtime_recursion(self):
        self.write('results/one.json', dict(moves=sample()))
        self.write('runtime/hidden.json', dict(moves=sample()))
        self.write('config.json', dict(config=True))
        records, errors = scan_directory(self.root)
        self.assertEqual(len(records), 1)
        self.assertFalse(errors)

    def test_bad_json_visible_and_not_overwritten(self):
        path = self.root / 'bad.json'
        path.write_text('{oops', encoding='utf-8')
        records, errors = scan_directory(self.root)
        self.assertFalse(records)
        self.assertEqual(len(errors), 1)
        self.assertEqual(path.read_text(encoding='utf-8'), '{oops')

    def test_branch_rewind_replace_and_original_unchanged(self):
        record = self.record()
        original = record.moves
        p = ReplayPosition(record)
        p.seek(2)
        p.toggle_branch()
        p.place(1, 1)
        p.place(1, 2)
        self.assertEqual(p.moves()[-1][2], 2)
        p.seek(3)
        p.place(2, 2)
        self.assertEqual(p.moves()[-1][:2], (2, 2))
        self.assertEqual(len(p.branch), 2)
        p.toggle_branch()
        self.assertEqual(p.ply, 2)
        p.seek(3)
        self.assertEqual(p.moves()[-1], original[-1])
        self.assertEqual(record.moves, original)

    def test_branch_rejects_occupied_outside_and_without_mode(self):
        p = ReplayPosition(self.record())
        with self.assertRaises(ValueError):
            p.place(0, 0)
        p.seek(1)
        p.toggle_branch()
        for r, c in [(7, 7), (-1, 0), (0, 15), (True, 0)]:
            with self.subTest(point=(r, c)), self.assertRaises(ValueError):
                p.place(r, c)

    def test_replay_does_not_construct_game_or_engine(self):
        record = self.record()
        with (patch('gomoku.game.Game', side_effect=AssertionError('no Game')),
              patch('gomoku.service.create_engine', side_effect=AssertionError('no engine'))):
            p = ReplayPosition(record)
            for n in [0, 1, 3, 0, 3]:
                p.seek(n)
                self.assertEqual(sum(v != 0 for row in p.board() for v in row), n)

    def test_directory_setting_is_separate_and_survives_new_reader(self):
        data = self.root / '数据目录'
        data.mkdir()
        home = self.root / 'home'
        with patch.dict('os.environ', {'GOMOKU_HOME': str(home)}):
            self.assertIsNone(_load_last_directory())
            _save_last_directory(data)
            self.assertEqual(_load_last_directory(), str(data.resolve()))
            self.assertEqual([p.name for p in home.iterdir()], ['replay-settings.json'])

    def test_failed_atomic_replace_preserves_old_directory(self):
        home = self.root / 'home'
        with patch.dict('os.environ', {'GOMOKU_HOME': str(home)}):
            _save_last_directory(self.root)
            before = (home / 'replay-settings.json').read_bytes()
            with patch('gomoku.tools.replay_gui.os.replace', side_effect=PermissionError('denied')):
                with self.assertRaises(PermissionError):
                    _save_last_directory(self.root / 'new')
            self.assertEqual((home / 'replay-settings.json').read_bytes(), before)
            self.assertEqual(len(list(home.iterdir())), 1)


if __name__ == '__main__':
    unittest.main()
