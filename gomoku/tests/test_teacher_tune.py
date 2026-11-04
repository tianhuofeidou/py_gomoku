# -*- coding: utf-8 -*-
"""离线调参反例：缓存口径、视角、数据泄漏、预算、断点与实际 CLI。"""
import copy
import io
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from gomoku.core import search as S
from gomoku.core.deep_search import DeepSearch
from gomoku.core.utils import empty_board
from gomoku.tools import teacher_tune as U, tune_params as T


MOVES = [[7, 7, 1], [6, 6, 2], [5, 5, 1], [6, 8, 2],
         [5, 6, 1], [7, 6, 2], [4, 6, 1], [8, 8, 2]]


def position(moves=None):
    moves = copy.deepcopy(MOVES if moves is None else moves)
    return dict(id=T.digest(moves), moves=moves, player=len(moves) % 2 + 1,
                group='example', split='train')


def sample_games():
    rng = random.Random(19)
    games = []
    for _ in range(6):
        cells = [(r, c) for r in range(3, 12) for c in range(3, 12)]
        rng.shuffle(cells)
        moves = [dict(r=r, c=c, player=i % 2 + 1) for i, (r, c) in enumerate(cells[:12])]
        games.append(dict(moves=moves, opening_moves=[[m['r'], m['c']] for m in moves[:3]]))
    return dict(games=games)


class ParameterTests(unittest.TestCase):
    def test_exact_groups_and_restore_on_exception(self):
        p = T.defaults()
        self.assertEqual([len(v) for v in T.GROUPS.values()], [24, 24, 8, 4])
        self.assertEqual(set(p), set(k for keys in T.GROUPS.values() for k in keys))
        before = copy.deepcopy(p)
        score_max, t0 = S.SCORE_MAX, DeepSearch.T0
        changed = dict(p, my_0=450, type_5=210, dt_m=320)
        with self.assertRaisesRegex(RuntimeError, 'abort'):
            with T.parameter_scope(changed, 80):
                self.assertEqual(DeepSearch.MY_SUB_VALUE[5], 450)
                self.assertEqual(S.TYPE_SCORE[5], 210)
                self.assertEqual(DeepSearch._dt(80), 320 * .2 ** 2 + 3)
                raise RuntimeError('abort')
        self.assertEqual(T.defaults(), before)
        self.assertEqual((S.SCORE_MAX, DeepSearch.T0), (score_max, t0))

    def test_projection_prevents_scale_explosion_and_invalid_order(self):
        base = T.defaults()
        p = dict(base, my_0=.001, my_3=900000, threshold_0=-10, threshold_2=10,
                 gear_window=9.8, near_attack_opp=1000)
        p = T.project(p, base)
        T.validate(p, base)
        self.assertEqual(p['gear_window'], 10)
        self.assertLess(p['near_attack_opp'], 0)
        self.assertGreaterEqual(p['my_0'], p['my_3'])
        self.assertGreaterEqual(p['threshold_0'], p['threshold_2'])
        self.assertLessEqual(sum(p[k] for k in T.LEAF_KEYS), 1.2 * sum(base[k] for k in T.LEAF_KEYS) + 1e-6)
        for seed in range(20):
            for group in T.GROUPS:
                T.validate(T.perturb(base, base, group, seed), base)
        for invalid in (float('nan'), float('inf'), True):
            with self.assertRaises(ValueError):
                T.project(dict(base, dt_m=invalid), base)

    def test_expression_shared_branches_and_normalized_weights(self):
        feature = tuple([1] + [0] * 23 + [1.0])
        leaf = T.Expression(20, ('leaf', feature))
        value = DeepSearch._weighted_average([(1, leaf), (.6, leaf), (.2, 100)])
        constant, terms = T.flatten(value)
        self.assertAlmostEqual(value, 52 / 1.8)
        self.assertAlmostEqual(constant, 20 / 1.8)
        self.assertEqual(len(terms), 1)
        self.assertAlmostEqual(terms[0][0], 1.6 / 1.8)

    def test_leaf_derivative_matches_finite_difference(self):
        p = T.defaults()
        feature = list(range(1, 25)) + [1.2]
        row = dict(constant=25, terms=[[.6, feature]])
        _, grad = T.replay_row(row, p, gradient=True)
        for i in (0, 5, 12, 23):
            name = T.LEAF_KEYS[i]
            a, b = dict(p), dict(p)
            a[name] += .0001
            b[name] -= .0001
            numerical = (T.replay_row(row, a)[0] - T.replay_row(row, b)[0]) / .0002
            self.assertAlmostEqual(grad[i], numerical, places=5)

    def test_fit_can_improve_a_wrong_ranking_without_neural_network(self):
        p = T.defaults()
        a = [0] * 24 + [1.0]
        b = a.copy()
        a[0], b[4] = 1, 1
        rows = [dict(r=1, c=1, state=0, constant=0, terms=[[1, a]]),
                dict(r=1, c=2, state=0, constant=0, terms=[[1, b]])]
        case = dict(tree=dict(rows=rows), utilities={'1,1': -1, '1,2': 1})
        before = T.replay_row(rows[1], p)[0] - T.replay_row(rows[0], p)[0]
        fitted, info = T.fit_leaf([case], p, p, steps=120)
        after = T.replay_row(rows[1], fitted)[0] - T.replay_row(rows[0], fitted)[0]
        self.assertGreater(after, before + 10)
        self.assertEqual(info['pairs'], 1)
        # 无有效梯度不能谎称训练成功；不同五态档也不能靠评分跨档。
        rows[0]['state'] = 1
        unchanged, info = T.fit_leaf([case], p, p, steps=10)
        self.assertEqual(unchanged, p)
        self.assertEqual(info['steps'], 0)
        self.assertEqual(T.choose(rows, fitted)['c'], 1)


class SearchTests(unittest.TestCase):
    def test_black_white_replay_and_changed_leaf_weights_match_fresh_search(self):
        base = T.defaults()
        for moves in (MOVES, MOVES + [[5, 8, 1]]):
            pos = position(moves)
            original = copy.deepcopy(pos)
            tree = T.search_position(pos, base, t0=80)
            normal = T.search_position(pos, base, t0=80, record=False)
            self.assertEqual(tree['move'], normal['move'])
            self.assertEqual(pos, original)
            changed = T.perturb(base, base, 'leaf', 9)
            fresh = T.search_position(pos, changed, t0=80)
            self.assertEqual([(r['r'], r['c'], r['state']) for r in tree['rows']],
                             [(r['r'], r['c'], r['state']) for r in fresh['rows']])
            for row, actual in zip(tree['rows'], fresh['rows']):
                self.assertAlmostEqual(T.replay_row(row, changed)[0], actual['score'], places=6)
            self.assertEqual(T.defaults(), base)
        self.assertTrue(any(row['terms'] for row in tree['rows']))

    def test_search_budget_and_parameters_restore(self):
        p = T.defaults()
        with self.assertRaises(T.SearchLimit):
            T.search_position(position(), p, t0=240, node_limit=1)
        self.assertEqual(T.defaults(), p)

    def test_cache_reuse_only_for_leaf_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = dict(t0=80, node_limit=1000, search_seconds=5)
            e = U.Experiment(directory, settings, [], None, 'source-v1', time.monotonic() + 30)
            base = T.defaults()
            result = dict(rows=[], move=[1, 1])
            with mock.patch.object(U, 'isolated_search', return_value=result) as run:
                e.tree(position(), base)
                e.tree(position(), dict(base, my_0=450))
                self.assertEqual(run.call_count, 1)
                e.tree(position(), dict(base, type_5=210))
                self.assertEqual(run.call_count, 2)
                e.tree(position(MOVES[:-1]), base)
                self.assertEqual(run.call_count, 3)
                e.source = 'source-v2'
                e.tree(position(), base)
                self.assertEqual(run.call_count, 4)

    def test_compute_gate_rejects_before_labeling_and_lock_releases(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = dict(cost_ratio=1.1, time_ratio=1.25)
            e = U.Experiment(directory, settings, [position(), position()], None, 's', float('inf'))
            baseline = dict(nodes=100, seconds=1, positions=2)
            with mock.patch.object(e, 'tree', return_value=dict(nodes=120, seconds=.1)) as tree:
                with self.assertRaises(T.SearchLimit):
                    e.check_compute(T.defaults(), 'train', baseline)
                self.assertEqual(tree.call_count, 1)
            path = Path(directory) / 'run.lock'
            with U.RunLock(path):
                with self.assertRaises(ValueError):
                    with U.RunLock(path):
                        self.fail('不能获得已占用的锁')
            with U.RunLock(path):
                pass


class DatasetTests(unittest.TestCase):
    def test_grouped_split_deduplicates_mirrors_and_keeps_both_colors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'games.json'
            games = sample_games()
            twin = copy.deepcopy(games['games'][0])
            for m in twin['moves']:
                m['c'] = 14 - m['c']
            twin['opening_moves'] = [[r, 14-c] for r, c in twin['opening_moves']]
            games['games'].append(twin)
            U.write_json(path, games)
            data = U.prepare([path], limit=100, min_ply=8, max_ply=11, stride=2)
            U.validate_positions(data['positions'])
            self.assertEqual(len(data['positions']), 24)
            self.assertEqual({p['player'] for p in data['positions']}, {1, 2})
            self.assertEqual(data, U.prepare([path], limit=100, min_ply=8, max_ply=11, stride=2))
            duplicate = data['positions'] + [copy.deepcopy(data['positions'][0])]
            duplicate[-1]['split'] = 'test'
            with self.assertRaises(ValueError):
                U.validate_positions(duplicate)
            leak = copy.deepcopy(data['positions'])
            first = leak[0]
            first['group'] = next(p['group'] for p in leak if p['split'] != first['split'])
            with self.assertRaises(ValueError):
                U.validate_positions(leak)

    def test_illegal_or_terminal_input_is_not_silently_accepted(self):
        for moves in ([[[15, 0, 1]]], [[[0, 0, 2]]], [[[0, 0, 1], [0, 0, 2]]]):
            with self.assertRaises(ValueError):
                U.board_of(moves[0])
        win = [[7, 3, 1], [0, 0, 2], [7, 4, 1], [0, 2, 2], [7, 5, 1],
               [0, 4, 2], [7, 6, 1], [0, 6, 2], [7, 7, 1]]
        self.assertTrue(U.board_of(win)[1])
        with self.assertRaises(ValueError):
            U.board_of(win + [[1, 0, 2]])


class TeacherTests(unittest.TestCase):
    def test_teacher_transport_coordinates_cache_and_unknown_score(self):
        sent = []

        class Input:
            def write(self, s):
                sent.append(s)

            def flush(self):
                pass

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as directory:
            exe = Path(directory) / 'teacher.exe'
            exe.write_bytes(b'fixture')
            teacher = U.RapfiTeacher(exe, Path(directory) / 'run')
            proc = mock.Mock(stdin=Input(), stdout=io.StringIO('OK\nMESSAGE Eval -50 | Time 1\n1,2\n'))
            proc.poll.return_value = 0
            pos = position(MOVES + [[5, 8, 1]])
            with mock.patch.object(U.subprocess, 'Popen', return_value=proc) as popen:
                data = teacher.root(pos)
                self.assertEqual(data['move'], [2, 1])
                self.assertEqual(teacher.root(pos)['evaluation'], data['evaluation'])
                self.assertEqual(popen.call_count, 1)
            self.assertIn('INFO thread_num 1\n', sent)
            self.assertIn('INFO timeout_match 0\n', sent)
            self.assertIn('7,7,2\n', sent)  # 当前白方，黑子作为对手 field=2。
            self.assertIn('6,6,1\n', sent)
            label = next(teacher.cache.glob('*.json'))
            corrupted = U.read_json(label)
            corrupted['evaluation']['utility'] = 0
            U.write_json(label, corrupted)
            with self.assertRaisesRegex(ValueError, '缓存指纹'):
                teacher.root(pos)
            missing = U.RapfiTeacher(exe, Path(directory) / 'missing')
            proc = mock.Mock(stdin=Input(), stdout=io.StringIO('OK\n1,2\n'))
            proc.poll.return_value = 0
            with mock.patch.object(U.subprocess, 'Popen', return_value=proc):
                with self.assertRaisesRegex(ValueError, 'Eval'):
                    missing.root(pos)
            self.assertFalse(list(missing.cache.glob('*.json')))

    def test_normal_mate_and_missing_eval(self):
        self.assertGreater(U.parse_eval(['MESSAGE Depth 2 | Eval 123 | Time 1ms'])['utility'], 0)
        self.assertEqual(U.parse_eval(['MESSAGE Eval +M31 | Node 5'])['utility'], 1)
        self.assertEqual(U.parse_eval(['MESSAGE Eval -M5 | Node 5'])['utility'], -1)
        with self.assertRaises(ValueError):
            U.parse_eval(['MESSAGE no score', '1,1'])

    def test_child_value_is_opponents_perspective_and_terminal_skips_teacher(self):
        t = object.__new__(U.RapfiTeacher)
        with mock.patch.object(t, '_query', return_value={'evaluation': {'utility': .7}}) as query:
            self.assertEqual(t.utility(position(), [1, 1]), -.7)
            self.assertEqual(query.call_args.args[0][-1], [1, 1, 1])
            win = [[7, 3, 1], [0, 0, 2], [7, 4, 1], [0, 2, 2],
                   [7, 5, 1], [0, 4, 2], [7, 6, 1], [0, 6, 2]]
            self.assertEqual(t.utility(position(win), [7, 7]), 1)
            self.assertEqual(query.call_count, 1)
            with self.assertRaises(ValueError):
                t.utility(position(), [15, 0])

    def test_immediate_five_fact_restores_board(self):
        board = empty_board()
        for c in range(3, 7):
            board[7][c] = 2
        before = copy.deepcopy(board)
        self.assertEqual(U.immediate_wins(board, 2), [[7, 2], [7, 7]])
        self.assertEqual(board, before)

    def test_cost_gate_checks_compute_instead_of_only_temperature(self):
        base = dict(nodes=100, seconds=1, positions=10)
        settings = dict(cost_ratio=1.1, time_ratio=1.25)
        self.assertFalse(U.within_cost(dict(base, nodes=120), base, settings))
        self.assertFalse(U.within_cost(dict(base, seconds=1.5), base, settings))
        self.assertTrue(U.within_cost(base, base, settings))


class CliTests(unittest.TestCase):
    def test_project_outputs_only_allowed_in_ignored_experiments(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory) / 'repo'
            with mock.patch.object(U, 'ROOT', repo / 'gomoku'):
                for relative in ('', 'gomoku/data/run', 'storages-other/run', 'storages/experiments-other/run'):
                    with self.assertRaises(ValueError):
                        U.run_experiment(SimpleNamespace(out=str(repo / relative)))
                for path in (repo / 'storages/experiments/run', Path(directory) / 'external-run'):
                    with mock.patch.object(U, '_run_locked', return_value=0) as run:
                        self.assertEqual(U.run_experiment(SimpleNamespace(out=str(path))), 0)
                        self.assertEqual(run.call_count, 1)

    def command(self, *args, input=None, timeout=30):
        env = dict(os.environ, PYTHONIOENCODING='utf-8', DSH_GOMOKU_GA='0')
        return subprocess.run([sys.executable, '-B', '-m', 'gomoku.tools.teacher_tune', *args],
                              input=input, text=True, encoding='utf-8', capture_output=True,
                              env=env, timeout=timeout)

    def test_prepare_help_utf8_and_experimental_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            path, out = Path(directory) / 'games.json', Path(directory) / 'positions.json'
            U.write_json(path, sample_games())
            args = ('prepare', '--games', str(path), '--out', str(out), '--positions', '12')
            result = self.command(*args)
            self.assertEqual(result.returncode, 0, result.stderr)
            U.validate_positions(U.read_json(out)['positions'])
            self.assertNotEqual(self.command(*args).returncode, 0)
            self.assertEqual(self.command('run', '--help').returncode, 0)
            params = Path(directory) / 'params.json'
            U.write_json(params, dict(schema=U.SCHEMA, source=U.source_fingerprint(),
                                     params=T.defaults(), baseline=T.defaults(), t0=80))
            result = self.command('protocol', '--params', str(params), input='START 15\nBEGIN\nEND\n')
            self.assertEqual(result.returncode, 0, result.stderr)
            lines = result.stdout.splitlines()
            self.assertEqual(lines, ['OK', '7,7'])
            self.assertIn('实验参数', result.stderr)

    def test_autorun_resume_and_holdout_with_fake_teacher_real_search(self):
        # 教师替身只替代外部标注，搜索仍由实际 CLI 子进程执行。
        calls = []

        class FakeTeacher:
            def __init__(self, *args):
                self.identity = {'fixture': 1, 'nodes': 200000, 'milliseconds': 2000}

            def root(self, pos):
                calls.append(pos['split'])
                return {'move': [14, 14]}

            def utility(self, pos, move):
                calls.append(pos['split'])
                return (move[0] + move[1]) / 28

        with tempfile.TemporaryDirectory() as directory:
            games = Path(directory) / 'games.json'
            out = Path(directory) / 'run'
            U.write_json(games, sample_games())
            args = ['run', '--games', str(games), '--teacher', 'fixture', '--out', str(out),
                    '--positions', '6', '--rounds', '1', '--trials', '1', '--leaf-steps', '2',
                    '--max-seconds', '90']
            with mock.patch.object(U, 'RapfiTeacher', FakeTeacher):
                with mock.patch.object(U.Experiment, 'cases', side_effect=U.BudgetStop('fixture-budget')):
                    self.assertEqual(U.main(args), 2)
                self.assertEqual(U.read_json(out / 'checkpoint.json')['status'], 'paused')
                self.assertEqual(U.main(args + ['--resume']), 0)
                checkpoint = U.read_json(out / 'checkpoint.json')
                self.assertEqual(checkpoint['status'], 'complete')
                self.assertEqual(checkpoint['next_step'], 3)
                self.assertNotIn('test', calls)
                self.assertIn('validation', calls)
                self.assertTrue((out / 'best.json').exists())
                self.assertEqual(U.main(args + ['--resume']), 0)
                self.assertEqual(U.main(args + ['--resume', '--t0', '160']), 1)
                params_before = U.read_json(out / 'best.json')['params']
                self.assertEqual(U.main(['evaluate', '--out', str(out), '--teacher', 'fixture']), 0)
                self.assertIn('test', calls)
                self.assertEqual(U.read_json(out / 'best.json')['params'], params_before)
                self.assertEqual(len(list(out.glob('test-*.json'))), 1)



# 本轮发布默认改为8/3/120；以下旧机制回归显式固定旧口径。
from gomoku.tests._legacy_defaults import enter_legacy as _enter_legacy, leave_legacy as _leave_legacy

def setUpModule():
    _enter_legacy()

def tearDownModule():
    _leave_legacy()
if __name__ == '__main__':
    unittest.main()
