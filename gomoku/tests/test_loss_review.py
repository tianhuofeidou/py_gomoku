"""先冻结人工标签，再检验定位器；已见教师棋例不冒充独立真值。"""
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from gomoku.tools import loss_review as R
from gomoku import loss_book as L
from gomoku.game import check_win

ANNOTATIONS = json.loads((Path(__file__).parent / 'data' / 'loss_locator_annotations.json').read_text(encoding='utf-8'))


def point(name):
    return [int(name[1:]) - 1, ord(name[0]) - 65]


def case_of(fixture):
    return dict(moves=[dict(r=point(n)[0], c=point(n)[1], player=i % 2 + 1)
                       for i, n in enumerate(fixture['moves'])], aiColor=fixture['aiColor'],
                kind='loss', config=dict(opening_plies=fixture['setup_plies']))


def fixture(name):
    return next(f for f in ANNOTATIONS['cases'] if f['id'] == name)


def exact_wins(board, color):
    """独立检查标注真值，不调用定位器的棋型扫描/评分/五态。"""
    wins = set()
    for r in range(15):
        for c in range(15):
            if board[r][c]:
                continue
            child = [line[:] for line in board]
            child[r][c] = color
            if check_win(child, r, c, color):
                wins.add((r, c))
    return wins


class ProvenForkTeacher:
    """协议层确定性替身；赢/输标签来自下方独立核验的人工强制链。"""
    def __init__(self, f, unstable=False):
        self.case = case_of(f)
        self.h = [[m['r'], m['c'], m['player']] for m in self.case['moves']][:f['setup_plies']]
        self.actual = point(f['moves'][f['setup_plies']])
        self.unstable = unstable
        self.requests = []

    def identity(self, nodes):
        return dict(test_teacher='manual-forcing-chain', nodes=nodes)

    def prefetch(self, histories, nodes):
        self.requests += [(h, nodes) for h in histories]

    def read(self, h, nodes):
        if h == self.h:
            move, raw = point('H8'), '+M5'
        elif h == self.h + [[*self.actual, self.case['aiColor']]]:
            move, raw = point('H2'), '+M3'
        elif h == self.h + [[*point('H8'), self.case['aiColor']]]:
            move, raw = point('I8'), ('+M3' if self.unstable and nodes > 200000 else '-M4')
        else:
            raise AssertionError('未预期的教师请求')
        names = R.cd(move)
        return dict(key=R.digest([h, nodes]), identity=self.identity(nodes), move=move,
            evaluation=R.value(raw), seconds=.01,
            messages=['MESSAGE Depth 9-10 | Eval %s | Time 1s | %s' % (raw, names)])


class ManualAnnotationTests(unittest.TestCase):
    def test_all_hard_annotations_are_legal_complete_losses(self):
        for f in ANNOTATIONS['cases']:
            if f['basis'] != 'manual-rule-proof':
                continue
            with self.subTest(f=f['id']):
                board = [[0] * 15 for _ in range(15)]
                ended = False
                for i, name in enumerate(f['moves']):
                    r, c = point(name)
                    self.assertFalse(ended)
                    self.assertEqual(board[r][c], 0)
                    board[r][c] = i % 2 + 1
                    ended = check_win(board, r, c, i % 2 + 1)
                self.assertTrue(ended)
                self.assertNotEqual((len(f['moves']) - 1) % 2 + 1, f['aiColor'])

    def test_manual_forcing_chain_proofs_both_colors(self):
        for name in ('black-forcing-fork', 'white-forcing-fork'):
            f = fixture(name)
            ai = f['aiColor']
            board = [[0] * 15 for _ in range(15)]
            for i, pos in enumerate(f['moves'][:f['setup_plies']]):
                r, c = point(pos)
                board[r][c] = i % 2 + 1
            self.assertFalse(exact_wins(board, ai))
            self.assertFalse(exact_wins(board, 3 - ai))
            after = [line[:] for line in board]
            r, c = point('H8'); after[r][c] = ai
            self.assertEqual(exact_wins(after, ai), {tuple(point('I8'))})
            self.assertFalse(exact_wins(after, 3 - ai))
            r, c = point('I8'); after[r][c] = 3 - ai
            r, c = point('H9'); after[r][c] = ai
            self.assertEqual(exact_wins(after, ai), {tuple(point('H5')), tuple(point('H10'))})
            self.assertFalse(exact_wins(after, 3 - ai))
            # 实战A3让对手H2产生双胜点；本方无立即成五可反击。
            actual = [line[:] for line in board]
            r, c = point('A3'); actual[r][c] = ai
            r, c = point('H2'); actual[r][c] = 3 - ai
            self.assertEqual(exact_wins(actual, 3 - ai), {tuple(point('D2')), tuple(point('I2'))})
            self.assertFalse(exact_wins(actual, ai))

    def test_rule_locator_against_frozen_labels(self):
        for f in ANNOTATIONS['cases']:
            if f.get('teacher_required'):
                continue
            with self.subTest(f=f['id']):
                result = R.review_case(case_of(f))
                self.assertEqual(result['status'], 'complete')
                self.assertEqual(sorted(s['ply'] for s in result['suspects']), f['expected_suspect_plies'])
                self.assertTrue(all(s['reason'] == f['expected_reason'] for s in result['suspects']))

    def test_forcing_fork_locator_against_hand_proof(self):
        for name in ('black-forcing-fork', 'white-forcing-fork'):
            f = fixture(name)
            result = R.review_case(case_of(f), teacher=ProvenForkTeacher(f))
            self.assertEqual([s['ply'] for s in result['suspects']], f['expected_suspect_plies'])
            self.assertEqual(result['suspects'][0]['alternative'], 'H8')

    def test_latest_recovered_opportunity_wins_truncation(self):
        f = fixture('black-opportunity-restored')
        result = R.review_case(case_of(f), max_suspects=1)
        self.assertEqual(result['suspects'][0]['ply'], f['expected_primary'])

    def test_deeper_refutation_not_promoted(self):
        f = fixture('black-forcing-fork')
        result = R.review_case(case_of(f), teacher=ProvenForkTeacher(f, unstable=True))
        self.assertEqual(result['suspects'], [])
        self.assertEqual(result['rows'][0]['levels'][-1]['category'], 'both-searched-losing')

    def test_single_budget_cannot_confirm_search_loss(self):
        f = fixture('black-forcing-fork')
        self.assertFalse(R.review_case(case_of(f), teacher=ProvenForkTeacher(f), nodes=(200000,))['suspects'])

    def test_unknown_alternative_not_safety_evidence(self):
        f = fixture('black-forcing-fork')
        teacher = ProvenForkTeacher(f)
        original = teacher.read
        def read(h, n):
            out = original(h, n)
            if h != teacher.h and h[-1][:2] == point('H8'):
                out['evaluation'] = R.value('-300')
                out['messages'] = ['MESSAGE Eval -300 | Time 1s | I8']
            return out
        teacher.read = read
        self.assertFalse(R.review_case(case_of(f), teacher=teacher)['suspects'])

    def test_winner_truncation_illegal_and_abnormal_not_diagnosed(self):
        case = case_of(fixture('black-missed-unique-block'))
        for bad in [dict(case, moves=case['moves'][:-1]), dict(case, kind='timeout'),
                    dict(case, moves=case['moves'] + [dict(r=2, c=2, player=1)])]:
            self.assertFalse(R.review_case(bad)['suspects'])

    def test_no_own_engine_dependency_and_no_mutation(self):
        case = case_of(fixture('white-missed-unique-block'))
        before = copy.deepcopy(case)
        with patch('gomoku.tools.review.Engine', side_effect=AssertionError('不应调用本方引擎')):
            suspects, _ = L.locate_suspects(case)
        self.assertEqual(suspects[0]['ply'], 8)
        self.assertEqual(case, before)

    def test_explicit_setup_metadata_instead_of_fixed_skip_six(self):
        case = case_of(fixture('white-missed-unique-block'))
        self.assertEqual(R.review_case(case)['rows'][0]['ply'], 8)
        case['config'] = {}
        result = R.review_case(case)
        self.assertEqual(result['rows'][0]['ply'], 2)
        self.assertTrue(result['notes'])

    def test_rule_labels_survive_all_rotations_and_mirrors(self):
        from gomoku.core.opening_book import transform_point
        for f in ANNOTATIONS['cases']:
            if f.get('teacher_required'):
                continue
            for symmetry in range(8):
                case = case_of(f)
                for m in case['moves']:
                    m['r'], m['c'] = transform_point(m['r'], m['c'], symmetry)
                result = R.review_case(case)
                self.assertEqual(sorted(s['ply'] for s in result['suspects']), f['expected_suspect_plies'])


class ProtocolEvidenceTests(unittest.TestCase):
    class Capture(io.StringIO):
        def close(self):
            self.captured = self.getvalue()
            super().close()

    def process(self, output):
        class Process:
            def __init__(self, output):
                self.stdin = ProtocolEvidenceTests.Capture()
                self.stdout = io.StringIO(output)
                self.killed = False
                self.waited = False
            def poll(self): return None
            def kill(self): self.killed = True
            def wait(self, timeout): self.waited = True; return 0
        return Process(output)

    def test_native_protocol_coordinates_roles_limits_and_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            teacher = R.RapfiReviewer(sys.executable, tmp)
            proc = self.process('OK\nMESSAGE Eval -42\n10,4\n')
            with patch.object(R.subprocess, 'Popen', return_value=proc) as spawn:
                result = teacher._native([[1,5,1], [2,7,2]], teacher.identity(200000))
            self.assertEqual(result['move'], [4,10])
            self.assertIn('5,1,1\n7,2,2\nDONE', proc.stdin.captured)
            self.assertIn('INFO thread_num 1', proc.stdin.captured)
            self.assertIn('INFO max_node 200000', proc.stdin.captured)
            self.assertTrue(proc.killed and proc.waited)
            self.assertEqual(spawn.call_args.kwargs['creationflags'], getattr(subprocess, 'CREATE_NO_WINDOW', 0))

    def test_native_protocol_error_and_timeout_reap_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            for output, expired, error in [('ERROR broken\n', False, RuntimeError), ('', True, R.ReviewStop)]:
                teacher = R.RapfiReviewer(sys.executable, tmp)
                if expired: teacher.deadline = time.monotonic() - 1
                proc = self.process(output)
                with patch.object(R.subprocess, 'Popen', return_value=proc):
                    with self.assertRaises(error): teacher._native([], teacher.identity(200000))
                self.assertTrue(proc.killed and proc.waited)

    def test_query_quota_is_shared_before_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            with R.RapfiReviewer(sys.executable, tmp, max_queries=1, workers=4) as teacher:
                with patch.object(teacher, '_query', side_effect=AssertionError('不应派发')):
                    with self.assertRaises(R.ReviewStop): teacher.prefetch([[], [[0,0,1]]], 200000)

    def test_mate_zero_sign_and_ordinary_units(self):
        self.assertEqual(R.flip(R.value('+M0'))['raw'], '-M0')
        self.assertEqual(R.flip(R.value('-M0'))['raw'], '+M0')
        self.assertEqual(R.parse_eval(['MESSAGE Eval +M3', 'MESSAGE Eval -42'])['raw'], '-42')

    def test_pv_repeated_point_rejected_seconds_and_ms_supported(self):
        for unit in ('150ms', '15s'):
            label = dict(move=point('G6'), evaluation=R.value('100'),
                messages=['MESSAGE Eval 100 | Time %s | G6 H6 G6' % unit])
            self.assertFalse(R.pv_info(label, [])['valid'])
            label['messages'] = ['MESSAGE Eval 100 | Time %s | G6 H6' % unit]
            self.assertTrue(R.pv_info(label, [])['valid'])

    def test_invalid_pv_does_not_promote_stable_score(self):
        f = fixture('black-forcing-fork'); teacher = ProvenForkTeacher(f)
        original = teacher.read
        def read(h, nodes):
            out = original(h, nodes)
            if h != teacher.h and h[-1][:2] == point('H8'):
                out['messages'] = ['MESSAGE Eval -M4 | Time 1s | I8 H9 I8']
            return out
        teacher.read = read
        result = R.review_case(case_of(f), teacher=teacher)
        self.assertFalse(result['suspects']); self.assertTrue(result['warnings'])

    def test_budget_and_protocol_error_visible(self):
        f = fixture('black-forcing-fork')
        for error, status in [(R.ReviewStop('budget'), 'partial'), (ValueError('bad cache'), 'error')]:
            teacher = ProvenForkTeacher(f)
            teacher.prefetch = lambda *a, err=error: (_ for _ in ()).throw(err)
            self.assertEqual(R.review_case(case_of(f), teacher=teacher)['status'], status)

    def test_directory_lock_and_cache_only_never_executes(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = R.RapfiReviewer(sys.executable, tmp)
            second = R.RapfiReviewer(sys.executable, tmp)
            with first:
                with self.assertRaises(R.ReviewStop): second.__enter__()
            self.assertFalse((Path(tmp) / 'loss-review.lock').exists())
            with R.RapfiReviewer(sys.executable, tmp, cache_only=True) as read_only:
                with patch.object(read_only, '_native', side_effect=AssertionError('不允许启动')):
                    with self.assertRaises(R.ReviewStop): read_only.prefetch([[]], 200000)

    def test_cache_identity_checksum_and_raw_score_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            teacher = R.RapfiReviewer(sys.executable, tmp, cache_only=True)
            identity = teacher.identity(200000)
            data = dict(key=R.digest([identity, []]), identity=identity, move=[7,7], seconds=1,
                evaluation=R.value('10'), messages=['MESSAGE Eval 10', '7,7'])
            data['payload_sha256'] = R.digest(data)
            path = teacher._path([], 200000); R.write_json(path, data)
            self.assertEqual(teacher.read([], 200000)['move'], [7,7])
            data['evaluation'] = R.value('-10')
            R.write_json(path, data)
            with self.assertRaises(ValueError): teacher.read([], 200000)
            data.pop('payload_sha256'); data['payload_sha256'] = R.digest(data)
            R.write_json(path, data)
            with self.assertRaisesRegex(ValueError, '原始输出'): teacher.read([], 200000)


class StoreAndCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.old = os.environ.get('GOMOKU_HOME')
        os.environ['GOMOKU_HOME'] = self.tmp.name
        f = fixture('white-missed-unique-block'); data = case_of(f)
        self.case, _ = L.add_case(data['moves'], data['aiColor'], winner=1, kind='loss', config=data['config'])

    def tearDown(self):
        if self.old is None: os.environ.pop('GOMOKU_HOME', None)
        else: os.environ['GOMOKU_HOME'] = self.old
        from gomoku import memory
        memory._global_memory = None
        self.tmp.cleanup()

    def test_save_reloads_and_rejects_changed_or_deleted_case(self):
        snapshot = copy.deepcopy(self.case); result = R.review_case(snapshot)
        store = L.load_store(); store['cases'][0]['config']['changed'] = True; L.save_store(store)
        with self.assertRaises(ValueError): L.save_diagnosis(snapshot, result)
        store['cases'] = []; L.save_store(store)
        with self.assertRaises(ValueError): L.save_diagnosis(snapshot, result)
        self.assertFalse(L.load_store()['cases'])

    def test_partial_keeps_existing_manual_anchor_and_state(self):
        self.case = L.set_suspects(self.case, manual_ply=2)
        result = R.review_case(self.case); result['status'] = 'partial'
        L.save_diagnosis(copy.deepcopy(self.case), result)
        saved = L.load_store()['cases'][0]
        self.assertEqual(saved['manualPly'], 2); self.assertEqual(saved['status'], 'pending')

    def test_new_anchor_invalidates_old_pass_but_retains_history(self):
        store = L.load_store(); old = store['cases'][0]
        old.update(status='passed', retestBatch=dict(id='old'), retests=[dict(batchId='old')])
        L.save_store(store); snapshot = copy.deepcopy(old)
        result = R.review_case(snapshot); L.save_diagnosis(snapshot, result)
        current = L.load_store()['cases'][0]
        self.assertEqual(current['status'], 'pending'); self.assertIsNone(current['retestBatch'])
        self.assertEqual(current['diagnosis']['priorRetestBatch']['id'], 'old')
        self.assertEqual(len(current['retests']), 1)

    def test_real_cli_rule_locate_show_and_bad_manual_does_not_write(self):
        cmd = [sys.executable, '-B', '-m', 'gomoku.tools.loss_book']
        def run(*args):
            return subprocess.run(cmd + list(args), capture_output=True, text=True, encoding='utf-8', timeout=30)
        first = run('locate', '--id', self.case['id'])
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertIn('missed-block', first.stdout)
        show = run('show', '--id', self.case['id'])
        self.assertIn('上次离线诊断', show.stdout)
        before = Path(L.store_file()).read_bytes()
        bad = run('locate', '--id', self.case['id'], '--ply', '999')
        self.assertNotEqual(bad.returncode, 0); self.assertEqual(Path(L.store_file()).read_bytes(), before)
        dry = run('locate', '--id', self.case['id'], '--dry-run')
        self.assertEqual(dry.returncode, 0); self.assertEqual(Path(L.store_file()).read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
