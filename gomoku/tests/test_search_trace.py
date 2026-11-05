"""录像正确性：沿原搜索比较，另用独立手工事件核验时间轴。"""
import json
import multiprocessing as mp
import queue
import time
import unittest

from gomoku.core.engine import Engine
from gomoku.core.utils import empty_board
from gomoku.tools.search_trace import TraceModel, TraceRecorder, TraceStopped, debug_worker


MOVES = [(7, 7, 1), (7, 6, 2), (6, 6, 1), (6, 7, 2), (5, 8, 1), (8, 5, 2)]


def engine_with(observer=None):
    engine = Engine()
    engine._opening_symmetry = 0
    engine.deep_search.T0 = 40
    engine.deep_search.TACTICAL_FREE_PLIES = 0
    engine.deep_search.TACTICAL_COSTS = (8., 8., 8.)
    board = empty_board()
    for r, c, player in MOVES:
        board[r][c] = player
        engine.on_move(board, r, c, player)
    engine.deep_search._debug_observer = observer
    return engine, board


class TraceTest(unittest.TestCase):
    def test_time_travel_does_not_read_future_scores(self):
        events = []
        recorder = TraceRecorder(events.append)
        root = dict(board=bytes(225), children={})
        recorder.record('root', root)
        recorder.record('result', root, score=17., state=0, complete=False)
        recorder.record('result', root, score=99., state=1, complete=True)
        model = TraceModel()
        for event in events:
            model.append(event)
        model.seek(1)
        self.assertNotIn('score', model.at('n1'))
        model.seek(2)
        self.assertEqual(model.at('n1')['score'], 17.)
        self.assertFalse(model.at('n1')['complete'])
        model.seek(3)
        self.assertEqual(model.at('n1')['state'], 1)
        self.assertEqual(model.at('n1')['score'], 99.)

    def test_same_board_different_paths_remain_distinct(self):
        events = []
        recorder = TraceRecorder(events.append)
        a, b = dict(board=bytes(225)), dict(board=bytes(225))
        recorder.record('root', a)
        recorder.record('root', b)
        self.assertEqual(len(recorder.nodes), 2)
        json.dumps(events, allow_nan=False)

    def test_observer_preserves_cold_results_and_rolls_back(self):
        events = []
        recorder = TraceRecorder(events.append)
        plain, board1 = engine_with()
        traced, board2 = engine_with(recorder)
        before = [row[:] for row in board2]
        before_tables = ([row[:] for row in traced.search.sb], [row[:] for row in traced.search.sw])
        expected = plain.analyze_turn(board1, 1)
        recorder.begin(MOVES, 1, traced.deep_search.T0)
        actual = traced.analyze_turn(board2, 1)
        self.assertEqual(expected, actual)
        self.assertEqual(board2, before)
        self.assertEqual((traced.search.sb, traced.search.sw), before_tables)
        self.assertEqual(plain.deep_search.cache_stats, traced.deep_search.cache_stats)
        model = TraceModel()
        for event in events:
            model.append(event)
        model.seek(len(events)-1)
        self.assertTrue(any(e['kind']=='temperature' for e in events))
        self.assertTrue(any(e['kind']=='leaf' for e in events))
        for node_id, node in model.nodes.items():
            board = model.board(node_id)
            moves = model.moves(node_id)
            rebuilt = empty_board()
            for r, c, player in moves:
                self.assertEqual(rebuilt[r][c], 0)
                rebuilt[r][c] = player
            self.assertEqual(board, rebuilt)
        for event in events:
            if event['kind'] == 'temperature':
                before_count = event['tactical_count']-int(event['tactical'])
                expected_cost = traced.deep_search._path_temperature_cost(
                    event['cost_weight'], event['tactical'], before_count,
                    event['temperature_before'])
                self.assertAlmostEqual(event['cost'], expected_cost)

    def test_real_moves_keep_cache_hit_and_node_identity(self):
        events = []
        recorder = TraceRecorder(events.append)
        engine, board = engine_with(recorder)
        engine.deep_search.T0 = 160
        recorder.begin(MOVES, 1, 160)
        engine.analyze_turn(board, 1)
        root = engine.deep_search._tree_root
        path = next(([(r,c,p), move] for (r,c,p), child in root['children'].items()
                     for move in child['children']), None)
        self.assertIsNotNone(path, '真实小局面须含两手缓存路径')
        moves = list(MOVES)
        for r, c, player in path:
            moves.append((r,c,player))
            recorder.real_moves = list(moves)
            board[r][c] = player
            engine.on_move(board, r, c, player)
            self.assertTrue(events[-1]['hit'])
        retained = engine.deep_search._tree_root
        retained_id = recorder.identify(retained)
        recorder.begin(moves, 1, 160)
        engine.analyze_turn(board, 1)
        self.assertIs(engine.deep_search._tree_root, retained)
        self.assertTrue(any(e['kind']=='cache_hit' and e.get('reason')=='cached_turn'
                            and e['node']==retained_id for e in events))
        self.assertEqual(engine.moves, moves)

    def test_resume_records_actual_effective_temperature_once(self):
        events = []
        recorder = TraceRecorder(events.append)
        engine, board = engine_with(recorder)
        ds = engine.deep_search
        node = ds._tree_root
        ds._tree_epoch = 8
        ds._cache_credit = 80.
        node['frontier'] = dict(player=1, temperature=-15., path_weight=.5, epoch=7,
            before_move=True, depth=4, score=120., state=0, terminal=False, tactical_count=5)
        expected, blocked = ds._continuation_temperature(node['frontier'], depth=4)
        first = ds._resume_temperature(node, 1, -15., depth=4, before_move=True)
        second = ds._resume_temperature(node, 1, -15., depth=4, before_move=True)
        self.assertEqual((first, second), (expected, expected))
        resumes = [e for e in events if e['kind']=='resume']
        self.assertEqual(len(resumes), 1)
        self.assertEqual(resumes[0]['old_temperature'], -15.)
        self.assertEqual(resumes[0]['temperature'], expected)
        self.assertEqual(resumes[0]['round_t0'], ds.T0)
        self.assertEqual(resumes[0]['continuation_base'], 80.)
        self.assertEqual(resumes[0]['current_depth'], 4)
        self.assertEqual(resumes[0]['blocked'], blocked)

    def test_hard_cached_result_is_reused_without_expansion(self):
        events = []
        recorder = TraceRecorder(events.append)
        engine, board = engine_with(recorder)
        ds = engine.deep_search
        node = ds._tree_root
        ds._store_cached_move(node, 123., -1, 0, 1, 1, 1)
        value = ds._continue_cached_move(board, node, 1, [], [], 160., 1.)
        self.assertEqual(value['score'], 1000.)
        self.assertEqual(value['state'], -1)
        self.assertTrue(any(e['kind']=='cache_hit' for e in events))
        self.assertFalse(any(e['kind'] in ('enter','temperature') for e in events))

    def test_copied_subtree_is_imported_not_reexpanded(self):
        events = []
        recorder = TraceRecorder(events.append)
        root = dict(board=bytes(225), children={})
        board = bytearray(225)
        board[112] = 1
        child = dict(board=bytes(board), children={}, main_player=1,
                     move_result=dict(score=42., state=0, complete=True))
        root['children'][(7,7,1)] = child
        recorder.record('cache_copy', root, source=root)
        self.assertEqual(len(recorder.nodes), 2)
        self.assertFalse(any(e['kind']=='enter' for e in events))
        self.assertTrue(any(e['kind']=='node' and e['origin']=='cache' for e in events))

    def test_limits_and_malformed_sequence_are_explicit(self):
        recorder = TraceRecorder(lambda event: None, max_events=1)
        recorder.event('round')
        with self.assertRaises(TraceStopped):
            recorder.event('complete')
        with self.assertRaises(ValueError):
            TraceModel().append(dict(seq=2,kind='round',node=None))

    def test_isolated_worker_pause_resume_and_clean_exit(self):
        context = mp.get_context('spawn')
        commands, messages = context.Queue(), context.Queue()
        running, stopping = context.Event(), context.Event()
        running.set()
        process = context.Process(target=debug_worker,
            args=(commands,messages,running,stopping,MOVES), daemon=True)
        process.start()
        try:
            kind, payload = messages.get(timeout=8)
            self.assertEqual(kind,'ready',payload)
            running.clear()
            commands.put(dict(action='analyze', t0=20., seconds=10., memory_mib=512))
            with self.assertRaises(queue.Empty):
                messages.get(timeout=.15)
            running.set()
            end = time.monotonic()+12
            seen, complete = [], False
            while time.monotonic()<end:
                kind,payload = messages.get(timeout=5)
                self.assertNotIn(kind, ('error','stopped'), payload)
                if kind=='events':
                    seen.extend(payload)
                if kind=='done':
                    complete=True
                    break
            self.assertTrue(complete)
            self.assertTrue(any(e['kind']=='complete' for e in seen))
            json.dumps(seen, allow_nan=False)
            stopping.set()
            running.set()
            process.join(timeout=3)
            self.assertFalse(process.is_alive())
            self.assertEqual(process.exitcode,0)
        finally:
            stopping.set()
            running.set()
            if process.is_alive():
                process.terminate()
            process.join(timeout=2)
            commands.close()
            messages.close()


if __name__ == '__main__':
    unittest.main()
