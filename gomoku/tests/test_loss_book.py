# -*- coding: utf-8 -*-
"""败局回归库的回归测试。

约定：
- 全部使用临时 ``GOMOKU_HOME``，绝不碰用户真实存档；
- 不实例化真引擎（太慢），用最小 mock 实现 ``on_move``；
- 每个用例结束恢复 ``GOMOKU_HOME``。
"""
import json
import os
import shutil
import tempfile
import unittest

from gomoku import loss_book as lb
from gomoku.game import BLACK, WHITE, check_win


class MockEngine:
    """最小引擎替身：只记录调用，不计算。"""

    def __init__(self):
        self.moves = []

    def on_move(self, board, r, c, player):
        self.moves.append((r, c, player))

    def reset(self):
        self.moves = []


def mock_engine_factory():
    return MockEngine()


class LossBookBase(unittest.TestCase):
    def setUp(self):
        self._old = os.environ.get('GOMOKU_HOME')
        self.tmp = tempfile.mkdtemp(prefix='gomoku_lb_')
        os.environ['GOMOKU_HOME'] = self.tmp

    def tearDown(self):
        # 关键：memory 用模块级缓存（_global_memory），engine 会读其中的 bad_lines
        # 来调整候选权重。若不重置，本文件的失败收录会污染后续测试文件。
        try:
            from gomoku import memory as _mem
            _mem._global_memory = None
        except Exception:
            pass
        try:
            from gomoku import game as _game
            _game.games.pop('t', None)
        except Exception:
            pass
        if self._old is None:
            os.environ.pop('GOMOKU_HOME', None)
        else:
            os.environ['GOMOKU_HOME'] = self._old
        shutil.rmtree(self.tmp, ignore_errors=True)

    # 一段合法的 5 手棋谱：黑(7,7) 白(7,8) 黑(8,8) 白(6,6) 黑(9,9)
    MOVES = [
        {'r': 7, 'c': 7, 'player': BLACK},
        {'r': 7, 'c': 8, 'player': WHITE},
        {'r': 8, 'c': 8, 'player': BLACK},
        {'r': 6, 'c': 6, 'player': WHITE},
        {'r': 9, 'c': 9, 'player': BLACK},
    ]

    def store(self):
        return lb.load_store()


# ---------------------------------------------------------------------------
# 1. 黑白双方失败收录
# ---------------------------------------------------------------------------
class TestRecordBothColors(LossBookBase):
    def test_ai_black_loss(self):
        case, is_new = lb.add_case(self.MOVES, BLACK, winner=WHITE,
                                   kind=lb.KIND_LOSS, source='live', store=self.store())
        self.assertTrue(is_new)
        self.assertEqual(case['aiColor'], BLACK)
        self.assertEqual(case['kind'], lb.KIND_LOSS)

    def test_ai_white_loss(self):
        case, is_new = lb.add_case(self.MOVES, WHITE, winner=BLACK,
                                   kind=lb.KIND_LOSS, source='live', store=self.store())
        self.assertTrue(is_new)
        self.assertEqual(case['aiColor'], WHITE)

    def test_different_ai_color_not_merged(self):
        st = self.store()
        lb.add_case(self.MOVES, BLACK, winner=WHITE, store=st)
        lb.add_case(self.MOVES, WHITE, winner=BLACK, store=st)
        st = lb.load_store()
        self.assertEqual(len(st['cases']), 2, '执色不同不能当成同一案例')


# ---------------------------------------------------------------------------
# 2. 去重、重复失败与悔棋撤销
# ---------------------------------------------------------------------------
class TestDedupAndRollback(LossBookBase):
    def test_same_record_merges_but_keeps_sources(self):
        st = self.store()
        c1, new1 = lb.add_case(self.MOVES, BLACK, winner=WHITE, store=st)
        c2, new2 = lb.add_case(self.MOVES, BLACK, winner=WHITE, store=st)
        self.assertTrue(new1)
        self.assertFalse(new2)
        self.assertEqual(len(c2['sources']), 2)
        self.assertEqual(c1['id'], c2['id'])

    def test_mirror_is_deduped(self):
        """旋转/镜像后的同形棋谱应落到同一 recordKey。"""
        mirrored = [{'r': m['r'], 'c': 14 - m['c'], 'player': m['player']} for m in self.MOVES]
        st = self.store()
        lb.add_case(self.MOVES, BLACK, winner=WHITE, store=st)
        _, is_new = lb.add_case(mirrored, BLACK, winner=WHITE, store=st)
        self.assertFalse(is_new, '镜像棋谱应被去重')

    def test_remove_one_source_keeps_other(self):
        st = self.store()
        c1, _ = lb.add_case(self.MOVES, BLACK, winner=WHITE, source='first', store=st)
        first_id = c1['sources'][0]['id']
        c2, _ = lb.add_case(self.MOVES, BLACK, winner=WHITE, source='second', store=st)
        second_id = c2['sources'][1]['id']

        removed = lb.remove_source(first_id)
        self.assertIsNone(removed, '还有来源时不应整案删除')
        st2 = lb.load_store()
        self.assertEqual(len(st2['cases']), 1)
        self.assertEqual(len(st2['cases'][0]['sources']), 1)
        self.assertEqual(st2['cases'][0]['sources'][0]['id'], second_id)

    def test_remove_last_source_drops_case(self):
        st = self.store()
        c, _ = lb.add_case(self.MOVES, BLACK, winner=WHITE, store=st)
        sid = c['sources'][0]['id']
        removed = lb.remove_source(sid)
        self.assertIsNotNone(removed)
        self.assertEqual(len(lb.load_store()['cases']), 0)

    def test_remove_unknown_source_is_noop(self):
        st = self.store()
        lb.add_case(self.MOVES, BLACK, winner=WHITE, store=st)
        self.assertIsNone(lb.remove_source('not-a-real-id'))
        self.assertEqual(len(lb.load_store()['cases']), 1)


# ---------------------------------------------------------------------------
# 3. 疑似败手无法确定
# ---------------------------------------------------------------------------
class TestSuspectUndetermined(LossBookBase):
    def test_empty_moves(self):
        case, _ = lb.add_case([], BLACK, winner=WHITE, store=self.store())
        suspects, why = lb.locate_suspects(case)
        self.assertEqual(suspects, [])
        self.assertEqual(why, 'no-moves')

    def test_ai_never_moved(self):
        moves = [{'r': 7, 'c': 7, 'player': BLACK}]   # 只有黑走，AI 执白
        case, _ = lb.add_case(moves, WHITE, winner=BLACK, store=self.store())
        suspects, why = lb.locate_suspects(case)
        self.assertEqual(suspects, [])
        self.assertEqual(why, 'ai-never-moved')

    def test_manual_ply_accepted(self):
        case, _ = lb.add_case(self.MOVES, BLACK, winner=WHITE, store=self.store())
        c = lb.set_suspects(case, suspects=[], undetermined='no-signal', manual_ply=2)
        self.assertEqual(c['manualPly'], 2)
        starts = lb.retest_starts(c, manual_ply=2)
        self.assertTrue(starts)
        self.assertEqual(starts[0]['back'], 0)

    def test_manual_ply_out_of_range(self):
        case, _ = lb.add_case(self.MOVES, BLACK, winner=WHITE, store=self.store())
        with self.assertRaises(ValueError):
            lb.retest_starts(case, manual_ply=99)


# ---------------------------------------------------------------------------
# 4. 多个附近起点及状态恢复
# ---------------------------------------------------------------------------
class TestRetestStarts(LossBookBase):
    def test_three_starts_back(self):
        case, _ = lb.add_case(self.MOVES, BLACK, winner=WHITE, store=self.store())
        # 人工指定本方第 3 次落子（黑在 moves 里是下标 0,2,4）
        starts = lb.retest_starts(case, manual_ply=3)
        self.assertEqual(len(starts), 3)
        self.assertEqual([s['back'] for s in starts], [0, 1, 2])
        self.assertEqual(starts[0]['startPly'], 5)   # 第 5 手（1 基）
        self.assertEqual(starts[1]['startPly'], 3)
        self.assertEqual(starts[2]['startPly'], 1)

    def test_starts_truncate_when_early(self):
        case, _ = lb.add_case(self.MOVES, BLACK, winner=WHITE, store=self.store())
        starts = lb.retest_starts(case, manual_ply=1)
        self.assertEqual(len(starts), 1, '第一次落子没有更早的本方回合')

    def test_prefix_restore_feeds_engine_every_move(self):
        """前缀必须逐手同步到引擎，不能只摆最后几颗子。"""
        board, engine, err = lb._restore_prefix(self.MOVES, 3, mock_engine_factory)
        self.assertIsNone(err)
        self.assertEqual(len(engine.moves), 3, '前缀 3 手必须调用 3 次 on_move')
        self.assertEqual(engine.moves[0], (7, 7, BLACK))


# ---------------------------------------------------------------------------
# 5. 对手旧谱落点冲突 → diverged
# ---------------------------------------------------------------------------
class TestDiverged(LossBookBase):
    def test_diverged_when_script_point_occupied(self):
        case, _ = lb.add_case(self.MOVES, BLACK, winner=WHITE, store=self.store())
        lb.set_suspects(case, suspects=[], manual_ply=1)
        case = lb.load_store()['cases'][0]

        # AI 执黑，起手就用掉 (7,8)（旧谱里白第 2 手的位置），使对手撞车
        def ai_move(board, moves, player, engine=None):
            return (7, 8), [], False

        r = lb.retest_case(case, manual_ply=1, ai_move_fn=ai_move,
                           engine_factory=mock_engine_factory, max_plies=10)
        self.assertEqual(r['result'], lb.RETEST_DIVERGED)
        self.assertIn('旧败线改变', r['detail'])
        self.assertNotIn(r['result'], lb.PASSING_RETESTS)


# ---------------------------------------------------------------------------
# 6. 棋谱结束但未终局 → script-exhausted
# ---------------------------------------------------------------------------
class TestScriptExhausted(LossBookBase):
    def test_script_end_without_terminal(self):
        case, _ = lb.add_case(self.MOVES, BLACK, winner=WHITE, store=self.store())
        lb.set_suspects(case, suspects=[], manual_ply=3)
        case = lb.load_store()['cases'][0]

        def ai_move(board, moves, player, engine=None):
            # 每步都走空位，不封盘
            for r in range(15):
                for c in range(15):
                    if board[r][c] == 0:
                        return (r, c), [], False
            return None, [], False

        r = lb.retest_case(case, manual_ply=3, ai_move_fn=ai_move,
                           engine_factory=mock_engine_factory, max_plies=40)
        self.assertIn(r['result'], (lb.RETEST_SCRIPT_END, lb.RETEST_DIVERGED))
        self.assertNotIn(r['result'], lb.PASSING_RETESTS)


# ---------------------------------------------------------------------------
# 7. 真正失败 / 真正胜利 / 合法和棋
# ---------------------------------------------------------------------------
class TestOutcomes(LossBookBase):
    def _run(self, ai_move, max_plies=40, moves=None):
        case, _ = lb.add_case(moves or self.MOVES, BLACK, winner=WHITE, store=self.store())
        lb.set_suspects(case, suspects=[], manual_ply=1)
        case = lb.load_store()['cases'][0]
        return lb.retest_case(case, manual_ply=1, ai_move_fn=ai_move,
                              engine_factory=mock_engine_factory, max_plies=max_plies)

    # 更长的原谱：对手有 5 个着法，够 AI 连成五
    LONG = [
        {'r': 7, 'c': 7, 'player': BLACK},
        {'r': 0, 'c': 0, 'player': WHITE},
        {'r': 8, 'c': 8, 'player': BLACK},
        {'r': 0, 'c': 2, 'player': WHITE},
        {'r': 9, 'c': 9, 'player': BLACK},
        {'r': 0, 'c': 4, 'player': WHITE},
        {'r': 10, 'c': 10, 'player': BLACK},
        {'r': 2, 'c': 0, 'player': WHITE},
        {'r': 11, 'c': 11, 'player': BLACK},
        {'r': 2, 'c': 2, 'player': WHITE},
    ]

    def test_real_win_passes(self):
        # AI 执黑，从第 1 手起沿主对角线连成五；对手按 LONG 里的 5 个白子落
        seq = [(7, 7), (8, 8), (9, 9), (10, 10), (11, 11)]
        it = iter(seq)

        def ai_move(board, moves, player, engine=None):
            try:
                return next(it), [], False
            except StopIteration:
                return (14, 0), [], False

        r = self._run(ai_move, moves=self.LONG)
        self.assertEqual(r['result'], lb.RETEST_WIN)
        self.assertIn(r['result'], lb.PASSING_RETESTS)
        self.assertEqual(lb.load_store()['cases'][0]['status'], 'passed')

    def test_real_loss_stays_pending(self):
        # AI 走"间隔"的点，保证不会连成五；对手按旧谱走完 → 必然不通过
        candidates = [(0, 0), (0, 2), (0, 4), (2, 0), (2, 2), (2, 4), (4, 0), (4, 2)]

        def ai_move(board, moves, player, engine=None):
            for r, c in candidates:
                if board[r][c] == 0:
                    return (r, c), [], False
            return None, [], False

        r = self._run(ai_move)
        self.assertNotIn(r['result'], lb.PASSING_RETESTS)
        self.assertEqual(lb.load_store()['cases'][0]['status'], 'pending')

    def test_budget_exhausted_is_not_draw(self):
        def ai_move(board, moves, player, engine=None):
            for r in range(15):
                for c in range(15):
                    if board[r][c] == 0:
                        return (r, c), [], False
            return None, [], False

        r = self._run(ai_move, max_plies=4)
        self.assertEqual(r['result'], lb.RETEST_BUDGET)
        self.assertNotIn(r['result'], lb.PASSING_RETESTS)
        self.assertNotEqual(lb.load_store()['cases'][0]['status'], 'passed')


# ---------------------------------------------------------------------------
# 8. 异常 / 截断不能误判通过
# ---------------------------------------------------------------------------
class TestNoFalsePass(LossBookBase):
    def _case(self):
        case, _ = lb.add_case(self.MOVES, BLACK, winner=WHITE, store=self.store())
        lb.set_suspects(case, suspects=[], manual_ply=1)
        return lb.load_store()['cases'][0]

    def test_ai_exception_reported(self):
        case = self._case()

        def boom(board, moves, player, engine=None):
            raise RuntimeError('引擎炸了')

        r = lb.retest_case(case, manual_ply=1, ai_move_fn=boom,
                           engine_factory=mock_engine_factory, max_plies=10)
        self.assertEqual(r['result'], lb.RETEST_ERROR)
        self.assertIn('ai-move-failed', r['detail'])
        self.assertEqual(lb.load_store()['cases'][0]['status'], 'pending')

    def test_ai_returns_none(self):
        case = self._case()

        def none_move(board, moves, player, engine=None):
            return None, [], False

        r = lb.retest_case(case, manual_ply=1, ai_move_fn=none_move,
                           engine_factory=mock_engine_factory, max_plies=10)
        self.assertEqual(r['result'], lb.RETEST_ERROR)
        self.assertIn('ai-move-empty', r['detail'])

    def test_truncated_case_not_passed(self):
        """只有 half 棋谱、且 AI 从未落子的案例，定位应失败而不是"通过"。"""
        moves = [{'r': 7, 'c': 7, 'player': BLACK}]
        case, _ = lb.add_case(moves, WHITE, winner=BLACK, kind=lb.KIND_ABORTED, store=self.store())
        suspects, why = lb.locate_suspects(case)
        self.assertEqual(suspects, [])
        self.assertEqual(case['status'], 'pending')

    def test_abnormal_kind_kept_separate(self):
        st = self.store()
        lb.add_case(self.MOVES, BLACK, winner=WHITE, kind=lb.KIND_LOSS, store=st)
        lb.add_case(self.MOVES, BLACK, winner=0, kind=lb.KIND_TIMEOUT, store=st)
        kinds = sorted(c['kind'] for c in lb.load_store()['cases'])
        self.assertIn(lb.KIND_LOSS, kinds)
        self.assertIn(lb.KIND_TIMEOUT, kinds)


# ---------------------------------------------------------------------------
# 9. 版本筛选 + 通过后移出归档
# ---------------------------------------------------------------------------
class TestVersionAndArchive(LossBookBase):
    def test_stale_filter(self):
        st = self.store()
        old_fp = {'engine': {'digest': 'aaaaaaaaaaaaaaaa'}, 'book': 'bbbb', 'rule': lb.RULE_ID}
        cur_fp = lb.current_fingerprint()
        lb.add_case(self.MOVES, BLACK, winner=WHITE, fingerprint=old_fp, store=st)
        stale = lb.list_cases(status='pending', fingerprint=cur_fp, store=lb.load_store())
        self.assertEqual(len(stale), 1, '指纹不同的案例应被选为待复测')

        lb2 = lb.add_case(self.MOVES, WHITE, winner=BLACK, fingerprint=cur_fp,
                          store=lb.load_store())[0]
        stale2 = lb.list_cases(status='pending', fingerprint=cur_fp, store=lb.load_store())
        self.assertEqual(len(stale2), 1, '指纹相同的案例不该出现在待复测里')

    def test_archive_moves_passed_only(self):
        st = self.store()
        c1, _ = lb.add_case(self.MOVES, BLACK, winner=WHITE, store=st)
        # 用一局形状不同的棋谱，避免被对称去重合并
        other = [
            {'r': 3, 'c': 3, 'player': BLACK},
            {'r': 3, 'c': 4, 'player': WHITE},
            {'r': 4, 'c': 4, 'player': BLACK},
            {'r': 2, 'c': 2, 'player': WHITE},
        ]
        lb.add_case(other, BLACK, winner=WHITE, store=st)
        self.assertEqual(len(lb.load_store()['cases']), 2)

        st = lb.load_store()
        st['cases'][0]['status'] = 'passed'
        lb.save_store(st)

        moved = lb.archive_passed()
        self.assertEqual(len(moved), 1)
        st2 = lb.load_store()
        self.assertEqual(len(st2['cases']), 1, '未通过的案例应留在待处理')
        self.assertEqual(len(st2['archive']), 1)
        # 归档保留原棋谱
        self.assertTrue(st2['archive'][0]['moves'])
        self.assertEqual(st2['archive'][0]['id'], st['cases'][0]['id'])

    def test_archive_keeps_retest_history(self):
        st = self.store()
        c, _ = lb.add_case(self.MOVES, BLACK, winner=WHITE, store=st)
        st = lb.load_store()
        st['cases'][0].setdefault('retests', []).append(
            {'at': 1, 'startPly': 1, 'result': lb.RETEST_WIN, 'plies': 5})
        st['cases'][0]['status'] = 'passed'
        lb.save_store(st)
        lb.archive_passed()
        arch = lb.load_store()['archive'][0]
        self.assertEqual(len(arch['retests']), 1)


# ---------------------------------------------------------------------------
# 10. 验收回归：一个起点赢、另一个起点输 → 不能归档
# ---------------------------------------------------------------------------
class TestPartialPassNotArchived(LossBookBase):
    def test_one_win_one_loss_stays_pending(self):
        case, _ = lb.add_case(self.MOVES, BLACK, winner=WHITE, store=self.store())
        lb.set_suspects(case, suspects=[], manual_ply=1)
        case = lb.load_store()['cases'][0]

        # 先记一条"赢"
        lb._record_retest(case, {'startPly': 1, 'result': lb.RETEST_WIN, 'plies': 5}, None)
        st = lb.load_store()
        self.assertEqual(st['cases'][0]['status'], 'passed')

        # 再记一条"输"——必须退回 pending
        lb._record_retest(case, {'startPly': 3, 'result': lb.RETEST_LOSS, 'plies': 8}, None)
        st = lb.load_store()
        self.assertEqual(st['cases'][0]['status'], 'pending',
                         '任一复测失败就不能移出待处理库')
        self.assertEqual(len(lb.archive_passed()), 0)

    def test_all_passing_stays_passed(self):
        case, _ = lb.add_case(self.MOVES, BLACK, winner=WHITE, store=self.store())
        lb.set_suspects(case, suspects=[], manual_ply=1)
        case = lb.load_store()['cases'][0]
        lb._record_retest(case, {'startPly': 1, 'result': lb.RETEST_WIN, 'plies': 5}, None)
        lb._record_retest(case, {'startPly': 3, 'result': lb.RETEST_DRAW, 'plies': 40}, None)
        self.assertEqual(lb.load_store()['cases'][0]['status'], 'passed')


# ---------------------------------------------------------------------------
# 11. 验收回归：对战台用 'B'/'W'/'draw' 字符串
# ---------------------------------------------------------------------------
class TestStringWinner(LossBookBase):
    def test_normalize_winner_variants(self):
        self.assertEqual(lb.normalize_winner('B'), BLACK)
        self.assertEqual(lb.normalize_winner('W'), WHITE)
        self.assertEqual(lb.normalize_winner('draw'), 0)
        self.assertEqual(lb.normalize_winner(1), BLACK)
        self.assertEqual(lb.normalize_winner(2), WHITE)
        self.assertEqual(lb.normalize_winner(0), 0)
        self.assertIsNone(lb.normalize_winner(None))
        with self.assertRaises(ValueError):
            lb.normalize_winner('oops')

    def test_add_case_accepts_string_winner(self):
        case, _ = lb.add_case(self.MOVES, BLACK, winner='W', store=self.store())
        self.assertEqual(case['kind'], lb.KIND_LOSS)
        self.assertEqual(case['winner'], WHITE)

    def test_win_by_ai_not_recorded_as_loss(self):
        """AI 执黑，winner='B' → 是 AI 赢，不该记成 loss。"""
        case, _ = lb.add_case(self.MOVES, BLACK, winner='B', store=self.store())
        self.assertEqual(case['kind'], lb.KIND_WIN)

    def test_import_skips_win_and_draw(self):
        """导入工具只收 AI 输的局。"""
        import json, os
        from gomoku.tools import loss_book as cli

        payload = {
            'games': [
                {'game': 1, 'a_white': False, 'winner': 'W', 'moves': self.MOVES},   # AI(黑) 输
                {'game': 2, 'a_white': False, 'winner': 'B', 'moves': self.MOVES},   # AI(黑) 赢
                {'game': 3, 'a_white': False, 'winner': 'draw', 'moves': self.MOVES},
            ]
        }
        fp = os.path.join(self.tmp, 'm.json')
        with open(fp, 'w', encoding='utf-8') as f:
            json.dump(payload, f)
        cli.main(['import', '--json', fp, '--ai-side', 'a'])
        st = lb.load_store()
        self.assertEqual(len(st['cases']), 1, '只应有 1 局 AI 失败被收录')
        self.assertEqual(st['cases'][0]['kind'], lb.KIND_LOSS)


# ---------------------------------------------------------------------------
# 12. 验收回归：满盘和棋
# ---------------------------------------------------------------------------
class TestFullBoardDraw(LossBookBase):
    def test_terminal_returns_zero_when_board_full(self):
        from gomoku.core.utils import empty_board, place
        b = empty_board()
        # 棋盘填满；末手放在 (0,0) 且周围都是白子，
        # 这样 check_win 只看末手附近 → 判 False，从而走到"满盘"分支。
        for r in range(15):
            for c in range(15):
                b[r][c] = WHITE
        b[0][0] = BLACK
        last = {'r': 0, 'c': 0, 'player': BLACK}
        self.assertFalse(check_win(b, 0, 0, BLACK), '末手周围不该构成五连')
        self.assertEqual(lb._terminal(b, last), 0, '满盘无五连应判为合法和棋')

    def test_terminal_none_when_board_not_full(self):
        from gomoku.core.utils import empty_board
        b = empty_board()
        b[0][0] = BLACK
        self.assertIsNone(lb._terminal(b, {'r': 0, 'c': 0, 'player': BLACK}))


# ---------------------------------------------------------------------------
# 13. 验收回归：版本筛选要看"最近一次复测"
# ---------------------------------------------------------------------------
class TestStaleUsesLastRetest(LossBookBase):
    def test_tested_with_current_version_is_excluded(self):
        cur = lb.current_fingerprint()
        # 收录时是旧版本，但已经用当前版本复测过
        old_fp = {'engine': {'digest': 'oldoldoldoldold0'}, 'book': 'oldbook', 'rule': lb.RULE_ID}
        case, _ = lb.add_case(self.MOVES, BLACK, winner=WHITE, fingerprint=old_fp,
                              store=self.store())
        lb._record_retest(case, {'startPly': 1, 'result': lb.RETEST_LOSS,
                                 'plies': 6, 'fingerprint': cur}, None)
        stale = lb.list_cases(status='pending', fingerprint=cur, store=lb.load_store())
        self.assertEqual(len(stale), 0, '已用当前版本测过，不该再入选')

    def test_tested_with_older_version_still_selected(self):
        cur = lb.current_fingerprint()
        older = {'engine': {'digest': 'olderolder00000'}, 'book': 'olderbook', 'rule': lb.RULE_ID}
        case, _ = lb.add_case(self.MOVES, BLACK, winner=WHITE, fingerprint=cur,
                              store=self.store())
        lb._record_retest(case, {'startPly': 1, 'result': lb.RETEST_LOSS,
                                 'plies': 6, 'fingerprint': older}, None)
        stale = lb.list_cases(status='pending', fingerprint=cur, store=lb.load_store())
        self.assertEqual(len(stale), 1, '最近复测是旧版本，应该重新入选')


# ---------------------------------------------------------------------------
# 14. 验收回归：Game 自动收录 + 悔棋撤销
# ---------------------------------------------------------------------------
class TestGameIntegration(LossBookBase):
    def _game(self, human=BLACK):
        from gomoku.game import Game
        g = Game(human_player=human, mode='ai', ai_mode='engine')
        g.register('t')
        return g

    # 黑在 c=0 竖着连成五（黑：第 0,2,4,6,8 手）
    HUMAN_WIN_MOVES = [
        {'r': 0, 'c': 0, 'player': BLACK},
        {'r': 0, 'c': 1, 'player': WHITE},
        {'r': 1, 'c': 0, 'player': BLACK},
        {'r': 1, 'c': 1, 'player': WHITE},
        {'r': 2, 'c': 0, 'player': BLACK},
        {'r': 2, 'c': 1, 'player': WHITE},
        {'r': 3, 'c': 0, 'player': BLACK},
        {'r': 3, 'c': 1, 'player': WHITE},
        {'r': 4, 'c': 0, 'player': BLACK},     # 竖五成立
    ]

    def _force_finish(self, g, moves, winner):
        """直接摆好终局状态再走归档——place() 是轮流落子，构造不出单方连五。"""
        import time as _t
        g.moves = [dict(m) for m in moves]
        g.winner = winner
        g.started_at = int(_t.time() * 1000)
        g.archived = False
        g.solidified = False
        g._archive_finished()
        return g

    def test_live_loss_creates_case(self):
        g = self._game(human=BLACK)
        self._force_finish(g, self.HUMAN_WIN_MOVES, BLACK)   # 人类黑赢 → AI 输
        self.assertEqual(g.winner, BLACK)
        st = lb.load_store()
        self.assertEqual(len(st['cases']), 1, 'AI 失败应自动收录')
        self.assertEqual(st['cases'][0]['kind'], lb.KIND_LOSS)
        self.assertEqual(st['cases'][0]['sources'][0]['source'], 'live')
        self.assertIsNotNone(g.history[-1].get('lossBookSourceId'))

    def test_undo_removes_case(self):
        g = self._game(human=BLACK)
        self._force_finish(g, self.HUMAN_WIN_MOVES, BLACK)
        self.assertEqual(len(lb.load_store()['cases']), 1)
        g.undo_last_move()
        st = lb.load_store()
        self.assertEqual(len(st['cases']), 0, '悔棋撤销终局后应撤销收录')

    def test_ai_win_creates_no_case(self):
        g = self._game(human=WHITE)          # 人类执白，则黑方是 AI
        # 黑（AI）在 c=0 连成五
        self._force_finish(g, self.HUMAN_WIN_MOVES, BLACK)
        self.assertEqual(len(lb.load_store()['cases']), 0, 'AI 赢不该收录')


if __name__ == '__main__':
    unittest.main()