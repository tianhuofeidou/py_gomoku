"""非终局固定落点仍进入搜索；保留预算、缓存与compute_move形态。"""
import copy
import unittest
from unittest.mock import patch

from gomoku.core.engine import Engine
from gomoku.core.opening_book import OpeningBook
from gomoku.core.utils import BLACK, WHITE, empty_board


def prepare(moves=()):
    engine = Engine()
    engine._opening_symmetry = 0
    engine._memory_bonus = lambda: {}  # 不读取本机棋谱经验，唯一候选权重由本用例给定。
    board = empty_board()
    for r,c,p in moves:
        board[r][c] = p
    engine.search.rebuild_all(board)
    # 单元场景只验证建树/换根；不作为性能测量或棋力验收。
    d = engine.deep_search
    d.T0 = d.OPENING_COLD_T0 = d.MIDDLE_COLD_T0 = 12.0
    return engine, board


def snapshot(engine, board):
    s = engine.search
    return copy.deepcopy((board,s.sb,s.sw,s._direction_levels,s.gear,s.gear_progress,
                          s._gear_active,s.zero_history,s.zero_active,engine.moves))


class FixedMoveSearchTests(unittest.TestCase):
    def test_block_five_searches_and_real_two_moves_keep_the_subtree(self):
        for player in (BLACK, WHITE):
            with self.subTest(player=player):
                opp = 3-player
                moves = [(7,c,opp) for c in range(3,7)] + [(7,2,player),(0,0,player),(1,2,player)]
                engine,board = prepare(moves)
                before = snapshot(engine,board)
                result = engine.analyze_turn(board,player)
                self.assertEqual(result['part3_candidates']['reason'],'opp-five')
                self.assertEqual(result['part4_result']['move'],(7,7))
                self.assertEqual(result['part4_result']['type'],'searched')
                self.assertEqual(snapshot(engine,board),before)
                root = engine.deep_search._tree_root
                child = root['children'][(7,7,player)]
                self.assertTrue(child['edges'],'堵五后应有实际对方应手')
                self.assertEqual(root['edges'][0][3],100.)
                r,c,p,*_ = child['edges'][0]
                reply = child['children'][(r,c,p)]
                board[7][7] = player
                engine.on_move(board,7,7,player)
                self.assertIs(engine.deep_search._tree_root,child)
                board[r][c] = p
                engine.on_move(board,r,c,p)
                self.assertIs(engine.deep_search._tree_root,reply)
                before_next = snapshot(engine,board)
                with patch.object(engine,'_rank_search',side_effect=AssertionError('命中旧树不应冷建树')):
                    cached = engine.analyze_turn(board,player)
                self.assertEqual(cached['part4_result']['type'],'searched')
                self.assertEqual(engine.deep_search.last_temperature_kind,'continuation')
                self.assertIs(engine.deep_search._tree_root,reply)
                self.assertEqual(snapshot(engine,board),before_next)

    def test_multiple_opponent_fives_keep_original_first_block(self):
        for player in (BLACK, WHITE):
            engine,board = prepare([(7,c,3-player) for c in range(3,7)])
            first = next(iter(engine.search.tactical_candidates(board,player)[1]))
            result = engine.analyze_turn(board,player)
            self.assertEqual(result['part4_result']['move'],first)
            self.assertEqual(len(engine.deep_search._tree_root['edges']),1)
            self.assertEqual(result['part4_result']['ranked'][0]['state'],-1)

    def test_single_ordinary_keeps_its_real_weight_and_searches(self):
        for player in (BLACK, WHITE):
            engine,board = prepare([(7,7,BLACK),(6,6,WHITE)])
            engine.opening_book = OpeningBook(())
            target = (9,8)
            initial = engine.deep_search._position_key(board)
            ordinary = engine.search._ordinary_candidates
            def candidates(b,player=WHITE,with_score=False):
                if engine.deep_search._position_key(b) == initial:
                    return [(9,8,80.)] if with_score else [target]
                return ordinary(b,player,with_score)
            before = snapshot(engine,board)
            with patch.object(engine.search,'_ordinary_candidates',side_effect=candidates):
                result = engine.analyze_turn(board,player)
            self.assertEqual(result['part4_result']['type'],'searched')
            self.assertEqual(result['part4_result']['move'],target)
            self.assertEqual(engine.deep_search._tree_root['edges'][0][3],80.)
            self.assertIn((*target,player),engine.deep_search._tree_root['children'])
            self.assertEqual(snapshot(engine,board),before)

    def test_single_tactical_dispatch_preserves_weights_and_completeness_option(self):
        for player in (BLACK, WHITE):
            for reason,weight in (('opp-vcf',80.),('my-vcf',100.),('my-d33',100.)):
                with self.subTest(player=player,reason=reason):
                    engine,board = prepare()
                    engine.opening_book = OpeningBook(())
                    point = (5,6)
                    ranked = [engine.deep_search._candidate_result(*point,123.)]
                    with patch.object(engine.search,'tactical_candidates',return_value=(reason,{point:weight})), \
                            patch.object(engine.deep_search,'rank_candidates',return_value=ranked) as rank:
                        result = engine.analyze_turn(board,player,complete_scores=True)
                    rank.assert_called_once_with(board,{point:weight},player,complete_scores=True)
                    self.assertEqual(result['part4_result']['type'],'searched')
                    self.assertEqual(result['part4_result']['move'],point)
                    self.assertEqual(result['part4_result']['ranked'],ranked)

    def test_book_builds_leaf_and_keeps_real_temperature_debt(self):
        engine,board = prepare()
        # 单独检查谱着本身扣出的欠温；默认最低三层另有回归覆盖。
        engine.deep_search.MIN_SEARCH_PLIES = 0
        before = snapshot(engine,board)
        with patch.object(engine.deep_search,'rank_candidates',wraps=engine.deep_search.rank_candidates) as rank:
            result = engine.analyze_turn(board,BLACK)
        self.assertEqual(result['part3_candidates']['reason'],'opening-book')
        self.assertEqual(result['part4_result']['move'],(7,7))
        self.assertEqual(result['part4_result']['type'],'searched')
        rank.assert_called_once_with(board,{(7,7):1.},BLACK)
        child = engine.deep_search._tree_root['children'][(7,7,BLACK)]
        self.assertIsNotNone(child['move_result'])
        self.assertAlmostEqual(child['frontier']['temperature'],12.-engine.deep_search._dt(0.))
        self.assertLess(child['frontier']['temperature'],0.)
        self.assertEqual(snapshot(engine,board),before)
        self.assertNotEqual(result['part4_result']['ranked'][0]['score'],1e9)

    def test_compute_move_keeps_opening_list_but_cannot_bypass_search(self):
        engine,board = prepare()
        self.assertEqual(engine.compute_move(board,BLACK),[(7,7)])
        root = engine.deep_search._tree_root
        self.assertIsNotNone(root)
        child = root['children'][(7,7,BLACK)]
        with patch.object(engine,'_rank_search',side_effect=AssertionError('同盘面应命中旧树')):
            self.assertEqual(engine.compute_move(board,BLACK),[(7,7)])
        self.assertIs(engine.deep_search._tree_root,root)
        self.assertIs(root['children'][(7,7,BLACK)],child)

    def test_actual_five_still_finishes_without_search(self):
        for player in (BLACK, WHITE):
            engine,board = prepare([(7,c,player) for c in range(3,7)] + [(7,2,3-player)])
            before = snapshot(engine,board)
            with patch.object(engine,'_rank_search',side_effect=AssertionError('终局不用继续建树')):
                result = engine.analyze_turn(board,player)
            row = result['part4_result']['ranked'][0]
            self.assertEqual(result['part4_result']['type'],'direct')
            self.assertEqual(result['part4_result']['move'],(7,7))
            self.assertEqual((row['score'],row['state']),(1000.,1))
            self.assertIsNone(engine.deep_search._tree_root)
            self.assertEqual(snapshot(engine,board),before)


if __name__=='__main__':
    unittest.main()
