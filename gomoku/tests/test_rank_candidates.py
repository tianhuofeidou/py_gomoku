# -*- coding: utf-8 -*-
"""候选点打分回归测试（定稿规则：候选>1 必须深推逐个打分、全返回+原始分；
唯一候选不深推、给超大分 1e9）。"""
import unittest

from gomoku.core.utils import empty_board, place, BLACK, WHITE
from gomoku.core.engine import Engine


def replay(e, b, moves):
    for (r, c, p) in moves:
        place(b, r, c, p)
        e.on_move(b, r, c, p)


class RankCandidatesTest(unittest.TestCase):

    def test_single_candidate_direct_huge_score(self):
        """唯一五连点：不深推，ranked 给 1e9"""
        e = Engine()
        b = empty_board()
        # 黑横四 (7,3)-(7,6)，左端 (7,2) 白堵，右端 (7,7) 空 → 唯一五连点 (7,7)
        moves = [(7, 3, BLACK), (6, 6, WHITE), (7, 4, BLACK), (6, 7, WHITE),
                 (7, 5, BLACK), (7, 2, WHITE), (7, 6, BLACK)]
        replay(e, b, moves)
        res = e.analyze_turn(b, BLACK)
        r4 = res['part4_result']
        self.assertEqual(r4['type'], 'direct')
        self.assertEqual(r4['move'], (7, 7))
        self.assertEqual(len(r4['ranked']), 1)
        self.assertEqual(r4['ranked'][0]['score'], 1e9)
        self.assertEqual((r4['ranked'][0]['r'], r4['ranked'][0]['c']), (7, 7))

    def test_multi_candidates_all_ranked_desc(self):
        """普通局面：全部候选深推打分，降序返回，move = 第一名"""
        e = Engine()
        b = empty_board()
        moves = [(7, 7, BLACK), (6, 6, WHITE), (5, 5, BLACK), (6, 8, WHITE),
                 (5, 6, BLACK), (7, 6, WHITE), (4, 6, BLACK), (8, 8, WHITE),
                 (5, 8, BLACK), (6, 7, WHITE)]
        replay(e, b, moves)
        res = e.analyze_turn(b, WHITE)
        r4 = res['part4_result']
        self.assertEqual(r4['type'], 'searched')
        ranked = r4['ranked']
        self.assertTrue(len(ranked) >= 2, '普通局面候选应 ≥2')
        keys = [e.deep_search._pick_key_state(x.get('state'), x['score']) for x in ranked]
        self.assertEqual(keys, sorted(keys, reverse=True), '应按状态档位+候选分降序')
        self.assertEqual(r4['move'], (ranked[0]['r'], ranked[0]['c']))
        for x in ranked:
            self.assertEqual(b[x['r']][x['c']], 0, '候选点应为空位')

    def test_deep_search_old_api_compatible(self):
        """deep_search 旧接口兼容：仍返回最高分点"""
        e = Engine()
        b = empty_board()
        replay(e, b, [(7, 7, BLACK), (6, 6, WHITE), (5, 5, BLACK), (6, 8, WHITE)])
        cands = [(5, 6), (6, 5), (7, 6), (6, 7), (5, 7)]
        pt = e.deep_search.deep_search(b, cands, WHITE)
        ranked = e.deep_search.rank_candidates(b, cands, WHITE)
        self.assertEqual(pt, (ranked[0]['r'], ranked[0]['c']))

    def test_candidates_have_discrete_state_fields(self):
        """候选结果新增 5 态字段；旧 forced/fatal_ratio 仅作兼容，不参与排序。"""
        e = Engine()
        b = empty_board()
        replay(e, b, [(7, 7, BLACK), (6, 6, WHITE), (5, 5, BLACK), (6, 8, WHITE)])
        cands = [(5, 6), (6, 5), (7, 6), (6, 7), (5, 7)]
        ranked = e.deep_search.rank_candidates(b, cands, WHITE)
        self.assertTrue(ranked)
        names = {'必胜', '存在必胜', '无', '存在必败', '必败'}
        for x in ranked:
            self.assertIn('forced', x)
            self.assertIn('fatal_ratio', x)
            self.assertIn(x['state'], (-2, -1, 0, 1, 2))
            self.assertIn(x['state_name'], names)
            self.assertIn(x['child_state'], ('none', 'exists_win', 'exists_lose'))
            self.assertEqual(x['state_name'], e.deep_search._state_name(x['state']))
            self.assertEqual(x['child_state'], e.deep_search._state_child_state(x['state']))


if __name__ == '__main__':
    unittest.main(verbosity=2)
