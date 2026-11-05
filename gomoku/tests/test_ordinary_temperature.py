"""旗舰普通耗温系数200：数值口径、缓存扣温及身份失效。"""
import unittest
from unittest.mock import patch

from gomoku.core.engine import Engine
from gomoku.core.deep_search import DeepSearch
from gomoku.core.utils import empty_board, BLACK


class OrdinaryTemperatureTests(unittest.TestCase):
    def test_default_coefficient_and_agreed_costs(self):
        self.assertEqual((DeepSearch.DT_M,DeepSearch.DT_POW,DeepSearch.DT_N),(200.,2.,3.))
        for weight,cost in ((1.,199.02),(15.,147.5),(25.,115.5),(50.,53.),(70.,21.),(80.,11.),(100.,3.)):
            with self.subTest(weight=weight):
                self.assertAlmostEqual(DeepSearch._dt(weight),cost)

    def test_ordinary_cost_ignores_tactical_count_and_keeps_exhaustion_rule(self):
        d=Engine().deep_search
        for count in (0,7,8,16,24,100):
            for temperature in (160.,0.,-100.):
                self.assertEqual(d._path_temperature_cost(50.,False,count,temperature),53.)
        with patch.object(d,'_tactical_result',return_value=(None,{})):
            self.assertTrue(d._temperature_exhausted(empty_board(),BLACK,-1.,0))
        self.assertEqual((d.OPENING_COLD_T0,d.MIDDLE_COLD_T0,d.T0),(160.,120.,160.))

    def test_actual_cached_edge_uses_new_cost_without_changing_weight(self):
        d=Engine().deep_search
        board=empty_board()
        d._begin_search_cache(board,False)
        try:
            d._cache_edges(board,d._tree_root,BLACK,[(1.,7,7,1.)],160.,BLACK)
            edge=d._tree_root['edges'][0]
            self.assertEqual(edge[:4],(7,7,BLACK,1.))
            # 空盘该方在此无棋型基础分，纯棋型权重0，耗温203；综合边权仍为1。
            self.assertAlmostEqual(edge[4],-43.)
        finally:
            d._reuse_active=False

    def test_old_coefficient_tree_is_invalidated(self):
        d=Engine().deep_search
        board=empty_board()
        try:
            with patch.object(DeepSearch,'DT_M',400.):
                d._begin_search_cache(board,False)
                old=d._tree_root
                signature=d._tree_signature
            d._begin_search_cache(board,False)
            self.assertIsNot(d._tree_root,old)
            self.assertNotEqual(d._tree_signature,signature)
        finally:
            d._reuse_active=False


if __name__=='__main__':
    unittest.main()
