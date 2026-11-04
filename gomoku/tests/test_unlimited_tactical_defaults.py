"""发布默认：160/120冷启动、前8手战术免费；另测旧的全程免费口径。"""
import copy
import unittest
from unittest.mock import patch

from gomoku.core.engine import Engine
from gomoku.core.utils import BLACK,WHITE,empty_board


class UnlimitedTacticalDefaultsTest(unittest.TestCase):
    def test_default_constants_and_costs_at_all_phases(self):
        d=Engine().deep_search
        self.assertEqual((d.OPENING_COLD_T0,d.MIDDLE_COLD_T0,d.T0),(160.,120.,160.))
        self.assertEqual(d.TACTICAL_FREE_PLIES,8)
        self.assertIsNone(d.TACTICAL_EXTENSION_PLIES)
        self.assertEqual(d.TACTICAL_BATCH_SIZE,3)
        for temperature in (-100.,0.):
            for count in (0,7,8,16,24,10000):
                self.assertEqual(d._path_temperature_cost(80.,True,count,temperature),0.)
                self.assertEqual(d._path_temperature_cost(80.,False,count,temperature),d._dt(80.))
        for count, expected in ((0,0.),(7,0.),(8,2.),(16,4.),(24,8.),(10000,8.)):
            self.assertEqual(d._path_temperature_cost(80.,True,count,160.),expected)
            self.assertEqual(d._path_temperature_cost(80.,False,count,160.),d._dt(80.))
        self.assertFalse(d._extension_limit_reached(10000))

    def test_only_current_tactical_identity_can_continue_at_zero(self):
        d=Engine().deep_search; d.TACTICAL_FREE_PLIES=None; board=empty_board()
        for player in (BLACK,WHITE):
            with patch.object(d,'_tactical_result',return_value=('my-vcf',{(1,1):100.})):
                self.assertFalse(d._temperature_exhausted(board,player,0.,10000,extension_count=10000))
            with patch.object(d,'_tactical_result',return_value=(None,{})):
                self.assertTrue(d._temperature_exhausted(board,player,0.,0))
        context={}
        with patch.object(d,'_active_threat_points',side_effect=AssertionError('关闭限额不应统计延伸奖励')):
            d._prepare_extension_step(context,board,BLACK,0.,True,(0,0,0))
            d._finish_extension_step(context,board,BLACK)
        self.assertEqual(context['extension_state'],(0,0,0))

    def test_twelve_zero_cost_moves_can_reach_terminal_both_colors(self):
        for player in (BLACK,WHITE):
            e=Engine(); d=e.deep_search; d.TACTICAL_FREE_PLIES=None; d.T0=0.; board=empty_board()
            points=[(1+2*(i//5),1+2*(i%5)) for i in range(12)]
            steps=[]
            class Observer:
                def record(self,kind,node=None,**data):
                    if kind=='temperature': steps.append(data)
            d._debug_observer=Observer()
            count=lambda b:sum(cell!=0 for row in b for cell in row)
            def tactical(b,p):
                n=count(b)
                return ('my-vcf',{points[n]:100.}) if n<len(points) else (None,{})
            def candidates(b,p,n=5,plies=0):
                return [(*point,w) for point,w in tactical(b,p)[1].items()]
            s=e.search
            before=copy.deepcopy((board,e.moves,s.sb,s.sw,s.zero_history,s.zero_active,s._direction_levels,s.gear))
            with patch.object(d,'_tactical_result',side_effect=tactical), \
                 patch.object(d,'_candidate_points',side_effect=candidates), \
                 patch.object(d,'_check_win',side_effect=lambda b,*args:count(b)==12):
                rows=d.rank_candidates(board,{points[0]:100.},player)
            self.assertEqual(len(steps),12)
            self.assertTrue(all(step['cost']==0. and step['extension_state']==(0,0,0) for step in steps))
            self.assertEqual((rows[0]['state'],rows[0]['score']),(-1,-1000.))
            self.assertEqual((board,e.moves,s.sb,s.sw,s.zero_history,s.zero_active,s._direction_levels,s.gear),before)


if __name__=='__main__':
    unittest.main()
