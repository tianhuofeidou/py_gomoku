"""显式实验温度同时控制建树/续温；未覆盖时保留正式160/120。"""
import unittest
from unittest.mock import patch

from gomoku.core.deep_search import DeepSearch
from gomoku.core.engine import Engine
from gomoku.core.utils import BLACK, empty_board
from gomoku.tools import tune_params as T


class TemperatureEntrypointsTest(unittest.TestCase):
    def cold_temperatures(self):
        engine=Engine()
        seen=[]
        with patch.object(engine.deep_search,'rank_candidates',
                          side_effect=lambda *a,**kw: seen.append(engine.deep_search._initial_temperature()) or []):
            for _ in range(2):
                engine._rank_search(empty_board(),{},BLACK,False)
        return seen

    def test_default_cold_temperatures(self):
        self.assertEqual(self.cold_temperatures(),[160.,120.])

    def test_explicit_offline_temperature_and_restore(self):
        before=tuple(getattr(DeepSearch,n) for n in ('T0','OPENING_COLD_T0','MIDDLE_COLD_T0'))
        for temperature in (1.,40.,80.,240.):
            with self.subTest(temperature=temperature):
                with T.parameter_scope(T.defaults(),temperature):
                    self.assertEqual(self.cold_temperatures(),[temperature]*2)
                    self.assertEqual(DeepSearch.T0,temperature)
                self.assertEqual(tuple(getattr(DeepSearch,n) for n in
                                       ('T0','OPENING_COLD_T0','MIDDLE_COLD_T0')),before)

    def test_exception_restores_cold_temperatures(self):
        before=tuple(getattr(DeepSearch,n) for n in ('T0','OPENING_COLD_T0','MIDDLE_COLD_T0'))
        with self.assertRaisesRegex(RuntimeError,'stop'):
            with T.parameter_scope(T.defaults(),40.):
                self.assertEqual(self.cold_temperatures(),[40.,40.])
                raise RuntimeError('stop')
        self.assertEqual(tuple(getattr(DeepSearch,n) for n in
                               ('T0','OPENING_COLD_T0','MIDDLE_COLD_T0')),before)

    def test_protocol_explicit_temperature(self):
        from gomoku import protocol
        with patch.object(DeepSearch,'T0',160.), patch.object(DeepSearch,'OPENING_COLD_T0',160.), \
             patch.object(DeepSearch,'MIDDLE_COLD_T0',120.):
            with patch.object(protocol,'run',side_effect=lambda:self.cold_temperatures()) as run:
                self.assertEqual(protocol.main(['--t0','40']),[40.,40.])
                run.assert_called_once()

    def test_protocol_default_keeps_split_cold_temperatures(self):
        from gomoku import protocol
        with patch.object(protocol,'run',side_effect=lambda:self.cold_temperatures()):
            self.assertEqual(protocol.main([]),[160.,120.])

    def test_vs_mode_defaults_and_explicit_temperatures(self):
        from gomoku.cli import vs_mode as V
        for argv,expected in (([],(160.,160.,120.)),(['--t0','40'],(40.,40.,40.)),
                              (['--black-params','unused.json'],(40.,40.,40.))):
            with self.subTest(argv=argv):
                engine=Engine();seen=[]
                def analyze(*args,**kwargs):
                    d=engine.deep_search
                    seen.append((d.T0,d.OPENING_COLD_T0,d.MIDDLE_COLD_T0))
                    return {'part4_result':{'move':None}}
                with patch.object(V,'Engine',return_value=engine), \
                     patch.object(engine,'analyze_turn',side_effect=analyze), \
                     patch.object(V.G,'apply_params'), \
                     patch.object(V,'load_params',return_value=dict(V.G.BASELINE,deep_t0=40.)), \
                     patch.object(V.sys,'argv',['vs_mode','--quiet',*argv]),patch('builtins.print'):
                    V.main()
                self.assertEqual(seen,[expected])
