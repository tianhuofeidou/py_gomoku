"""成对边界提案的小范围反例；不调用教师或实际棋局搜索。"""
import copy
import unittest

from gomoku.tools import pair_fit as P, tune_params as T


def fixture():
    base = T.defaults()
    feature = [0.0]*24+[1.0]
    feature[0] = 1
    variable = dict(r=1,c=2,state=0,constant=0,terms=[[1,feature]])
    score = T.replay_row(variable, base)[0]
    rows = [dict(r=1,c=1,state=0,constant=score+5,terms=[]),variable]
    return dict(position=dict(id='pair',group='g'),tree=dict(rows=rows),
                utilities={'1,1':0.,'1,2':1.}), base


class PairTests(unittest.TestCase):
    def test_crosses_actual_choice_and_preserves_inputs_and_front(self):
        case, base = fixture()
        before = copy.deepcopy((case,base))
        proposals = list(P.pair_proposals(case,base,base))
        self.assertTrue(proposals)
        for proposal in proposals:
            params = proposal['params']
            T.validate(params,base)
            self.assertEqual(T.choose(case['tree']['rows'],params)['c'],2)
            self.assertEqual(proposal['utility_gain'],1.)
            self.assertTrue(all(params[k]==base[k] for k in base if k not in T.LEAF_KEYS))
        self.assertEqual((case,base),before)
        self.assertEqual(proposals,list(P.pair_proposals(case,base,base)))

    def test_does_not_cross_state_tier(self):
        case, base = fixture()
        case['tree']['rows'][0]['state'] = 1
        self.assertEqual(list(P.pair_proposals(case,base,base)),[])

    def test_constant_or_identical_features_cannot_produce_false_success(self):
        case, base = fixture()
        case['tree']['rows'][0]['terms'] = copy.deepcopy(case['tree']['rows'][1]['terms'])
        self.assertEqual(list(P.pair_proposals(case,base,base)),[])

    def test_third_candidate_blocks_target(self):
        case, base = fixture()
        bad = copy.deepcopy(case['tree']['rows'][1])
        bad.update(r=2,c=2,constant=1e9)
        case['tree']['rows'].append(bad)
        case['utilities']['2,2'] = 0.
        self.assertEqual(list(P.pair_proposals(case,base,base)),[])

    def test_no_material_gap_or_already_good(self):
        case, base = fixture()
        case['utilities']['1,2'] = .01
        self.assertEqual(list(P.pair_proposals(case,base,base)),[])
        case['utilities']['1,2'] = -1.
        self.assertEqual(list(P.pair_proposals(case,base,base)),[])

    def test_fails_on_missing_teacher_value(self):
        case, base = fixture()
        del case['utilities']['1,2']
        with self.assertRaises(KeyError):
            list(P.pair_proposals(case,base,base))


if __name__ == '__main__':
    unittest.main()
