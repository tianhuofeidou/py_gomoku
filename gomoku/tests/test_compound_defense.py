"""多必杀：先E交C，否则D；截图反例与独立集合控制。"""
import copy
import unittest
from unittest.mock import patch

from gomoku.core.engine import Engine
from gomoku.core.pattern import PatternAnalyzer
from gomoku.core.search import Search
from gomoku.core import search as search_module
from gomoku.tools.search_trace import TraceRecorder


def rc(text):
    return int(text[1:])-1, ord(text[0])-65


def picture(kind='two43'):
    black = 'H8 J8 M8 I9 L9 M9 J10 K10 I11 N11 M12 L13 H14'
    white = 'I8 K9 H10 L10 J11 K11 L11 M11 J12 K12 L12 I13 J15'
    if kind == 'disjoint':
        black = 'H8 J8 I9 L9 M9 J10 K10 I11 N11 M12 H14'
        white = 'I8 K9 L10 K11 L11 M11 J12 K12 L12 I13 K13'
    board = [[0]*15 for _ in range(15)]
    for player, stones in ((1,black),(2,white)):
        for text in stones.split():
            r,c=rc(text)
            board[r][c]=player
    search=Search(PatternAnalyzer())
    search.rebuild_all(board)
    return search,board


def controls(kind='vcf', counters=()):
    search=Search(PatternAnalyzer())
    board=[[0]*15 for _ in range(15)]
    a,b,common,only_a,only_b=(4,4),(4,7),(6,6),(5,5),(5,8)
    buckets={k:[] for k in ('b_five','w_five','b_vcf','w_vcf','b_d33','w_d33')}
    buckets['w_'+kind]=[a,b]
    groups={a:{common:80.,only_a:90.},b:{common:100.,only_b:80.}}
    for point in counters:
        search.sb[point[0]][point[1]]=5  # 本手形成单冲四，非直接成五。
    return search,board,buckets,groups,common,only_a,only_b


class CompoundDefenseTest(unittest.TestCase):
    def test_empty_common_set_does_not_hide_counter_inside_union(self):
        search,board,buckets,groups,common,only_a,only_b=controls(counters=[(5,5)])
        groups[buckets['w_vcf'][0]]={only_a:90.}
        groups[buckets['w_vcf'][1]]={only_b:80.}
        with patch.object(search,'candidates',return_value=buckets),patch.object(search,'_defense_candidates',side_effect=lambda b,r,c,**kw:groups[(r,c)]):
            self.assertEqual(search.tactical_candidates(board,1),('opp-vcf',{only_a:90.}))

    def test_ordinary_empty_candidates_are_not_declared_losing(self):
        engine=Engine();board=[[0]*15 for _ in range(15)];ds=engine.deep_search
        with patch.object(ds,'_candidate_points',return_value=[]):
            ds._begin_search_cache(board,False)
            result=ds._recursive(board,1,2,1,160.,[],[],{})
        self.assertEqual((result[1],result[5]),(0,0))
        self.assertIsNone(ds._tree_root['frontier'])

    def test_candidate_policy_change_invalidates_existing_tree(self):
        engine=Engine();board=[[0]*15 for _ in range(15)];ds=engine.deep_search
        ds._begin_search_cache(board,False);old=ds._tree_root
        with patch.object(search_module,'DEFENSE_POLICY','different-policy-test'):
            ds._begin_search_cache(board,False)
            self.assertIsNot(ds._tree_root,old)

    def test_no_defense_root_is_legal_losing_terminal_and_reuses_without_expansion(self):
        _,board=picture('disjoint')
        engine=Engine()
        black=[(r,c,1) for r in range(15) for c in range(15) if board[r][c]==1]
        white=[(r,c,2) for r in range(15) for c in range(15) if board[r][c]==2]
        current=[[0]*15 for _ in range(15)]
        for pair in zip(black,white):
            for r,c,p in pair:
                current[r][c]=p;engine.on_move(current,r,c,p)
        before=copy.deepcopy((current,engine.moves,engine.search.sb,engine.search.sw,engine.search.zero_history))
        events=[];engine.deep_search._debug_observer=TraceRecorder(events.append)
        result=engine.analyze_turn(current,1)
        row=result['part4_result']['ranked'][0]
        self.assertEqual(result['part3_candidates']['points'],{})
        self.assertEqual(row['state'],-1)
        r,c=result['part4_result']['move'];self.assertEqual(current[r][c],0)
        root=engine.deep_search._tree_root
        self.assertEqual(root['children'],{})
        self.assertEqual(root['move_result']['state'],1)  # 根的上一落子方是白。
        self.assertTrue(any(e['kind']=='defense_end' and e.get('mover')==2 for e in events))
        self.assertEqual(root['frontier']['leaf_end_reason'],'no_common_defense')
        self.assertEqual(row['score'],-1000.)
        count=len(events)
        second=engine.analyze_turn(current,1)
        self.assertEqual(second['part4_result']['ranked'][0]['state'],-1)
        self.assertEqual(second['part4_result']['move'],result['part4_result']['move'])
        self.assertIs(engine.deep_search._tree_root,root)
        self.assertFalse(any(e['kind'] in ('enter','candidates','resume') for e in events[count:]))
        self.assertEqual((current,engine.moves,engine.search.sb,engine.search.sw,engine.search.zero_history),before)

    def test_recursive_no_defense_is_not_unknown_or_temperature_cutoff(self):
        engine=Engine();board=[[0]*15 for _ in range(15)]
        board[7][7]=2;engine.on_move(board,7,7,2)
        ds=engine.deep_search
        ds.TACTICAL_FREE_PLIES=0
        with patch.object(engine.search,'tactical_candidates',return_value=('opp-vcf',{})):
            ds._begin_search_cache(board,False)
            result=ds._recursive(board,1,2,1,-100.,[],[],{})
        self.assertEqual((result[1],result[5]),(-1,-1))
        self.assertEqual(ds._tree_root['move_result']['state'],1)
        self.assertEqual(ds._tree_root['frontier']['leaf_end_reason'],'no_common_defense')
        self.assertEqual(ds._tree_root['children'],{})

    def test_opponent_no_defense_makes_actual_attacking_node_win(self):
        engine=Engine();board=[[0]*15 for _ in range(15)];ds=engine.deep_search
        with patch.object(engine.search,'tactical_candidates',side_effect=lambda b,p:('opp-d33',{}) if p==2 else (None,{})),patch.object(engine.search,'_ordinary_candidates',return_value=[(7,7,100.),(7,8,80.)]):
            result=ds.rank_candidates(board,{(7,7):100.},1)
        self.assertEqual(result[0]['state'],1)
        child=ds._tree_root['children'][(7,7,1)]
        self.assertEqual(child['move_result']['state'],1)
        self.assertEqual(child['frontier']['leaf_end_reason'],'no_common_defense')
        self.assertEqual(ds.leaf_eval_stats['total'],0)  # 已有硬证明无需再静态评分。
        self.assertEqual(result[0]['score'],1000.)
        self.assertEqual(board,[[0]*15 for _ in range(15)])

    def test_two_four_three_picture_has_two_common_defenses(self):
        search,board=picture()
        before=copy.deepcopy((board,search.sb,search.sw,search._direction_levels))
        reason,points=search.tactical_candidates(board,1)
        self.assertEqual(reason,'opp-vcf')
        self.assertEqual(list(points.items()),[(rc('J13'),100.),(rc('J14'),100.)])
        self.assertEqual((board,search.sb,search.sw,search._direction_levels),before)

    def test_disjoint_picture_returns_empty_defense_not_union(self):
        search,board=picture('disjoint')
        reason,points=search.tactical_candidates(board,1)
        self.assertEqual(reason,'opp-vcf')
        self.assertEqual(points,{})

    def test_union_counter_takes_priority_over_common_defense(self):
        search,board,buckets,groups,common,only_a,only_b=controls(counters=[(5,5)])
        with patch.object(search,'candidates',return_value=buckets),patch.object(search,'_defense_candidates',side_effect=lambda b,r,c,**kw:groups[(r,c)]):
            self.assertEqual(search.tactical_candidates(board,1),('opp-vcf',{only_a:90.}))

    def test_counter_intersection_uses_union_not_only_common_points(self):
        search,board,buckets,groups,common,only_a,only_b=controls(counters=[(5,5),(6,6)])
        with patch.object(search,'candidates',return_value=buckets),patch.object(search,'_defense_candidates',side_effect=lambda b,r,c,**kw:groups[(r,c)]):
            self.assertEqual(search.tactical_candidates(board,1),('opp-vcf',{common:100.,only_a:90.}))

    def test_counter_outside_union_is_not_added(self):
        search,board,buckets,groups,common,only_a,only_b=controls(counters=[(0,0)])
        with patch.object(search,'candidates',return_value=buckets),patch.object(search,'_defense_candidates',side_effect=lambda b,r,c,**kw:groups[(r,c)]):
            self.assertEqual(search.tactical_candidates(board,1),('opp-vcf',{common:100.}))

    def test_double_three_uses_same_rule(self):
        search,board,buckets,groups,common,only_a,only_b=controls('d33',[(5,5)])
        with patch.object(search,'candidates',return_value=buckets),patch.object(search,'_defense_candidates',side_effect=lambda b,r,c,**kw:groups[(r,c)]):
            self.assertEqual(search.tactical_candidates(board,1),('opp-d33',{only_a:90.}))

    def test_single_source_keeps_original_defenses_without_counter(self):
        search,board,buckets,groups,common,only_a,_=controls()
        buckets['w_vcf']=buckets['w_vcf'][:1]
        with patch.object(search,'candidates',return_value=buckets),patch.object(search,'_defense_candidates',side_effect=lambda b,r,c,**kw:groups[(r,c)]):
            self.assertEqual(search.tactical_candidates(board,1),('opp-vcf',{common:80.,only_a:90.}))

    def test_rotation_and_color_swap_keep_the_two_picture_points(self):
        _,board=picture()
        for swap in (False,True):
            rotated=[[0]*15 for _ in range(15)]
            for r in range(15):
                for c in range(15):
                    rotated[c][14-r]=3-board[r][c] if swap and board[r][c] else board[r][c]
            search=Search(PatternAnalyzer());search.rebuild_all(rotated)
            reason,points=search.tactical_candidates(rotated,2 if swap else 1)
            self.assertEqual(reason,'opp-vcf')
            self.assertEqual(set(points),{(c,14-r) for r,c in [rc('J13'),rc('J14')]})


if __name__=='__main__':
    unittest.main()
