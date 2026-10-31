"""第二代离线调参调度：统一候选池、完整轮次拟合、隔离多进程与断点。

入口 python -m gomoku.tools.teacher_train run；正式引擎不导入此模块。
教师与搜索进程仍复用 teacher_tune 的适配器和正式引擎。
"""
import argparse
import copy
import hashlib
import math
import os
import threading
from pathlib import Path
import sys
import time
from collections import defaultdict

from gomoku.tools import teacher_tune as U, tune_params as T
from gomoku.tools import leaf_fit as F, training_jobs as J

SCHEMA = 2


def fingerprint():
    return T.digest([U.source_fingerprint(), {p.name: hashlib.sha256(
        p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        for p in (Path(__file__), Path(F.__file__), Path(J.__file__))}])


def search_fingerprint():
    """隔离搜索的依赖身份；优化器变化仍使训练断点失效，但不使相同搜索失效。"""
    return T.digest(['isolated-search-cache-v1',U.source_fingerprint(),
                     sys.version,sys.platform,sys.implementation.name])


class SearchReuse:
    """只读导入已完成规范实验的树/探针；不导入教师标签或串行计时。

    旧版规范实验用完整训练source命名缓存；有基础源码与Python身份时，
    可证明其搜索依赖一致后换键复制。来源缺项正常新算，存在但损坏则失败。
    复用回执随目标条目原子保存，恢复计数不依赖最后一次训练断点是否落盘。
    """
    def __init__(self,out,settings,roots=()):
        self.out,self.settings=Path(out).resolve(),dict(settings)
        self.source=search_fingerprint()
        self.base_source=U.source_fingerprint()
        self._lock=threading.RLock()
        self.stats=dict(trees=0,probes=0)
        self.origins=[]
        for root in dict.fromkeys(str(Path(p).resolve()) for p in roots):
            path=Path(root)
            if self.out.is_relative_to(path):
                raise ValueError('搜索缓存来源必须在目标实验目录之外，原目录保持只读')
            self.origins.append(self._read_origin(path))
        self.descriptors=[o['descriptor'] for o in self.origins]
        if self.origins:
            self._restore_counts()

    def _read_origin(self,path):
        checkpoint=U.read_json(path/'checkpoint.json')
        if not isinstance(checkpoint,dict):
            raise ValueError('搜索缓存来源断点不是对象')
        identity=checkpoint.get('identity',{})
        if not isinstance(identity,dict):
            raise ValueError('搜索缓存来源身份损坏')
        if checkpoint.get('status')!='complete' or identity.get('schema')!=SCHEMA:
            raise ValueError('搜索缓存来源必须是已完成的规范teacher_train实验')
        if identity.get('base_source')!=self.base_source:
            raise ValueError('搜索缓存基础源码/上下文不兼容（旧棋谱记忆环境不可复用）')
        if identity.get('python')!=sys.version:
            raise ValueError('搜索缓存Python身份不兼容')
        settings=identity.get('settings',{})
        # margin与成本接受门槛不影响隔离搜索；原始完整settings仍用于算来源键。
        if not isinstance(settings,dict) or any(k not in settings or settings[k]!=self.settings.get(k)
               for k in ('t0','node_limit','search_seconds')):
            raise ValueError('搜索缓存温度、节点或单局面时间预算不兼容')
        namespace=identity.get('search_source',identity.get('source'))
        if ('search_source' in identity and namespace!=self.source
                or not isinstance(namespace,str) or len(namespace)!=64
                or any(c not in '0123456789abcdef' for c in namespace)):
            raise ValueError('搜索缓存命名空间不兼容或损坏')
        dataset=U.read_json(path/'positions.json')
        if (not isinstance(dataset,dict) or T.digest(dataset)!=identity.get('dataset')
                or not isinstance(dataset.get('positions'),list)):
            raise ValueError('搜索缓存来源局面指纹损坏')
        return dict(path=path,settings=settings,namespace=namespace,
            positions={T.digest(p) for p in dataset['positions']},
            descriptor=dict(path=str(path),identity_sha256=T.digest(identity),cache_source=namespace),
            stamps=self._stamps(path))

    @staticmethod
    def _stamps(path):
        stats=[p.stat() for p in (path/'checkpoint.json',path/'positions.json')]
        return tuple((s.st_size,s.st_mtime_ns) for s in stats)

    def _refresh(self,origin):
        if self._stamps(origin['path'])!=origin['stamps']:
            current=self._read_origin(origin['path'])
            if current['descriptor']!=origin['descriptor']:
                raise ValueError('搜索缓存来源实验身份发生变化')
            origin.update(current)

    @staticmethod
    def _read_entry(path,key):
        data=U.read_json(path)
        if (not isinstance(key,str) or len(key)!=64 or any(c not in '0123456789abcdef' for c in key)
                or not isinstance(data,dict) or data.get('key')!=key or not isinstance(data.get('result'),dict)
                or data.get('payload_sha256')!=T.digest(data['result'])):
            raise ValueError('复用搜索缓存键或内容指纹损坏')
        return data

    @staticmethod
    def _key(source,settings,pos,params,record):
        # 与Experiment.tree的完整位置/有效参数/record键格式一致。
        effective={k:v for k,v in params.items() if not record or k not in T.LEAF_KEYS}
        return T.digest([source,settings,pos,effective,record])

    def _restore_counts(self):
        for folder in self.stats:
            for path in (self.out/folder).glob('*.json'):
                data=U.read_json(path)
                if not isinstance(data,dict):
                    raise ValueError('目标搜索缓存不是对象')
                receipt=data.get('search_cache_reuse')
                if receipt is None:
                    continue
                if not isinstance(receipt,dict):
                    raise ValueError('目标搜索缓存复用回执损坏')
                data=self._read_entry(path,path.stem)
                origin=next((o for o in self.origins if o['descriptor']==receipt.get('origin')),None)
                key=receipt.get('source_key')
                if (origin is None or receipt.get('search_source')!=self.source
                        or not isinstance(key,str) or len(key)!=64
                        or any(c not in '0123456789abcdef' for c in key)):
                    raise ValueError('目标搜索缓存复用回执损坏或来源变化')
                original=self._read_entry(origin['path']/folder/(key+'.json'),key)
                if original['payload_sha256']!=data['payload_sha256']:
                    raise ValueError('搜索缓存复用回执与来源内容不一致')
                self.stats[folder]+=1

    def import_entry(self,pos,params,record):
        if not self.origins:
            return
        folder='trees' if record else 'probes'
        key=self._key(self.source,self.settings,pos,params,record)
        target=self.out/folder/(key+'.json')
        with self._lock:
            # 已有目标条目交回Experiment核验，不能用来源覆盖以隐藏损坏。
            if target.exists():
                return
            position_key=T.digest(pos)
            for origin in self.origins:
                self._refresh(origin)
                if position_key not in origin['positions']:
                    continue
                old_key=self._key(origin['namespace'],origin['settings'],pos,params,record)
                path=origin['path']/folder/(old_key+'.json')
                if not path.exists():
                    continue
                data=self._read_entry(path,old_key)
                U.write_json(target,dict(key=key,result=data['result'],payload_sha256=data['payload_sha256'],
                    search_cache_reuse=dict(origin=origin['descriptor'],source_key=old_key,search_source=self.source)))
                self.stats[folder]+=1
                return


    def check_sources(self):
        with self._lock:
            for origin in self.origins:
                self._refresh(origin)


def quality_reasons(candidate, current, improvement=True):
    """真实选择差距为主目标；排序损失只用于拟合/诊断，不替代选点质量。"""
    reasons = []
    if improvement and candidate['regret'] >= current['regret'] - 1e-9:
        reasons.append('平均教师差距未改善')
    for key, title in (('family_regret', '按棋局家族平均的差距'),
                       ('selection_regret', '训练内留出组差距'),
                       ('worst_fold_regret', '最差训练分折差距'),
                       ('tail_regret', '最差十分之一局面的平均差距'),
                       ('severe', '严重误选数')):
        if candidate[key] > current[key] + 1e-9:
            reasons.append(title + '增加')
    return reasons


def keep_candidate(checkpoint, proposal, metrics, previous):
    """优化器可以继续探索；对外保留的始终是通过约束的最好快照。"""
    reasons = quality_reasons(metrics, previous)
    if not reasons:
        checkpoint['params'] = dict(proposal)
    return reasons


class Corpus:
    """共享候选池只增长；同次比较的原参数、当前参数和试参共用参考值。"""
    def __init__(self, out, positions, teacher, settings, workers, deadline, reuse=(),reuse_search=()):
        self.out, self.positions, self.settings = Path(out), positions, settings
        self.deadline, self.source = deadline, fingerprint()
        self.jobs = J.Jobs(workers)
        self.labels = J.Labels(teacher, self.jobs, reuse)
        self.search_reuse=(reuse_search if isinstance(reuse_search,SearchReuse)
                           else SearchReuse(out,settings,reuse_search))
        if (self.search_reuse.out!=Path(out).resolve() or self.search_reuse.settings!=dict(settings)
                or self.search_reuse.source!=search_fingerprint()):
            raise ValueError('搜索缓存复用对象不属于当前实验/设置/源码')
        self.search_source=self.search_reuse.source
        self.cache = U.Experiment(out, settings, positions, teacher, self.search_source, deadline)
        self._memory, self._utilities, self._roots = {}, {}, {}
        disk_tree = self.cache.tree

        def cached_tree(pos, params, record=True):
            effective = {k:v for k,v in params.items() if not record or k not in T.LEAF_KEYS}
            key = T.digest([pos,effective,record])
            if key not in self._memory:
                self.search_reuse.import_entry(pos,params,record)
                self._memory[key] = disk_tree(pos,params,record)
            return self._memory[key]

        self.cache.tree = cached_tree
        self._teacher_stats = {name: (teacher.exe.parent/name).stat()
                               for name in teacher.identity.get('files',{})}
        self.pool_path = self.out / 'pool.json'
        if self.pool_path.exists():
            envelope = U.read_json(self.pool_path)
            if envelope.get('sha256') != T.digest(envelope['positions']):
                raise ValueError('公共候选池指纹损坏')
            self.pool = envelope['positions']
        else:
            self.pool = {}
        self.fold = {}
        for split in ('train', 'validation', 'test'):
            groups = sorted({p['group'] for p in positions if p['split'] == split})
            self.fold.update({(split, g): i % min(3, len(groups)) for i, g in enumerate(groups)})

    def selected(self, split):
        return [p for p in self.positions if p['split'] == split]

    def check_deadline(self):
        if time.monotonic() >= self.deadline:
            raise U.BudgetStop('达到本轮总时长预算')
        if fingerprint() != self.source:
            raise ValueError('训练过程中源码发生变化，停止以避免混合版本缓存')
        self.search_reuse.check_sources()
        for name, previous in self._teacher_stats.items():
            path = self.labels.teacher.exe.parent/name
            current = path.stat()
            if (current.st_size,current.st_mtime_ns)!=(previous.st_size,previous.st_mtime_ns):
                if hashlib.sha256(path.read_bytes()).hexdigest()!=self.labels.teacher.identity['files'][name]:
                    raise ValueError('训练过程中教师程序/配置/权重发生变化')
                self._teacher_stats[name]=current

    def trees(self, params, split, record=True):
        self.check_deadline()
        positions = self.selected(split)
        return self.jobs.map(lambda p: self.cache.tree(p, params, record), positions)

    def prepare(self, parameters, split):
        """先准备全部结构，再一次性标注共同候选，避免给新参数换尺子。"""
        self.check_deadline()
        positions = self.selected(split)
        working_pool = copy.deepcopy(self.pool)
        unique = {}
        for params in parameters:
            key = T.digest({k: v for k, v in params.items() if k not in T.LEAF_KEYS})
            unique[key] = params
        for params in unique.values():
            for pos, tree in zip(positions, self.trees(params, split)):
                points = set(map(tuple, working_pool.get(pos['id'], [])))
                points.update((r['r'], r['c']) for r in tree['rows'])
                working_pool[pos['id']] = [list(point) for point in sorted(points)]
        self.labels.prefetch(p['moves'] for p in positions)
        for pos in positions:
            points = set(map(tuple, working_pool[pos['id']]))
            root=self.labels.read(pos['moves'])['move']
            self._roots[pos['id']]=root
            points.add(tuple(root))
            working_pool[pos['id']] = [list(point) for point in sorted(points)]
        histories = []
        for pos in positions:
            for point in working_pool[pos['id']]:
                history = self.labels.child_history(pos, point)
                if history is not None:
                    histories.append(history)
        self.labels.prefetch(histories)
        self.pool = working_pool
        U.write_json(self.pool_path, dict(positions=self.pool, sha256=T.digest(self.pool)))
        print(f'{split}: 公共候选池就绪，{len(positions)}局面，'
              f'{sum(len(self.pool[p["id"]]) for p in positions)}候选', flush=True)

    def cases(self, params, split):
        result = []
        for pos in self.selected(split):
            tree = self.cache.tree(pos, params)
            utilities=self._utilities.setdefault(pos['id'],{})
            for r,c in self.pool[pos['id']]:
                key=f'{r},{c}'
                if key not in utilities:
                    utilities[key]=self.labels.utility(pos,[r,c])
            result.append(dict(position=pos, tree=tree, utilities=utilities,
                               reference=max(utilities.values()),
                               teacher_move=self._roots[pos['id']]))
        return result

    def fit_cases(self, params):
        cases = self.cases(params, 'train')
        if len({case['position']['group'] for case in cases}) < 3:
            raise ValueError('新版拟合至少需要三个训练棋局家族，以保留训练内选优组')
        # 固定保留 fold 0 只做选优；不能让每个 epoch 都在拟合样本上自证。
        return [case for case in cases if self.fold[('train',case['position']['group'])] != 0]

    def metrics(self, params, split, baseline):
        cases, anchors = self.cases(params, split), self.cases(baseline, split)
        regrets, attainable, losses, families, folds = [], [], [], defaultdict(list), defaultdict(list)
        severe = new_bad = absent = blocked = differing = 0
        for case, anchor in zip(cases, anchors):
            rows, utilities = case['tree']['rows'], case['utilities']
            chosen = T.choose(rows, params)
            value = utilities[f'{chosen["r"]},{chosen["c"]}']
            anchor_row = T.choose(anchor['tree']['rows'], baseline)
            anchor_value = utilities[f'{anchor_row["r"]},{anchor_row["c"]}']
            regret = max(0.0, case['reference'] - value)
            regrets.append(regret)
            severe += regret >= .5
            new_bad += (case['reference'] - anchor_value <= self.settings['margin']
                        and anchor_value - value > self.settings['margin'])
            families[case['position']['group']].append(regret)
            folds[self.fold[(split, case['position']['group'])]].append(regret)
            points = {(r['r'], r['c']) for r in rows}
            best_points = {tuple(map(int, key.split(','))) for key, v in utilities.items()
                           if math.isclose(v, case['reference'], abs_tol=1e-12)}
            absent += not bool(best_points & points)
            tier = max(T.DeepSearch.STATE_ORDER[r['state']] for r in rows)
            top = [r for r in rows if T.DeepSearch.STATE_ORDER[r['state']] == tier]
            best_value = max(utilities[f'{r["r"]},{r["c"]}'] for r in top)
            attainable.append(max(0.0, best_value-value))
            blocked += bool(best_points & points) and not any((r['r'], r['c']) in best_points for r in top)
            differing += [chosen['r'],chosen['c']] != case['teacher_move']
            best = max(top, key=lambda r: utilities[f'{r["r"]},{r["c"]}'])
            best_score = T.replay_row(best, params)[0]
            pair_losses = []
            for row in top:
                gap = best_value - utilities[f'{row["r"]},{row["c"]}']
                if gap > self.settings['margin']:
                    z = (T.replay_row(row, params)[0] - best_score)/500
                    pair_losses.append(gap*(max(z,0)+math.log1p(math.exp(-abs(z)))))
            losses.append(sum(pair_losses)/len(pair_losses) if pair_losses else 0.0)
        if not regrets:
            raise ValueError('数据分组无可评估局面')
        avg = lambda xs: sum(xs)/len(xs)
        fold_values = {str(k): avg(v) for k,v in sorted(folds.items())}
        return dict(positions=len(regrets), regret=avg(regrets), ranking_loss=avg(losses),
                    family_regret=avg([avg(v) for v in families.values()]),
                    selection_regret=fold_values['0'] if split=='train' else avg(regrets),
                    fold_regret=fold_values, worst_fold_regret=max(fold_values.values()),
                    tail_regret=avg(sorted(regrets,reverse=True)[:max(1,math.ceil(len(regrets)*.1))]),
                    severe=severe, new_bad=new_bad, attainable_regret=avg(attainable),
                    best_move_absent=absent, state_blocked=blocked, different_coordinate=differing,
                    pool_sha256=T.digest({p['id']:self.pool[p['id']] for p in self.selected(split)}))

    def verify(self, params, split):
        """每次拟合阶段选优后，用实际早停搜索核对，不能仅信缓存表达式。"""
        probes = self.trees(params, split, record=False)
        for case, probe in zip(self.cases(params, split), probes):
            selected = T.choose(case['tree']['rows'], params)
            if probe['move'] != [selected['r'],selected['c']]:
                raise AssertionError('固定树选点与正式早停搜索不一致')
            row = next(r for r in probe['rows'] if [r['r'],r['c']]==probe['move'])
            if (row['state'] != selected['state'] or not math.isclose(
                    row['score'], T.replay_row(selected,params)[0], rel_tol=1e-10,abs_tol=1e-7)):
                raise AssertionError('固定树分数/五态与正式搜索不一致')
        return dict(nodes=sum(p['nodes'] for p in probes),
                    parallel_search_seconds=sum(p['seconds'] for p in probes))

    def compute_guard(self, params, baseline_nodes):
        total = 0
        positions = self.selected('train')
        for start in range(0,len(positions),self.jobs.workers):
            batch = positions[start:start+self.jobs.workers]
            try:
                probes = self.jobs.map(lambda p:self.cache.tree(p,params,False),batch)
            except T.SearchLimit:
                if time.monotonic() >= self.deadline:
                    raise U.BudgetStop('试参搜索达到总时长预算') from None
                return False,total
            total += sum(p['nodes'] for p in probes)
            if total > max(1,baseline_nodes)*self.settings['cost_ratio']:
                return False,total
        return True,total

    def calibrate(self, params, split):
        """并行工作已结束后串行计时；不把争用时长当成参数固有性能。"""
        measurements = []
        for pos in self.selected(split):
            self.check_deadline()
            key = T.digest([self.search_source,self.settings,pos,params,'serial-calibration-v1'])
            path = self.out/'calibration'/(key+'.json')
            if path.exists():
                item = U.read_json(path)
                if item['sha256'] != T.digest(item['result']):
                    raise ValueError('串行计时缓存损坏')
                result = item['result']
            else:
                result = U.isolated_search(pos,params,self.settings,False,self.deadline)
                U.write_json(path,dict(result=result,sha256=T.digest(result)))
            chosen = T.choose(self.cache.tree(pos,params)['rows'],params)
            if result['move'] != [chosen['r'],chosen['c']]:
                raise AssertionError('串行验收选点与训练缓存不一致')
            measurements.append(result)
        times = sorted(r['seconds'] for r in measurements)
        return dict(positions=len(times),nodes=sum(r['nodes'] for r in measurements),
                    seconds=sum(times),p95_seconds=times[math.ceil(.95*len(times))-1])


def save(out, checkpoint):
    U.write_json(Path(out)/'checkpoint.json',checkpoint)


def score_axes(value):
    keys=[key.strip() for key in value.split(',')]
    if not keys or len(set(keys))!=len(keys) or any(key not in T.GROUPS['score'] for key in keys):
        raise argparse.ArgumentTypeError('score-keys须为不重复的候选评分参数名')
    return sorted(keys)


def coordinate_phase(corpus,checkpoint,phase,args,out,baseline,group='leaf'):
    """叶子复用树，前段重建树；公共池同步比较，每提案断点且不放宽约束。"""
    state=checkpoint.get('coordinate')
    if state is None:
        state=dict(phase=phase,group=group,key_index=0,proposal_index=0,origin=None)
        checkpoint['coordinate']=state
    if state['phase']!=phase or state['group']!=group:
        raise ValueError('坐标搜索断点阶段不一致')
    keys=F.coordinate_order(args.seed+1009*(phase+1),group)
    if group=='score' and getattr(args,'score_keys',None):
        keys=[key for key in keys if key in args.score_keys]
    while state['key_index']<len(keys):
        key=keys[state['key_index']]
        if state['origin'] is None:
            state['origin']=dict(checkpoint['params'])
            state['proposal_index']=0
            save(out,checkpoint)
        proposals=list(F.coordinate_proposals(key,state['origin'],baseline,group))
        before=corpus.metrics(checkpoint['params'],'train',baseline) if group=='leaf' else None
        for index in range(state['proposal_index'],len(proposals)):
            corpus.check_deadline()
            proposal=proposals[index]
            T.validate(proposal['params'],baseline)
            metrics,nodes=None,None
            if group=='leaf':
                metrics=corpus.metrics(proposal['params'],'train',baseline)
                reasons=keep_candidate(checkpoint,proposal['params'],metrics,before)
            else:
                before=None
                passed,nodes=corpus.compute_guard(proposal['params'],checkpoint['baseline_train_compute']['nodes'])
                reasons=['正式搜索节点超过同集基线限额，未继续教师标注']
                if passed:
                    try:
                        corpus.prepare([baseline,checkpoint['initial'],checkpoint['params'],proposal['params']],'train')
                        # 新前段可能增补公共候选；必须在同一新参考下重新评价当前最好参数。
                        before=corpus.metrics(checkpoint['params'],'train',baseline)
                        metrics=corpus.metrics(proposal['params'],'train',baseline)
                        reasons=quality_reasons(metrics,before)
                        if not reasons:
                            corpus.verify(proposal['params'],'train')
                            keep_candidate(checkpoint,proposal['params'],metrics,before)
                    except T.SearchLimit as exc:
                        if time.monotonic()>=corpus.deadline:
                            raise U.BudgetStop('坐标试参达到总时长预算') from None
                        before=metrics=None
                        reasons=[str(exc)]
            checkpoint['history'].append(dict(phase=phase,epoch=state['key_index']+1,
                group=group+'-coordinate',accepted=not reasons,reasons=reasons,nodes=nodes,
                coordinate_key=key,direction=proposal['direction'],relative_step=proposal['relative_step'],
                metrics=metrics,previous=before))
            if not reasons:
                before=metrics
                print(f'坐标 {key} {proposal["direction"]:+}×{proposal["relative_step"]} '
                      f'差距={metrics["regret"]:.6f} 接受=True',flush=True)
            state['proposal_index']=index+1
            save(out,checkpoint)
        state.update(key_index=state['key_index']+1,proposal_index=0,origin=None)
        save(out,checkpoint)
    checkpoint['candidate_train_compute']=corpus.verify(checkpoint['params'],'train')
    passed,_=corpus.compute_guard(checkpoint['params'],checkpoint['baseline_train_compute']['nodes'])
    if not passed:
        raise AssertionError('坐标搜索选优后实际搜索节点超过限额')
    checkpoint['coordinate']=None


def boundary_phase(corpus,checkpoint,phase,args,out,baseline):
    """只用拟合折的选点签名细化；完整训练质量门槛负责接受，外部验证不参与。"""
    previous=checkpoint.get('boundary')
    if previous is not None and previous.get('phase')!=phase:
        raise ValueError('分界断点阶段不一致')
    origin=previous['origin'] if previous is not None else dict(checkpoint['params'])
    fitter=F.BoundaryFit(corpus.fit_cases(origin),origin,baseline,
        seed=args.seed+1009*(phase+1),budget=args.leaf_evaluations,state=previous)
    fitter.state['phase']=phase
    checkpoint['boundary']=fitter.state
    if fitter.state.get('finished'):
        return
    selection=[case for case in corpus.cases(origin,'train')
               if corpus.fold[('train',case['position']['group'])]==0]
    selection_ranking=F.CachedRanking(selection)
    original_selection=selection_ranking.signature(origin)
    before=corpus.metrics(checkpoint['params'],'train',baseline)
    while True:
        corpus.check_deadline()
        proposal=fitter.next_proposal()
        if proposal is None:
            break
        # 评分前落盘预留，进程中断后保守计费；pending保证不漏算或重复记录接受。
        fitter.reserve()
        save(out,checkpoint)
        params=proposal['params']
        T.validate(params,baseline)
        metrics=corpus.metrics(params,'train',baseline)
        signature=fitter.ranking.signature(params)
        reasons=keep_candidate(checkpoint,params,metrics,before)
        fitter.observe(signature)
        event=dict(phase=phase,epoch=fitter.state['completed'],group='leaf-boundary',
            accepted=not reasons,reasons=reasons,metrics=metrics,previous=before,
            coordinate_key=proposal['key'],direction=proposal['direction'],
            relative_step=proposal['relative_step'],kind=proposal['kind'],
            evaluation_attempt=fitter.state['attempts'],
            changed_fields=[k for k in T.LEAF_KEYS if not math.isclose(params[k],origin[k],rel_tol=1e-10,abs_tol=1e-8)])
        if not reasons:
            selected=selection_ranking.signature(params)
            event['selection_choice_changes_from_origin']=sum(a!=b for a,b in zip(original_selection,selected))
            event['selection_positive_evidence']=metrics['selection_regret']<before['selection_regret']-1e-9
            before=metrics
            print(f'分界 {proposal["key"]} {proposal["kind"]} 差距={metrics["regret"]:.6f} 接受=True',flush=True)
        checkpoint['history'].append(event)
        save(out,checkpoint)
    checkpoint['candidate_train_compute']=corpus.verify(checkpoint['params'],'train')
    passed,_=corpus.compute_guard(checkpoint['params'],checkpoint['baseline_train_compute']['nodes'])
    if not passed:
        raise AssertionError('分界阶段选优后实际搜索节点超过限额')
    summary=dict(phase=phase,attempts=fitter.state['attempts'],completed=fitter.state['completed'],
        duplicates=fitter.state['duplicates'],stop_reason=fitter.state['stop_reason'],
        pending_unfinished=fitter.state['pending'] is not None,
        axes_observed=sum(len(v)>1 for v in fitter.state['samples'].values()))
    runs=checkpoint.setdefault('boundary_runs',[])
    runs[:]=[r for r in runs if r['phase']!=phase]+[summary]
    fitter.state['finished']=True
    # 保留finished断点直到主调度原子提交phase+1，防止完成后中断却重跑整阶段。
    save(out,checkpoint)


def write_report(out, checkpoint):
    lines=['# 分层教师调参报告','',f'状态：{checkpoint["status"]}',
           f'已完成阶段：{checkpoint.get("phase",0)}/{len(checkpoint["schedule"])}',
           f'建议采用：{checkpoint.get("recommended",False)}','',
           '参考值为公共候选池中教师评分的最大值；不是教师根节点首选的绝对正确性。',
           '优化只比较最高五态档内的候选；候选遗漏、五态压制、同档排错分别统计。',
           '公共池变化时基线与当前参数同步重算；历史行保留当时的比较条件。','',
           '训练家族分三折，固定第0折只做中间选优，不参与叶子梯度或分界生成；外部验证只在最后查询。','',
           '| 数据/参数 | 教师差距 | 家族均值 | 尾部差距 | 严重误选 | 新增退步 | 同档可优化差距 | 最优未入候选 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for key in ('baseline_train','initial_train','candidate_train','baseline_validation','initial_validation','candidate_validation'):
        if key in checkpoint:
            m=checkpoint[key]
            lines.append(f'| {key} | {m["regret"]:.6f} | {m["family_regret"]:.6f} | '
                         f'{m["tail_regret"]:.6f} | {m["severe"]} | {m["new_bad"]} | '
                         f'{m["attainable_regret"]:.6f} | {m["best_move_absent"]} |')
    lines+=['','| 阶段/epoch | 组别 | 接受 | 原因 |','|---|---|---|---|']
    for e in checkpoint['history']:
        lines.append(f'| {e["phase"]+1}/{e.get("epoch","—")} | {e["group"]} | '
                     f'{e["accepted"]} | {"；".join(e["reasons"]) or "质量约束全部通过"} |')
    for run in checkpoint.get('boundary_runs',[]):
        lines.append(f'分界阶段{run["phase"]+1}：评估预留{run["attempts"]}次，完成{run["completed"]}次，'
                     f'观察{run["axes_observed"]}轴；停止={run["stop_reason"]}，未完成提案={run["pending_unfinished"]}。')
    accepted=[h for h in checkpoint['history'] if h.get('group')=='leaf-boundary' and h['accepted']]
    if accepted and all(not h.get('selection_choice_changes_from_origin') for h in accepted):
        lines.append('分界接受更新未改变训练内留出折的选点；不退步不提供正向泛化证据。')
    lines+=['',f'验证用途：{checkpoint["identity"].get("validation_role","independent")}（seen表示已见开发回归，不能当成独立未见验收）。',
            f'搜索缓存复用：{checkpoint.get("search_cache_reuse",dict(trees=0,probes=0))}；'
            '仅复用相同搜索的树/探针，不导入其他实验的串行计时。',
            f'相对本轮初始参数同时改善训练/验证：{checkpoint.get("improved_initial",False)}',
            f'停止/拒绝原因：{checkpoint.get("stop_reason",checkpoint.get("final_reasons",[]))}',
            '未查询保留测试集，不自动修改正式默认参数；离线通过仍需独立配对对战。',
            '并行阶段只用节点数约束成本；最终另做串行搜索计时，时间仍受本机环境影响。','']
    (Path(out)/'report.md').write_text('\n'.join(lines),encoding='utf-8')


def dataset_for(args):
    dataset = U.read_json(args.positions_file) if args.positions_file else U.prepare(
        args.games,args.positions,args.seed,args.min_ply,args.max_ply,args.stride)
    if args.split_from:
        old = U.read_json(args.split_from)
        if old['sources'] != dataset['sources']:
            raise ValueError('继承分组的输入棋谱不同')
        groups = {p['group']:p['split'] for p in old['positions']}
        for pos in dataset['positions']:
            if pos['group'] in groups:
                pos['split'] = groups[pos['group']]
        old_ids={p['id']:p['split'] for p in old['positions']}
        if any(p['id'] in old_ids and p['split'] != old_ids[p['id']] for p in dataset['positions']):
            raise ValueError('旧保留局面分组发生冲突')
        dataset['inherited_splits_sha256']=T.digest(old)
    U.validate_positions(dataset['positions'])
    return dataset


def run(args):
    if sys.version_info[:2] != (3,14):
        raise ValueError('实验必须使用 Python 3.14')
    out,repo=Path(args.out).resolve(),U.ROOT.parent
    if out.is_relative_to(repo) and not out.is_relative_to(repo/'storages'/'experiments'):
        raise ValueError('项目内产物只能放 storages/experiments/')
    out.mkdir(parents=True,exist_ok=True)
    with U.RunLock(out/'run.lock'):
        return _run(args,out)


def _run(args,out):
    deadline=time.monotonic()+args.max_seconds
    dataset=dataset_for(args)
    teacher=U.RapfiTeacher(args.teacher,out,args.teacher_nodes,args.teacher_ms,deadline,args.max_queries)
    settings=dict(t0=args.t0,node_limit=args.node_limit,search_seconds=args.search_seconds,
                  margin=args.margin,cost_ratio=args.cost_ratio,time_ratio=args.time_ratio)
    groups=args.groups.split(',')
    if groups not in (['leaf'],['score'],['leaf','score']):
        raise ValueError('新版训练支持 leaf、score 或 leaf,score；gear/search 保持固定')
    if args.leaf_policy=='boundary' and 'leaf' not in groups:
        raise ValueError('boundary策略只适用于包含leaf的实验')
    if args.score_keys and (args.score_policy!='coordinate' or 'score' not in groups):
        raise ValueError('score-keys只适用于包含score组的前段坐标搜索')
    baseline=T.defaults()
    initial=dict(baseline)
    if args.initial_params:
        initial_data=U.read_json(args.initial_params)
        if initial_data.get('source')!=U.source_fingerprint():
            raise ValueError('初始参数与当前正式引擎源码不匹配')
        if args.validation_role!='seen':
            raise ValueError('使用既有评审参数时须明确 --validation-role seen')
        initial=initial_data['params']
        T.validate(initial,baseline)
    search_reuse=SearchReuse(out,settings,args.reuse_search_cache)
    identity=dict(schema=SCHEMA,source=fingerprint(),base_source=U.source_fingerprint(),
                  search_source=search_reuse.source,reuse_search_cache=search_reuse.descriptors,
                  dataset=T.digest(dataset),teacher=teacher.identity,settings=settings,
                  groups=groups,rounds=args.rounds,trials=args.trials,epochs=args.epochs,
                  batch_size=args.batch_size,seed=args.seed,workers=args.workers,python=sys.version,
                  initial=initial,leaf_policy=args.leaf_policy,leaf_rate=args.leaf_rate,
                  leaf_evaluations=args.leaf_evaluations,
                  score_policy=args.score_policy,
                  score_keys=args.score_keys,
                  validation_role=args.validation_role)
    path=out/'checkpoint.json'
    if path.exists():
        if not args.resume:
            raise ValueError('目录已有实验，请使用 --resume 或新目录')
        checkpoint=U.read_json(path)
        if checkpoint['identity'] != identity:
            raise ValueError('源码、数据、教师或训练配置已变化，拒绝续跑')
        if checkpoint['status']=='complete':
            print('实验已完成',flush=True)
            return 0
    else:
        if args.resume:
            raise ValueError('没有可续跑断点')
        schedule=[]
        for _ in range(args.rounds):
            if 'leaf' in groups:
                schedule.append('leaf')
            if 'score' in groups:
                schedule.extend(['score']*(1 if args.score_policy=='coordinate' else args.trials))
                if 'leaf' in groups:
                    schedule.append('leaf')
        checkpoint=dict(identity=identity,baseline=baseline,initial=initial,params=dict(initial),schedule=schedule,
                        phase=0,optimizer=None,status='running',history=[])
        U.write_json(out/'positions.json',dataset)
        save(out,checkpoint)
    checkpoint['search_cache_reuse']=search_reuse.stats
    corpus=Corpus(out,dataset['positions'],teacher,settings,args.workers,deadline,args.reuse_labels,search_reuse)
    baseline=checkpoint['baseline']
    started=time.monotonic()
    try:
        checkpoint['status']='running'
        checkpoint.pop('stop_reason',None)
        corpus.prepare([baseline,initial,checkpoint['params']],'train')
        base_compute=corpus.verify(baseline,'train')
        checkpoint['baseline_train_compute']=base_compute
        for phase in range(checkpoint['phase'],len(checkpoint['schedule'])):
            corpus.check_deadline()
            group=checkpoint['schedule'][phase]
            print(f'阶段 {phase+1}/{len(checkpoint["schedule"])} {group}',flush=True)
            current=checkpoint['params']
            corpus.prepare([baseline,initial,current],'train')
            if group=='leaf' and args.leaf_policy=='boundary':
                boundary_phase(corpus,checkpoint,phase,args,out,baseline)
            elif group=='leaf' and args.leaf_policy=='coordinate':
                coordinate_phase(corpus,checkpoint,phase,args,out,baseline)
            elif group=='score' and args.score_policy=='coordinate':
                coordinate_phase(corpus,checkpoint,phase,args,out,baseline,'score')
            elif group=='leaf':
                fitter=F.LeafFit(corpus.fit_cases(current),current,baseline,args.margin,
                                args.batch_size,args.seed+1009*(phase+1),checkpoint['optimizer'],
                                args.leaf_rate,args.leaf_policy=='backtrack')

                def persist(state):
                    checkpoint['optimizer']=copy.deepcopy(state)
                    save(out,checkpoint)

                if not fitter.data:
                    checkpoint['history'].append(dict(phase=phase,group=group,accepted=False,
                                                       reasons=['没有同档可拟合比较']))
                while fitter.data and fitter.state['epoch'] < args.epochs:
                    info=fitter.epoch(persist,deadline)
                    proposal=fitter.state['params']
                    before=corpus.metrics(checkpoint['params'],'train',baseline)
                    metrics=corpus.metrics(proposal,'train',baseline)
                    reasons=keep_candidate(checkpoint,proposal,metrics,before)
                    accepted=not reasons
                    fitter.feedback(accepted)
                    checkpoint['history'].append(dict(phase=phase,epoch=fitter.state['epoch'],group=group,
                        accepted=accepted,reasons=reasons,metrics=metrics,previous=before,fit=info))
                    persist(fitter.state)
                    write_report(out,checkpoint)
                    print(f'epoch {fitter.state["epoch"]}/{args.epochs} '
                          f'差距={metrics["regret"]:.6f} 接受={accepted} {reasons}',flush=True)
                checkpoint['candidate_train_compute']=corpus.verify(checkpoint['params'],'train')
                passed,nodes=corpus.compute_guard(checkpoint['params'],base_compute['nodes'])
                if not passed:
                    raise AssertionError('叶子拟合阶段选优后实际搜索节点超过限额')
            else:
                proposal=T.perturb(current,baseline,'score',args.seed+1009*(phase+1))
                passed,nodes=corpus.compute_guard(proposal,base_compute['nodes'])
                if passed:
                    try:
                        corpus.prepare([baseline,initial,current,proposal],'train')
                        before=corpus.metrics(current,'train',baseline)
                        metrics=corpus.metrics(proposal,'train',baseline)
                        reasons=quality_reasons(metrics,before)
                        if not reasons:
                            corpus.verify(proposal,'train')
                            keep_candidate(checkpoint,proposal,metrics,before)
                    except T.SearchLimit as exc:
                        if time.monotonic() >= deadline:
                            raise U.BudgetStop('试参达到总时长预算') from None
                        before=metrics=None
                        reasons=[str(exc)]
                else:
                    before=metrics=None
                    reasons=['正式搜索节点超过同集基线限额，未继续教师标注']
                checkpoint['history'].append(dict(phase=phase,group=group,accepted=not reasons,
                    reasons=reasons,metrics=metrics,previous=before,nodes=nodes))
            checkpoint['baseline_train']=corpus.metrics(baseline,'train',baseline)
            checkpoint['initial_train']=corpus.metrics(initial,'train',baseline)
            checkpoint['candidate_train']=corpus.metrics(checkpoint['params'],'train',baseline)
            checkpoint['phase']=phase+1
            checkpoint['boundary']=None
            checkpoint['optimizer']=None
            save(out,checkpoint)
            write_report(out,checkpoint)
        corpus.prepare([baseline,initial,checkpoint['params']],'validation')
        checkpoint['baseline_validation']=corpus.metrics(baseline,'validation',baseline)
        checkpoint['initial_validation']=corpus.metrics(initial,'validation',baseline)
        checkpoint['candidate_validation']=corpus.metrics(checkpoint['params'],'validation',baseline)
        corpus.verify(checkpoint['params'],'validation')
        reasons=quality_reasons(checkpoint['candidate_train'],checkpoint['baseline_train'])
        reasons+=['验证集：'+r for r in quality_reasons(
            checkpoint['candidate_validation'],checkpoint['baseline_validation'])]
        checkpoint['baseline_validation_compute']=corpus.calibrate(baseline,'validation')
        checkpoint['candidate_validation_compute']=corpus.calibrate(checkpoint['params'],'validation')
        if not U.within_cost(checkpoint['candidate_validation_compute'],
                             checkpoint['baseline_validation_compute'],settings):
            reasons.append('验证集串行节点/搜索时长超过限额')
        improved_initial=(not quality_reasons(checkpoint['candidate_train'],checkpoint['initial_train'])
                          and not quality_reasons(checkpoint['candidate_validation'],checkpoint['initial_validation']))
        if args.leaf_policy=='boundary' and args.initial_params:
            # 续训保底是已评审起点；不能用相对默认值的旧收益掩盖本轮验证退步。
            reasons+=['相对初始参数：'+r for r in quality_reasons(
                checkpoint['candidate_train'],checkpoint['initial_train'])]
            reasons+=['相对初始验证：'+r for r in quality_reasons(
                checkpoint['candidate_validation'],checkpoint['initial_validation'])]
        checkpoint.update(status='complete',recommended=not reasons,final_reasons=reasons,
                          improved_initial=improved_initial)
        export=dict(schema=U.SCHEMA,source=U.source_fingerprint(),training_source=fingerprint(),
                    experiment=T.digest(identity),t0=args.t0,baseline=baseline,params=checkpoint['params'],
                    recommended=not reasons,teacher=teacher.identity,validation_role=args.validation_role,
                    initial=initial,improved_initial=improved_initial)
        U.write_json(out/'candidate.json',export)
        fallback=initial if args.leaf_policy=='boundary' else baseline
        U.write_json(out/'best.json',dict(export,params=checkpoint['params'] if not reasons else fallback))
        print(f'完成，建议采用={not reasons}，原因={reasons}',flush=True)
        code=0
    except (U.BudgetStop,KeyboardInterrupt) as exc:
        checkpoint.update(status='paused',stop_reason=str(exc) or '用户中断')
        print(f'已保存断点：{checkpoint["stop_reason"]}',flush=True)
        code=2
    except (ValueError,RuntimeError,OSError,AssertionError,TimeoutError) as exc:
        checkpoint.update(status='failed',stop_reason=f'{type(exc).__name__}: {exc}')
        print(checkpoint['stop_reason'],file=sys.stderr,flush=True)
        code=1
    checkpoint['last_call_seconds']=time.monotonic()-started
    save(out,checkpoint)
    write_report(out,checkpoint)
    return code


def main(argv=None):
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):
            stream.reconfigure(encoding='utf-8',errors='replace')
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    status=sub.add_parser('status')
    status.add_argument('--out',required=True)
    p=sub.add_parser('run')
    p.add_argument('--out',required=True)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--games',action='append')
    source.add_argument('--positions-file')
    p.add_argument('--split-from',help='继承旧局面分组，防止把旧保留家族移入训练')
    p.add_argument('--reuse-labels',action='append',default=[],help='只读复用兼容教师缓存目录')
    p.add_argument('--reuse-search-cache',action='append',default=[],
                   help='只读复用已完成teacher_train实验的兼容树/探针；最终串行计时重测')
    p.add_argument('--teacher',required=True)
    p.add_argument('--initial-params',help='从已验证参数开始，正式默认值仍作为成本和范围基线')
    p.add_argument('--leaf-policy',choices=('fixed','backtrack','coordinate','boundary'),default='fixed')
    p.add_argument('--leaf-rate',type=U.positive,default=.015)
    p.add_argument('--leaf-evaluations',type=U.positive_int,default=1024,
                   help='仅boundary：每阶段候选评估尝试上限，含中断前预留次数')
    p.add_argument('--score-policy',choices=('random','coordinate'),default='random')
    p.add_argument('--score-keys',type=score_axes,help='前段坐标试探轴，以逗号分隔；省略时扫描完整score组')
    p.add_argument('--validation-role',choices=('independent','seen'),default='independent')
    for name,default in (('positions',600),('seed',1),('min-ply',8),('max-ply',60),('stride',6),
                         ('teacher-nodes',200000),('teacher-ms',2000),('max-queries',20000),
                         ('workers',4),('rounds',3),('trials',8),('epochs',30),('batch-size',8),
                         ('node-limit',50000)):
        p.add_argument('--'+name,type=int if name=='seed' else U.positive_int,default=default)
    for name,default in (('t0',80),('max-seconds',14400),('search-seconds',15),('margin',.08),
                         ('cost-ratio',1.1),('time-ratio',1.25)):
        p.add_argument('--'+name,type=U.positive,default=default)
    p.add_argument('--groups',default='leaf,score')
    p.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv)
    try:
        if args.command=='status':
            checkpoint=U.read_json(Path(args.out)/'checkpoint.json')
            if checkpoint['identity']['schema']!=SCHEMA:
                raise ValueError('这不是新版训练实验')
            write_report(args.out,checkpoint)
            print(Path(args.out)/'report.md')
            return 0
        if args.min_ply>args.max_ply or args.workers>max(1,os.cpu_count() or 1):
            raise ValueError('采样范围或 worker 数不合法')
        return run(args)
    except (ValueError,KeyError,OSError,RuntimeError) as exc:
        print(f'{type(exc).__name__}: {exc}',file=sys.stderr,flush=True)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
