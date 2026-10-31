"""离线参数提案与固定树上的小批次 Adam；无神经网络或第三方依赖。"""
import math
import random
import copy

from gomoku.tools import tune_params as T


def coordinate_order(seed, group='leaf'):
    if group not in ('leaf','score'):
        raise ValueError('坐标搜索只允许叶子或候选评分权重')
    keys=list(T.GROUPS[group])
    random.Random(seed).shuffle(keys)
    return keys


def coordinate_proposals(key, params, base, group='leaf'):
    """用原权重4%/8%/16%的正负步长探测真实选点目标，不依赖平滑梯度。"""
    if group not in ('leaf','score') or key not in T.GROUPS[group]:
        raise ValueError('坐标搜索参数不属于指定组')
    seen=set()
    for scale in (.04,.08,.16):
        for direction in (-1,1):
            proposal=T.project(dict(params,**{key:params[key]+direction*scale*base[key]}),base)
            # 投影可能造成其他组的浮点微差；固定非目标组，不能误改叶子或搜索设置。
            proposal.update({k:v for k,v in params.items() if k not in T.GROUPS[group]})
            if all(math.isclose(proposal[k],params[k],rel_tol=1e-10,abs_tol=1e-8) for k in base):
                continue
            signature=T.digest(proposal)
            if signature not in seen:
                seen.add(signature)
                yield dict(key=key,direction=direction,relative_step=scale,params=proposal)


class CachedRanking:
    """固定树最高五态档的选点签名；叶子公式始终复用 tune_params.replay_row。

    仅合并重复叶子特征的求值，不改每条候选的加权求和顺序或并列优先级。
    调用方只传拟合折，不能用外部验证/内部留出折引导分界细化。
    """
    def __init__(self, cases):
        self.leaves, self.rows = [], []
        indices = {}
        for case in cases:
            rows = case['tree']['rows']
            if not rows:
                raise ValueError('分界搜索不能使用空候选树')
            tier = max(T.DeepSearch.STATE_ORDER[r['state']] for r in rows)
            encoded = []
            for row in rows:
                if T.DeepSearch.STATE_ORDER[row['state']] != tier:
                    continue
                terms = []
                for weight, feature in row['terms']:
                    key = tuple(feature)
                    if key not in indices:
                        indices[key] = len(self.leaves)
                        self.leaves.append(dict(constant=0.0, terms=[[1.0, feature]]))
                    terms.append((weight, indices[key]))
                encoded.append(((row['r'], row['c']), row['constant'], terms))
            self.rows.append(encoded)

    def signature(self, params):
        values = [T.replay_row(leaf, params)[0] for leaf in self.leaves]
        result = []
        for encoded in self.rows:
            scored = []
            for point, constant, terms in encoded:
                value = constant
                for weight, index in terms:
                    value += weight * values[index]
                scored.append((point, value))
            result.append(max(scored, key=lambda item: item[1])[0])
        return result


class BoundaryFit:
    """有预算的范围扩展与选点分界细化；不改变合法参数范围或质量门槛。

    每阶段固定原点，先按半径轮询各轴扩展至原盒边界，再二分拟合折上
    已观察到选点不同的相邻区间。有限采样可能漏掉两端签名相同的窄窗口。
    pending 在评分前持久化；reserve 保守计入尝试预算，续跑不退还未完成尝试。
    """
    RADII = (.04, .08, .16, .32, .64, 1.28, 2.56)
    MAX_DEPTH = 8
    MIN_WIDTH = 1e-6

    def __init__(self, cases, initial, base, seed=1, budget=1024, state=None):
        if type(budget) is not int or budget < 1:
            raise ValueError('分界评分预算须为正整数')
        T.validate(initial,base)
        self.initial, self.base = dict(initial), dict(base)
        self.axes = coordinate_order(seed)
        self.ranking = CachedRanking(cases)
        context = T.digest(['leaf-boundary-v1', initial, base, self.axes, budget,
                            self.RADII, self.MAX_DEPTH, self.MIN_WIDTH,
                            [(c['position']['id'], c['tree']['rows']) for c in cases]])
        self.coarse = []
        for radius in self.RADII + (None,):
            for key in self.axes:
                lo, hi = T.bounds(base, key)
                lo, hi = (lo-initial[key])/base[key], (hi-initial[key])/base[key]
                for alpha in ((lo, hi) if radius is None else (-radius, radius)):
                    self.coarse.append(dict(key=key, alpha=min(hi, max(lo, alpha)), depth=0, kind='expand'))
        if state is not None:
            if state.get('context') != context:
                raise ValueError('分界搜索输入、原点、顺序或预算变化，拒绝续跑')
            self.state = copy.deepcopy(state)
            attempts,completed=state.get('attempts'),state.get('completed')
            cursor=state.get('cursor')
            if (state.get('budget')!=budget or type(attempts) is not int
                    or type(completed) is not int or not 0<=completed<=attempts<=budget
                    or type(cursor) is not int or not 0<=cursor<=len(self.coarse)
                    or set(state.get('samples',{}))!=set(self.axes)):
                raise ValueError('分界断点计数、预算或轴数据损坏')
            pending=state.get('pending')
            if pending is not None:
                if (pending.get('key') not in self.axes
                        or not isinstance(pending.get('alpha'),(int,float))
                        or not math.isfinite(pending['alpha'])):
                    raise ValueError('分界待评分提案损坏')
                expected=self._params(pending)
                if pending.get('params')!=expected or pending.get('signature_key')!=T.digest(expected):
                    raise ValueError('分界待评分参数与原点不符')
        else:
            signature = T.digest(self.ranking.signature(initial))
            self.state = dict(context=context, origin=dict(initial), budget=budget, cursor=0,
                              attempts=0, completed=0, pending=None, stop_reason=None,
                              samples={key:[dict(alpha=0.0, signature=signature, depth=0)] for key in self.axes},
                              seen={T.digest(initial):signature}, duplicates=0)

    def _params(self, item):
        if item['alpha'] == 0:
            return dict(self.initial)
        raw = dict(self.initial)
        key = item['key']
        raw[key] += item['alpha'] * self.base[key]
        params = T.project(raw, self.base)
        params.update({k:v for k,v in self.initial.items() if k not in T.LEAF_KEYS})
        return params

    def _sample(self, item, signature):
        samples = self.state['samples'][item['key']]
        if not any(s['alpha'] == item['alpha'] for s in samples):
            samples.append(dict(alpha=item['alpha'], signature=signature, depth=item['depth']))

    def _refinement(self):
        intervals = []
        for index, key in enumerate(self.axes):
            samples = sorted(self.state['samples'][key], key=lambda s:s['alpha'])
            for a,b in zip(samples,samples[1:]):
                width = b['alpha']-a['alpha']
                depth = max(a['depth'],b['depth'])+1
                if a['signature'] != b['signature'] and width > self.MIN_WIDTH and depth <= self.MAX_DEPTH:
                    intervals.append((width,-index,-a['alpha'],dict(key=key,
                        alpha=(a['alpha']+b['alpha'])/2,depth=depth,kind='refine')))
        return max(intervals,key=lambda item:item[:3])[3] if intervals else None

    def next_proposal(self):
        if self.state['attempts'] >= self.state['budget']:
            self.state['stop_reason'] = 'evaluation-budget'
            return None
        if self.state['pending'] is not None:
            return self.state['pending']
        while True:
            if self.state['cursor'] < len(self.coarse):
                item = dict(self.coarse[self.state['cursor']])
                self.state['cursor'] += 1
            else:
                item = self._refinement()
                if item is None:
                    self.state['stop_reason'] = 'range-and-observed-boundaries-exhausted'
                    return None
            params = self._params(item)
            key = T.digest(params)
            if key in self.state['seen']:
                self._sample(item,self.state['seen'][key])
                self.state['duplicates'] += 1
                continue
            item.update(params=params,signature_key=key,reserved=False,
                        direction=1 if item['alpha'] >= 0 else -1,relative_step=abs(item['alpha']))
            self.state['pending'] = item
            return item

    def reserve(self):
        pending = self.state['pending']
        if pending is None or self.state['attempts'] >= self.state['budget']:
            raise ValueError('没有待评分提案或分界预算已耗尽')
        self.state['attempts'] += 1
        pending['reserved'] = True

    def observe(self, signature):
        pending = self.state['pending']
        if pending is None or not pending['reserved']:
            raise ValueError('评分前必须先预留尝试预算')
        signature = T.digest(signature)
        self._sample(pending,signature)
        self.state['seen'][pending['signature_key']] = signature
        self.state['completed'] += 1
        self.state['pending'] = None


def examples(cases, margin):
    """最佳可达同档候选对其余候选；不让低五态分支跨档竞争。"""
    eligible = []
    for case in cases:
        rows = case['tree']['rows']
        tier = max(T.DeepSearch.STATE_ORDER[r['state']] for r in rows)
        rows = [r for r in rows if T.DeepSearch.STATE_ORDER[r['state']] == tier]
        values = [case['utilities'][f'{r["r"]},{r["c"]}'] for r in rows]
        best = max(range(len(rows)), key=lambda i: values[i])
        pairs = [(best, j, values[best] - value) for j, value in enumerate(values)
                 if values[best] - value > margin
                 and (rows[best]['terms'] or rows[j]['terms'])]
        if pairs:
            eligible.append(dict(rows=rows, pairs=pairs,
                                 group=case['position'].get('group', case['position']['id'])))
    return eligible


def batch_gradient(batch, params, base):
    grad, loss = [0.0] * len(T.LEAF_KEYS), 0.0
    for example in batch:
        values = [T.replay_row(row, params, gradient=True) for row in example['rows']]
        for i, j, gap in example['pairs']:
            z = (values[i][0] - values[j][0]) / 500
            sigmoid = 1 / (1 + math.exp(max(-60, min(60, z))))
            divisor = len(batch) * len(example['pairs'])
            loss += gap * (max(-z, 0) + math.log1p(math.exp(-abs(z)))) / divisor
            for k, name in enumerate(T.LEAF_KEYS):
                grad[k] -= gap * sigmoid * (values[i][1][k] - values[j][1][k]) * base[name] / 500 / divisor
    for k, name in enumerate(T.LEAF_KEYS):
        relative = params[name] / base[name] - 1
        grad[k] += .002 * relative
        loss += .001 * relative * relative
    return loss, grad


class LeafFit:
    """保存 Adam 动量与 epoch/cursor；崩溃后不重置优化器。"""
    def __init__(self, cases, initial, base, margin=.08, batch_size=8, seed=1, state=None,
                 learning_rate=.015, backtrack=False):
        self.data = examples(cases, margin)
        self.base, self.batch_size, self.seed = base, batch_size, seed
        self.backtrack = backtrack
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError('拟合步长必须为正有限数')
        context = T.digest([self.data, margin, batch_size, seed, base, learning_rate, backtrack])
        if state is None:
            self.state = dict(context=context, params=dict(initial), m=[0.0]*24, v=[0.0]*24,
                              updates=0, adam_step=0, epoch=0, cursor=0, rate=learning_rate,
                              backtracks=0, anchor=dict(params=dict(initial),m=[0.0]*24,
                                                       v=[0.0]*24,adam_step=0))
        else:
            if state['context'] != context:
                raise ValueError('拟合样本或配置变化，不能沿用优化器断点')
            self.state = state

    def feedback(self, accepted):
        """不改变验收条件；拒绝后恢复最好轨迹，缩小下一次更新的步长。"""
        if not self.backtrack:
            return
        if accepted:
            self.state['anchor'] = dict(params=dict(self.state['params']),
                m=list(self.state['m']),v=list(self.state['v']),adam_step=self.state['adam_step'])
        else:
            anchor=self.state['anchor']
            self.state.update(params=dict(anchor['params']),m=list(anchor['m']),v=list(anchor['v']),
                              adam_step=anchor['adam_step'],rate=self.state['rate']*.5,
                              backtracks=self.state['backtracks']+1)

    def epoch(self, checkpoint=None, deadline=None):
        if not self.data:
            return dict(positions=0, pairs=0, updates=0, loss=0.0)
        import time
        order = list(range(len(self.data)))
        random.Random(self.seed + self.state['epoch']).shuffle(order)
        losses, start_updates = [], self.state['updates']
        for start in range(self.state['cursor'], len(order), self.batch_size):
            if deadline is not None and time.monotonic() >= deadline:
                from gomoku.tools.teacher_tune import BudgetStop
                raise BudgetStop('叶子拟合达到总时长预算')
            batch = [self.data[i] for i in order[start:start+self.batch_size]]
            loss, grad = batch_gradient(batch, self.state['params'], self.base)
            losses.append(loss)
            self.state['updates'] += 1
            self.state['adam_step'] += 1
            step = self.state['adam_step']
            params = dict(self.state['params'])
            for k, name in enumerate(T.LEAF_KEYS):
                self.state['m'][k] = .9*self.state['m'][k] + .1*grad[k]
                self.state['v'][k] = .999*self.state['v'][k] + .001*grad[k]**2
                update = self.state['rate']*(self.state['m'][k]/(1-.9**step))/(math.sqrt(self.state['v'][k]/(1-.999**step))+1e-8)
                params[name] -= self.base[name]*update
            self.state['params'] = T.project(params, self.base)
            self.state['cursor'] = start + len(batch)
            if checkpoint:
                checkpoint(self.state)
        self.state['epoch'] += 1
        self.state['cursor'] = 0
        return dict(positions=len(self.data), pairs=sum(len(e['pairs']) for e in self.data),
                    updates=self.state['updates']-start_updates,
                    learning_rate=self.state['rate'],backtracks=self.state['backtracks'],
                    loss=sum(losses)/len(losses) if losses else None)
