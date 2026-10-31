"""固定树上的成对选点边界提案；仅供离线研究，不参与正式选棋。"""
import math

from gomoku.tools import tune_params as T


def pair_proposals(case, initial, base, margin=.08, iterations=24):
    """逐次修正目标与当前竞争点的分差，投影后确认真实选择确实改善。

    调用方仅传训练拟合折。局部梯度不是跨界证明：投影、其他候选和非线性
    都可能阻止换点，因此只返回精确重放后教师效用更高的参数。无提案不证明
    不可优化。固定五态优先级，非叶子字段逐字保留，不混用缺失候选或软状态。
    """
    if not math.isfinite(margin) or margin < 0 or type(iterations) is not int or iterations < 1:
        raise ValueError('成对提案范围无效')
    T.validate(initial, base)
    rows = case['tree']['rows']
    current = T.choose(rows, initial)
    utility = lambda row: case['utilities'][f'{row["r"]},{row["c"]}']
    tier = T.DeepSearch.STATE_ORDER[current['state']]
    targets = [row for row in rows if T.DeepSearch.STATE_ORDER[row['state']] == tier
               and utility(row) > utility(current) + margin]
    targets.sort(key=lambda row: (-utility(row), -T.replay_row(row, initial)[0]))
    seen = set()
    for target in targets:
        params = dict(initial)
        for iteration in range(iterations):
            rival = T.choose(rows, params)
            if utility(rival) > utility(current) + margin:
                # 在观察到成功的路径上收缩，最终仍精确检查，不能假设投影路径单调。
                low, high = 0.0, 1.0
                successful = dict(params)
                for _ in range(14):
                    alpha = (low + high) / 2
                    trial = _project(initial, {k:initial[k]+alpha*(params[k]-initial[k])
                                              for k in T.LEAF_KEYS}, base)
                    chosen = T.choose(rows, trial)
                    if utility(chosen) > utility(current) + margin:
                        high, successful = alpha, trial
                    else:
                        low = alpha
                # 多个略越界尺度用于抵抗边界附近的脆弱性；不是随机试参。
                for scale in (1.001, 1.05, 1.25, 2.0):
                    trial = _project(initial, {k:initial[k]+scale*(successful[k]-initial[k])
                                              for k in T.LEAF_KEYS}, base)
                    chosen = T.choose(rows, trial)
                    key = T.digest(trial)
                    if key not in seen and utility(chosen) > utility(current) + margin:
                        seen.add(key)
                        yield dict(params=trial, case_id=case['position']['id'],
                                   target=[target['r'], target['c']],
                                   before=[current['r'], current['c']],
                                   after=[chosen['r'], chosen['c']],
                                   utility_gain=utility(chosen)-utility(current),
                                   iterations=iteration, scale=scale)
                break
            good, gg = T.replay_row(target, params, gradient=True)
            bad, bg = T.replay_row(rival, params, gradient=True)
            direction = [(a-b)*base[k] for k,a,b in zip(T.LEAF_KEYS,gg,bg)]
            norm = sum(d*d for d in direction)
            if norm <= 1e-24:
                break
            # 相对参数空间中满足局部线性分差的最短方向；实际效果由重放决定。
            step = max(1e-5, bad-good+max(1e-5, abs(bad)*1e-8)) / norm * 1.05
            relative_max = max(abs(step*d) for d in direction)
            if relative_max > 2.56:
                step *= 2.56 / relative_max
            trial = _project(params, {k:params[k]+step*d*base[k]
                                     for k,d in zip(T.LEAF_KEYS,direction)}, base)
            if max(abs(trial[k]-params[k])/abs(base[k]) for k in T.LEAF_KEYS) < 1e-10:
                break
            params = trial


def _project(initial, leaf_values, base):
    params = T.project(dict(initial, **leaf_values), base)
    params.update({k:v for k,v in initial.items() if k not in T.LEAF_KEYS})
    return params
