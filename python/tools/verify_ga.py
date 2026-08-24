# ============================================================
# verify_ga.py —— GA 结果满深度验证
# 用法：python verify_ga.py --params ga_best_params.json --games 50 --workers 8
# 说明：最优参数 vs 基线默认，T0=80 满深度自对弈，黑白轮换
# ============================================================

import argparse
import json
from multiprocessing import Pool

from python.tools import ga_tune as G


def one_game(task):
    gi, genome, opening, max_moves, t0 = task
    G.FAST_T0 = t0
    black_is_trial = (gi % 2 == 0)
    pb = genome if black_is_trial else G.BASELINE
    pw = G.BASELINE if black_is_trial else genome
    return G.play_game(pb, pw, opening, max_moves)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--params', default='ga_best_params.json', help='GA 输出 JSON')
    ap.add_argument('--games', type=int, default=50, help='验证局数（黑白各半）')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--t0', type=float, default=80.0, help='满深度温度')
    ap.add_argument('--max-moves', type=int, default=60)
    args = ap.parse_args()

    data = json.load(open(args.params, encoding='utf-8'))
    genome = G._fix_genome(data['params'])
    half = args.games // 2
    tasks = [(i, genome, G.OPENINGS[i % len(G.OPENINGS)], args.max_moves, args.t0)
             for i in range(args.games)]
    pool = Pool(args.workers)
    res = pool.map(one_game, tasks)
    pool.close()
    pool.join()

    res_b = res[:half]
    res_w = res[half:]
    wins_b = res_b.count('B')
    wins_w = res_w.count('W')
    draws = res.count(None)
    print('as-black : %d/%d wins (%.1f%%)' % (wins_b, len(res_b), 100.0 * wins_b / max(len(res_b), 1)))
    print('as-white : %d/%d wins (%.1f%%)' % (wins_w, len(res_w), 100.0 * wins_w / max(len(res_w), 1)))
    print('draws    : %d/%d' % (draws, args.games))
    print('overall  : %d/%d wins (%.1f%%)' % (wins_b + wins_w, args.games, 100.0 * (wins_b + wins_w) / args.games))


if __name__ == '__main__':
    main()
