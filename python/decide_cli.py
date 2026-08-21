# ============================================================
# decide_cli.py —— 插件对接入口
# 用法: python decide_cli.py <sessionId>
#   1. 从 DSH 五子棋插件存档读取该会话的完整棋谱（moves 含顺序）
#   2. 重放全部落子 → 维护黑白状态表（3 段更新）
#   3. 轮到白（AI）时输出推荐落点 + 状态描述
# 输出: {"row":.., "col":.., "desc":"..."}（UTF-8 JSON）
# ============================================================

import json
import os
import sys

from utils import empty_board, place, BLACK, WHITE
from engine import Engine


def data_file():
    home = os.environ.get('DSH_HOME') or os.path.join(os.path.expanduser('~'), '.dsh')
    return os.path.join(home, 'storages', 'gomoku', 'games.json')


def load_moves(session_id):
    """从插件存档读取指定会话的落子序列 [{'r','c','player'}, ...]"""
    path = data_file()
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        games = json.load(f)
    g = games.get(session_id)
    if not g or not isinstance(g.get('moves'), list):
        return None
    return g['moves']


def main():
    if len(sys.argv) < 2:
        print(json.dumps({'error': '需要 sessionId 参数'}, ensure_ascii=False))
        return
    sid = sys.argv[1]
    moves = load_moves(sid)
    if moves is None:
        print(json.dumps({'error': '未找到会话存档: ' + sid}, ensure_ascii=False))
        return

    engine = Engine()
    board = empty_board()
    for m in moves:
        place(board, m['r'], m['c'], m['player'])
        engine.on_move(board, m['r'], m['c'], m['player'])

    result = engine.compute_move(board)
    # 开局时返回候选列表（供 agent 决策），否则返回单点
    if isinstance(result, list):
        out = {
            'opening': True,
            'candidates': [{'row': r, 'col': c} for (r, c) in result],
            'desc': engine.describe(board),
            'moves': len(moves),
        }
        print(json.dumps(out, ensure_ascii=False))
        return
    row, col = result
    out = {
        'row': row,
        'col': col,
        'desc': engine.describe(board),
        'moves': len(moves),
    }
    print(json.dumps(out, ensure_ascii=False))


if __name__ == '__main__':
    main()
