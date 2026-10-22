# -*- coding: utf-8 -*-
"""引擎服务：封装核心引擎，对外提供规范化 AI 落子。

所有入口统一委托 Engine.analyze_turn：
  - 本方前三手 → 棋谱/谱外开局规则；
  - 后续 → 谱点参与根候选，按原温度银行与五态排序选点。
返回 ((row, col), ranked, net_used)。ranked 供界面按五态和完整分展示全候选；未补算的已淘汰硬败分为 None。
"""
import json
import os

from gomoku.core.utils import SIZE, EMPTY, BLACK, WHITE
from gomoku.core.engine import Engine
from gomoku.tools import ga_tune as G

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_engine = None

# Engine.opening_source 的玩家可读说明。图形界面与命令行共用一份，避免两端文案漂移。
DECISION_SOURCE_LABELS = {
    'book': '棋谱开局',
    'rules': '谱外开局规则',
    'candidate': '棋谱候选＋纯算法',
}


def decision_source_label(source):
    """把 Engine.opening_source 翻成界面文案；未设置或未知来源时回退纯算法。"""
    return DECISION_SOURCE_LABELS.get(source, '纯算法')


def load_ga_params():
    """DSH_GOMOKU_GA=1 时加载包内 GA 参数，缺失或损坏时回退默认。"""
    if os.environ.get('DSH_GOMOKU_GA') != '1':
        return
    path = os.path.join(ROOT, 'gomoku', 'data', 'ga_best_params.json')
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        params = data.get('params', data)
        G.apply_params(G._fix_genome(params))
    except Exception:
        pass


def create_engine(use_decision_net=False):
    """新会话拥有独立引擎，所有运行入口使用同一配置。
    暂定：默认纯算法（决策网络停用），不再读取 DSH_GOMOKU_USE_NET。"""
    load_ga_params()
    return Engine(use_decision_net=bool(use_decision_net))


def get_engine(use_decision_net=None):
    """兼容无会话调用的单例；正常会话使用 create_engine 隔离状态。"""
    global _engine
    if _engine is None:
        _engine = create_engine(use_decision_net=use_decision_net)
    elif use_decision_net is not None:
        _engine.use_decision_net = use_decision_net
    return _engine


def ai_move(board, moves, player, engine=None):
    """规范化 AI 落子。engine 可传入已与 board 同步的实例（界面用）；缺省用单例。

    返回 ((row, col), ranked, net_used)。
    """
    if engine is None:
        engine = get_engine()
        engine.reset(moves)
    # 清空上一手的显示统计；核心会按本次开局/深推的实际路径重新记录。
    engine.deep_search.reset_leaf_eval_stats()
    if player not in (BLACK, WHITE):
        raise ValueError('bad player')
    if not any(EMPTY in row for row in board):
        return None, [], False

    # 开局、谱点候选和正式搜索统一由核心编排，所有适配入口同源。
    res = engine.analyze_turn(board, player)
    mv = res['part4_result']['move']
    ranked = res['part4_result'].get('ranked') or []
    net_used = engine.decision_net_used

    if isinstance(mv, list):
        mv = mv[0] if mv else None
    if mv is None:
        r, c = 7, 7
        if board[r][c] != 0:
            for rr in range(15):
                for cc in range(15):
                    if board[rr][cc] == 0:
                        r, c = rr, cc
                        break
                else:
                    continue
                break
        return (r, c), ranked, net_used
    return (mv[0], mv[1]), ranked, net_used
