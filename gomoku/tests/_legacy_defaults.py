# -*- coding: utf-8 -*-
"""旧机制夹具显式关闭根最低深度，并固定FREE/225/中途160；不改变发布默认。"""
from gomoku.core.deep_search import DeepSearch

_STACK = []
_NAMES = ('TACTICAL_FREE_PLIES', 'TACTICAL_BATCH_SIZE', 'MIDDLE_COLD_T0',
          'MIN_SEARCH_PLIES')


def enter_legacy():
    _STACK.append({name: getattr(DeepSearch, name) for name in _NAMES})
    DeepSearch.TACTICAL_FREE_PLIES = None
    DeepSearch.TACTICAL_BATCH_SIZE = 15 * 15
    DeepSearch.MIDDLE_COLD_T0 = 160.0
    DeepSearch.MIN_SEARCH_PLIES = 0


def leave_legacy():
    saved = _STACK.pop()
    for name, value in saved.items():
        setattr(DeepSearch, name, value)
