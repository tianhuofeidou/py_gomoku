# -*- coding: utf-8 -*-
"""测试作用域：把机制回归显式固定到旧FREE/225/中途160口径，不影响发布默认测试。"""
from gomoku.core.deep_search import DeepSearch

_STACK = []
_NAMES = ('TACTICAL_FREE_PLIES', 'TACTICAL_BATCH_SIZE', 'MIDDLE_COLD_T0')


def enter_legacy():
    _STACK.append({name: getattr(DeepSearch, name) for name in _NAMES})
    DeepSearch.TACTICAL_FREE_PLIES = None
    DeepSearch.TACTICAL_BATCH_SIZE = 15 * 15
    DeepSearch.MIDDLE_COLD_T0 = 160.0


def leave_legacy():
    saved = _STACK.pop()
    for name, value in saved.items():
        setattr(DeepSearch, name, value)
