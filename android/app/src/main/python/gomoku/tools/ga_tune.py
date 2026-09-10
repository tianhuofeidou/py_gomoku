# -*- coding: utf-8 -*-
"""Android 端 GA 参数加载占位。

桌面版的 tools/ga_tune.py 依赖 multiprocessing 等桌面调参链路；
移动端只运行纯算法引擎，不启用 GA 调参（需要 DSH_GOMOKU_GA=1）。
保留 service.load_ga_params 需要的两个接口，行为为无操作。
"""


def _fix_genome(params):
    return params


def apply_params(params):
    return None
