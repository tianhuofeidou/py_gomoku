# -*- coding: utf-8 -*-
"""独立版路径配置：把插件里散落在 ~/.dsh 的存档，改为独立根 ~/.gomoku。

可用环境变量 GOMOKU_HOME 覆盖存根（默认 ~/.gomoku）。
"""
import os
from os.path import join, expanduser


def base_dir():
    """独立版数据根目录（默认 ~/.gomoku，可用 GOMOKU_HOME 覆盖）。"""
    return os.environ.get('GOMOKU_HOME') or join(expanduser('~'), '.gomoku')


def games_file():
    return join(base_dir(), 'games.json')


def global_memory_file():
    return join(base_dir(), 'global-memory.json')


def self_memory_file():
    return join(base_dir(), 'self-memory.json')


def ensure_base():
    os.makedirs(base_dir(), exist_ok=True)
    return base_dir()
