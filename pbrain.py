# -*- coding: utf-8 -*-
"""Gomocup / Piskvork 协议入口（打包用）。

为什么单独一个入口：协议要求可执行文件名以 `pbrain-` 开头，且必须是**控制台程序**
（stdin/stdout 通信，不能是 GUI 子系统）。因此用本文件作为 PyInstaller 入口，
配合 `--console` 打包：

    python -m PyInstaller --noconfirm --clean --onefile --console \
      --name "pbrain-QifengXiaojing" \
      --add-data "gomoku/data/ga_best_params.json;gomoku/data" \
      --exclude-module torch --exclude-module numpy --exclude-module gomoku.nn \
      --specpath build --workpath build/work --distpath dist \
      pbrain.py

逻辑本身都在 `gomoku/protocol.py`（协议解析 + 选点委托 service.ai_move），这里只做入口，
避免出现第二份决策或协议实现。
"""
import sys

from gomoku.protocol import main

if __name__ == '__main__':
    sys.exit(main())
