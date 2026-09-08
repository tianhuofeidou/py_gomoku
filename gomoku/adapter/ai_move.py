# -*- coding: utf-8 -*-
"""单次 JSON 引擎入口，与常驻服务共用配置、校验及落子逻辑。"""
import json
import sys

from gomoku.adapter.server import handle
from gomoku.service import load_ga_params


def main():
    load_ga_params()
    data = json.load(sys.stdin)
    data['type'] = 'move'
    data['session'] = 'once'
    result = handle(data)
    if not result['ok']:
        raise ValueError(result['error'])
    print(json.dumps({key: result[key] for key in ('row', 'col', 'ranked', 'net_used')}))


if __name__ == '__main__':
    main()
