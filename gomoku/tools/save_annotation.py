# -*- coding: utf-8 -*-
"""对弈标注辅助（保存）：追加一条人类标注到 human_annotations.json。

输入（stdin JSON）：
  {
    "moves": [{"r":..,"c":..,"player":..}, ...],   # 标注时刻的完整着法序列
    "player": 1|2,                                  # 轮到谁（候选视角）
    "candidates": [{"r":..,"c":..,"score":..}, ...],# 展示给人类的候选（含深推分）
    "labels": [{"r":..,"c":..,"level":"best"|"second"|"bad"}, ...],
    "played": {"r":..,"c":..} | null                # 实际落子点
  }

level 语义：best=最好 / second=次好 / bad=不能走。
标注规范（人类标注者约定）：
  - bad 只标「走了会被绝杀」的点（对方必杀/直接输），普通差手不标 bad；
  - 其余「可走但不突出」的点一律不标，只记录明确的三档；
  - 不在引擎候选里的点可补进 candidates（deep=false, score=0），
    保留「引擎漏点」信息。
数据文件：gomoku/data/human_annotations.json（JSON 数组，读-追加-写）。
"""
import json
import os
import sys

# __file__ = gomoku/tools/save_annotation.py → 退 2 层到 gomoku/，再进 data/
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'data', 'human_annotations.json')


def main():
    rec = json.load(sys.stdin)
    rec.setdefault('played', None)
    rec.setdefault('labels', [])
    rec.setdefault('candidates', [])
    # 校验 level 合法性
    for lab in rec['labels']:
        assert lab['level'] in ('best', 'second', 'bad'), 'bad level: ' + str(lab)

    rows = []
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, encoding='utf-8') as f:
            try:
                rows = json.load(f)
            except Exception:
                rows = []
    rows.append(rec)
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    with open(DATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print(json.dumps({'ok': True, 'total': len(rows), 'path': DATA_PATH}, ensure_ascii=False))


if __name__ == '__main__':
    main()
