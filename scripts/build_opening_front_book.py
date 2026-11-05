# -*- coding: utf-8 -*-
"""从本地实验数据生成整盘前5手稀疏开局库的受跟踪纯数据清单。

本脚本只在需要重建数据时运行。正式引擎不读取 `storages/experiments`，
也不启动任何外部教师。生成物只保留坐标、候选、主着、评分视角和来源标签，
不带实验目录、绝对路径、日志或备份。

输入（相对仓库根目录）：
- 黑五候选：74 组 / 429 个候选。每组保底保留评分前三，其余普通分严格>400
  或带正杀棋标签保留；原 820 / 756 / 414 版本一律不得复用。
- 白四候选：75 条（全候选），最终确认 74 条，删除 (I10, G9, H11)。
- 黑三候选：27 条（24 条同口径逐点评分 + 3 条只有根推荐、无逐点评分）。

用法：
    python -B scripts/build_opening_front_book.py            # 生成数据文件
    python -B scripts/build_opening_front_book.py --check     # 只校验，不写文件
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        try:
            _stream.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gomoku.core.opening_book import (canonical_position, transform_point,  # noqa: E402
                                      inverse_point)
from gomoku.core.opening_book_data import RECORDS, ALTERNATIVES  # noqa: E402
from gomoku.core.utils import BLACK, WHITE, SIZE  # noqa: E402

OUTPUT = ROOT / 'gomoku' / 'core' / 'opening_book_front.py'
EXP = ROOT / 'storages' / 'experiments'

BLACK_FIVE_DIR = '20261105-黑五-棋理候选教师评估-Codex-01'
WHITE_FOUR_DIR = '20261105-白四-全候选教师复核-Codex-01'
WHITE_FOUR_MERGE_DIR = '20261105-白四-同形合并复核-Rapfi-Codex-01'
FIVE_CLASS_DIR = '20261105-白二五类-黑三Rapfi补齐-Codex-01'
KNIGHT_DIR = '20261105-白二走日-黑三Rapfi-Codex-01'
FAR_DIAGONAL_DIR = '20261105-白二远斜-黑三Rapfi-Codex-01'
BLACK_THREE_DIRS = {
    'H9': FIVE_CLASS_DIR,
    'I9': FIVE_CLASS_DIR,
    'H10': FIVE_CLASS_DIR,
    'I10': KNIGHT_DIR,
    'J10': FAR_DIAGONAL_DIR,
}
LABEL_DIRS = (BLACK_FIVE_DIR, WHITE_FOUR_DIR, WHITE_FOUR_MERGE_DIR,
              FIVE_CLASS_DIR, KNIGHT_DIR, FAR_DIAGONAL_DIR)

# 黑一固定 H8；白二按中心八种旋转/镜像归为五类。
# `selectable` 只表示"本方执白时可选"；其余三类用于覆盖对手可能的白二，
# 不构成我方随机开局策略。
WHITE_TWO_CLASSES = (
    ('H9', (8, 7), True, '正邻'),
    ('I9', (8, 8), True, '斜邻'),
    ('H10', (9, 7), False, '隔一路正交'),
    ('I10', (9, 8), False, '走日'),
    ('J10', (9, 9), False, '远斜'),
)

# 人工确认的黑三候选表（docs/开局库实施参考.md）。布尔表示是否有同口径
# 逐点评分；False 表示只有根推荐，只能列为待评。
CONFIRMED_BLACK_THREE = (
    ('H9', ('I7', True), ('I8', True), ('I9', True), ('H10', True), ('I10', True)),
    ('I9', ('I7', True), ('I8', True), ('G8', True), ('J9', True)),
    ('H10', ('G9', True), ('G10', True), ('G8', True), ('F8', True), ('G7', False)),
    ('I10', ('H9', True), ('H10', True), ('I8', True), ('I9', True), ('J9', True), ('G9', False)),
    ('J10', ('I8', True), ('J8', True), ('J9', True), ('K9', True), ('I7', True),
            ('H7', True), ('H6', False)),
)

# 低于 400 已移出的三项；生成时重新核对它们不进入正式库。
EXCLUDED_BLACK_THREE = (('I9', 'J10', 257), ('H10', 'F10', 276), ('I10', 'J10', 305))

# 人工明确首选的黑三：白二 H10 时黑三 G9。
HUMAN_BLACK_THREE = {('H10', 'G9'): '白二 H10 的人工明确偏好点'}

# 人工确认的白四首选：同形合并基准支根推荐 I7，对应支按坐标变换为 G10。
HUMAN_WHITE_FOUR = {
    ('I9', 'G8'): ('I7', '同形合并基准支根推荐'),
    ('I10', 'H9'): ('G10', '同形合并对应支按 I7→G10 变换'),
}
# 人工确认的同形合并对应关系；第二支旧候选 J9、H10、H11 已被替换，其中 H11
# 另按任务要求删除，不得恢复。
HUMAN_WHITE_FOUR_MERGE = {
    'canonical': ('H8', 'I9', 'G8'),
    'equivalent': ('H8', 'I10', 'H9'),
    'mapping': {'I7': 'G10', 'I8': 'H10', 'F8': 'H7'},
    'removed': ('J9', 'H11'),
    'dropped_branch': ('I10', 'G9', 'H11'),
}

COLUMNS = 'ABCDEFGHIJKLMNO'


def label(point):
    r, c = point
    return '%s%d' % (COLUMNS[c], r + 1)


def point_from_label(name):
    if len(name) < 2 or name[0] not in COLUMNS:
        raise SystemExit('无法解析坐标标签：%r' % (name,))
    r = int(name[1:]) - 1
    c = COLUMNS.index(name[0])
    if not (0 <= r < SIZE):
        raise SystemExit('坐标标签越界：%r' % (name,))
    return r, c


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    if not path.exists():
        raise SystemExit('缺少输入文件：%s' % path.relative_to(ROOT))
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def teacher_identity(record, origin):
    """返回可核验的教师身份摘要；口径不一致直接失败。"""
    identity = record.get('identity') or record.get('teacher_identity') or record.get('teacher')
    if not identity:
        raise SystemExit('%s 缺少教师身份' % origin)
    if identity.get('adapter') != 'rapfi-normal-v1':
        raise SystemExit('%s 教师适配器身份不符：%r' % (origin, identity.get('adapter')))
    if identity.get('rule') != 'freestyle15':
        raise SystemExit('%s 教师规则身份不符：%r' % (origin, identity.get('rule')))
    if (identity.get('threads'), identity.get('nodes'), identity.get('milliseconds')) != (1, 200000, 2000):
        raise SystemExit('%s 教师搜索口径不符' % origin)
    files = identity.get('files') or {}
    return {
        'adapter': identity['adapter'],
        'rule': identity['rule'],
        'threads': identity['threads'],
        'nodes': identity['nodes'],
        'milliseconds': identity['milliseconds'],
        'files_sha256': {name: files[name] for name in sorted(files)},
    }


def board_from_history(history):
    board = [[0] * SIZE for _ in range(SIZE)]
    for r, c, player in history:
        if not (0 <= r < SIZE and 0 <= c < SIZE):
            raise SystemExit('历史着法越界：%r' % ((r, c),))
        if board[r][c] != 0:
            raise SystemExit('历史着法重复占点：%r' % ((r, c),))
        board[r][c] = player
    return board


def canonical_entry(board, player, move):
    """把 (盘面, 行棋方, 落点) 压成完整盘面键 + 该键坐标系下的落点。"""
    key, symmetry = canonical_position(board, player)
    return key, transform_point(move[0], move[1], symmetry)


class LabelCache:
    """跨实验共享的原始教师回复；用于核对身份、评分视角与标签校验和。"""

    def __init__(self, identity):
        self.identity = identity
        self._cache = {}

    def get(self, key, origin):
        if key in self._cache:
            return self._cache[key]
        for directory in LABEL_DIRS:
            path = EXP / directory / 'labels' / ('%s.json' % key)
            if not path.exists():
                continue
            record = load_json(path)
            if record.get('key') != key:
                raise SystemExit('%s 标签校验和与内容不符：%s' % (origin, key))
            if teacher_identity(record, origin) != self.identity:
                raise SystemExit('%s 标签教师身份与主身份不符：%s' % (origin, key))
            self._cache[key] = record
            return record
        raise SystemExit('%s 引用的标签不存在：%s' % (origin, key))


def raw_magnitude(raw):
    text = str(raw).replace('+', '').replace('-', '').lstrip('M')
    return int(text)


def parse_black_three_scores(labels):
    """按白二类读取黑三逐点评分（黑方视角）。"""
    cache = {}
    scores = {}
    for white_two, directory in BLACK_THREE_DIRS.items():
        if directory not in cache:
            cache[directory] = load_json(EXP / directory / 'result.json')
        data = cache[directory]
        if white_two in data.get('cases', {}):
            case = data['cases'][white_two]
            root = case['root']
            scores[(white_two, '__root__')] = (tuple(root['move']),
                                              int(root['evaluation']['raw']))
            rows = case['results']
        else:
            root = data['root']
            scores[(white_two, '__root__')] = (tuple(root['move']),
                                              int(root['evaluation']['raw']))
            rows = data['results']
        for row in rows:
            if row.get('black_eval') is None:
                continue
            point = tuple(row['point']) if row.get('point') else point_from_label(row['move'])
            value = int(row['black_eval'])
            origin = '黑三 %s/%s' % (white_two, row['move'])
            if row.get('label_key'):
                record = labels.get(row['label_key'], origin)
                if raw_magnitude(record['evaluation']['raw']) != abs(value):
                    raise SystemExit('%s 标签评分与候选评分不一致' % origin)
            scores[(white_two, row['move'])] = (point, value)
    return scores


def build_labels_index():
    """先建立可核验的教师身份，再供标签缓存复用。"""
    identity = teacher_identity(load_json(EXP / WHITE_FOUR_DIR / 'identity.json'),
                               '白四全候选身份')
    for directory in (BLACK_FIVE_DIR, WHITE_FOUR_DIR, WHITE_FOUR_MERGE_DIR,
                      FIVE_CLASS_DIR, KNIGHT_DIR, FAR_DIAGONAL_DIR):
        path = EXP / directory / 'identity.json'
        origin = '%s/identity.json' % directory
        if path.exists():
            other = teacher_identity(load_json(path), origin)
        else:
            result = load_json(EXP / directory / 'result.json')
            other = teacher_identity(result, '%s/result.json' % directory)
        if other != identity:
            raise SystemExit('%s 教师身份与主身份不一致' % directory)
        result = EXP / directory / 'result.json'
        if result.exists():
            data = load_json(result)
            for holder in (data, data.get('teacher') or {}):
                if isinstance(holder, dict) and holder.get('rule'):
                    if teacher_identity({'identity': holder}, '%s/result.json' % directory) != identity:
                        raise SystemExit('%s 结果内教师身份与主身份不一致' % directory)
    return identity


def add_canonical(index, key, point, entry, origin, conflicts):
    """按已规范化的盘面键与落点写入；同键同点必须一致，否则记录冲突。"""
    if not (0 <= point[0] < SIZE and 0 <= point[1] < SIZE):
        raise SystemExit('%s：规范落点越界 %r' % (origin, point))
    if point[0] * SIZE + point[1] in key[1] or point[0] * SIZE + point[1] in key[2]:
        raise SystemExit('%s：规范落点已占 %r' % (origin, point))
    bucket = index.setdefault(key, {})
    if point in bucket:
        if bucket[point][0] != entry:
            conflicts.append((origin, key, point, bucket[point][0], entry))
        return
    bucket[point] = (entry, origin)


def add(index, board, player, move, kind, score, score_kind, view, source, origin, conflicts):
    """写入一条候选；同一盘面键同一落点必须一致，否则记录冲突。"""
    if not (0 <= move[0] < SIZE and 0 <= move[1] < SIZE):
        raise SystemExit('%s：落点越界 %r' % (origin, move))
    if board[move[0]][move[1]] != 0:
        raise SystemExit('%s：落点已占 %r' % (origin, move))
    key, point = canonical_entry(board, player, move)
    add_canonical(index, key, point,
                  (kind, score, score_kind, view, source), origin, conflicts)


def build():
    black_five = load_json(EXP / BLACK_FIVE_DIR / 'filtered-top3-or-over400.json')
    white_four = load_json(EXP / WHITE_FOUR_DIR / 'result.json')
    merge = load_json(EXP / WHITE_FOUR_MERGE_DIR / 'result.json')
    identity = build_labels_index()
    labels = LabelCache(identity)
    black_three_scores = parse_black_three_scores(labels)

    # ---- 输入数量核对 ----
    if black_five.get('groups') != 74 or len(black_five['cases']) != 74:
        raise SystemExit('黑五分组数不是 74：%r' % (black_five.get('groups'),))
    if black_five.get('retained_candidates') != 429:
        raise SystemExit('黑五保留候选数不是 429：%r' % (black_five.get('retained_candidates'),))
    if black_five.get('original_candidates') != 820:
        raise SystemExit('黑五原始候选数不是 820：%r' % (black_five.get('original_candidates'),))
    if sum(len(case['candidates']) for case in black_five['cases']) != 429:
        raise SystemExit('黑五实际候选数不是 429')
    if black_five.get('formal_engine_changed'):
        raise SystemExit('输入自述已改动正式引擎，身份不符')

    completed = EXP / BLACK_FIVE_DIR / 'completed-result.json'
    if not completed.exists():
        raise SystemExit('缺少黑五完整结果，无法核对来源摘要')
    if sha256_file(completed) != black_five.get('source_sha256'):
        raise SystemExit('黑五完整结果 SHA256 与筛选记录不符')

    if sum(len(rows) for _cls, *rows in CONFIRMED_BLACK_THREE) != 27:
        raise SystemExit('黑三确认清单不是 27 条')

    for white_two, move, value in EXCLUDED_BLACK_THREE:
        scored = black_three_scores.get((white_two, move))
        if scored is None or scored[1] != value:
            raise SystemExit('已移出的低分项与输入不符：%r' % ((white_two, move),))
        if value >= 400:
            raise SystemExit('被移出项分数并未低于 400：%r' % ((white_two, move),))
    for white_two, *rows in CONFIRMED_BLACK_THREE:
        for move, scored_flag in rows:
            if any((white_two, move) == (w, m) for w, m, _v in EXCLUDED_BLACK_THREE):
                raise SystemExit('低分项混入确认清单：%r' % ((white_two, move),))
            present = (white_two, move) in black_three_scores
            if scored_flag and not present:
                raise SystemExit('确认候选缺少逐点评分：%r' % ((white_two, move),))
            if scored_flag and black_three_scores[(white_two, move)][1] < 400:
                raise SystemExit('确认候选低于 400：%r' % ((white_two, move),))
            if not scored_flag and present:
                raise SystemExit('待评候选其实已有同口径评分：%r' % ((white_two, move),))

    # ---- 白四：75 条全候选，最终确认 74 条 ----
    four_candidates = []
    for case in white_four['cases']:
        for row in case['candidates']:
            point = tuple(row['point']) if row.get('point') else point_from_label(row['move'])
            four_candidates.append((case['white_two'], case['black_three'], row, point))
    if len(four_candidates) != 75:
        raise SystemExit('白四全候选不是 75 条：%r' % (len(four_candidates),))
    final_four = {(case['white_two'], case['black_three'], case['white_four'])
                  for case in black_five['cases']}
    all_four = {(w2, b3, row['move']) for w2, b3, row, _p in four_candidates}
    missing = all_four - final_four
    extra = final_four - all_four
    if extra:
        raise SystemExit('最终白四出现全候选里没有的分支：%r' % (sorted(extra),))
    if missing != {HUMAN_WHITE_FOUR_MERGE['dropped_branch']}:
        raise SystemExit('白四删除项与任务要求不符：%r' % (sorted(missing),))

    # 同形合并：第二支旧候选 J9、H11 不得恢复。
    merged_branch = {name for w2, b3, name in final_four if (w2, b3) == ('I10', 'H9')}
    if merged_branch & set(HUMAN_WHITE_FOUR_MERGE['removed']):
        raise SystemExit('同形合并第二支的旧候选被恢复：%r' % (sorted(merged_branch),))
    expected_merged = {name for name in HUMAN_WHITE_FOUR_MERGE['mapping'].values()}
    if not expected_merged <= merged_branch:
        raise SystemExit('同形合并对应支缺少变换后的候选：%r' % (sorted(merged_branch),))
    if merge['canonical_history'] != list(HUMAN_WHITE_FOUR_MERGE['canonical']):
        raise SystemExit('同形合并基准支不符：%r' % (merge['canonical_history'],))
    if merge['equivalent_history'] != list(HUMAN_WHITE_FOUR_MERGE['equivalent']):
        raise SystemExit('同形合并对应支不符：%r' % (merge['equivalent_history'],))
    if merge['mapping'] != HUMAN_WHITE_FOUR_MERGE['mapping']:
        raise SystemExit('同形合并坐标映射不符：%r' % (merge['mapping'],))

    # ---- 组装索引 ----
    index = {}
    conflicts = []
    report_rows = []

    # 黑一：空盘固定 H8。
    empty = board_from_history(())
    add(index, empty, BLACK, (7, 7), 'primary', None, None, None, 'rule:h8-first',
        '黑一 H8', conflicts)
    report_rows.append((1, '黑一', '空盘', 'H8', '人工规则', '无评分', '固定天元'))

    # 白二：沿用既有正邻/斜邻选择方式，取自打包推荐与备选记录。
    black_one = board_from_history(((7, 7, BLACK),))
    h8_cell = 7 * SIZE + 7
    primary_two = [row for row in RECORDS
                   if row[0] == WHITE and row[1] == (h8_cell,) and row[2] == ()]
    if len(primary_two) != 1:
        raise SystemExit('打包白二主着记录不唯一')
    alt_two = [row for row in ALTERNATIVES
               if row[0] == WHITE and row[1] == (h8_cell,) and row[2] == ()]
    if len(alt_two) != 1:
        raise SystemExit('打包白二备选记录不唯一')
    adjacent = ((primary_two[0][3], primary_two[0][4]), (alt_two[0][3], alt_two[0][4]))
    if adjacent[0] == adjacent[1]:
        raise SystemExit('白二正邻/斜邻两个方向相同')
    # 记录存的是规范坐标系下的落点，按相对 H8 的偏移核对方向族：
    # 正邻为 (±1, 0)，斜邻为 (±1, ±1)，都只差一格且不重合。
    offsets = [(point[0] - 7, point[1] - 7) for point in adjacent]
    if sorted(abs(offset[0]) + abs(offset[1]) for offset in offsets) != [1, 2]:
        raise SystemExit('白二两个方向不是一格邻接：%r' % (offsets,))
    if max(abs(offsets[0][0]), abs(offsets[0][1])) != 1 or max(abs(offsets[1][0]), abs(offsets[1][1])) != 1:
        raise SystemExit('白二方向超出相邻格：%r' % (offsets,))
    if offsets[0][0] and offsets[0][1]:
        raise SystemExit('白二主着不是正邻：%r' % (offsets[0],))
    if not (offsets[1][0] and offsets[1][1]):
        raise SystemExit('白二备选不是斜邻：%r' % (offsets[1],))
    for point, family in zip(adjacent, ('正邻', '斜邻')):
        add(index, black_one, WHITE, point, 'policy', None, None, None,
            'human:adjacent-two', '白二%s %s' % (family, label(point)), conflicts)
    report_rows.append((2, '白二', 'H8 后', '正邻 / 斜邻（相对天元一格正交、一格对角）',
                        '人工策略（正邻/斜邻）', '无评分',
                        '每局八种视角前半取正邻、后半取斜邻；不新增随机开局策略'))

    # 黑三：27 条确认候选，按白二五类分别建键。
    confirmed_rows = {rows[0]: rows[1:] for rows in CONFIRMED_BLACK_THREE}
    for white_two, offset, selectable, class_name in WHITE_TWO_CLASSES:
        board = board_from_history(((7, 7, BLACK), (offset[0], offset[1], WHITE)))
        if board[offset[0]][offset[1]] != WHITE:
            raise SystemExit('白二落点未生效：%r' % (white_two,))
        rows = confirmed_rows[white_two]
        scored = []
        for move, scored_flag in rows:
            if scored_flag:
                point, value = black_three_scores[(white_two, move)]
            else:
                # 待评点只有根推荐，没有同口径逐点评分：保留覆盖，分数记 None。
                point, value = point_from_label(move), None
            scored.append((move, point, value))
        if (white_two, rows[0][0]) in HUMAN_BLACK_THREE:
            primary_name = rows[0][0]
        else:
            rated = [row for row in scored if row[2] is not None]
            if not rated:
                raise SystemExit('黑三整组无评分，无法确定暂定主着：%r' % (white_two,))
            # 同分保持稳定顺序：只在严格更高时替换，取原清单中最先出现者。
            primary_name, best_value = rated[0][0], rated[0][2]
            for move, _point, value in rated[1:]:
                if value > best_value:
                    primary_name, best_value = move, value
        for move, point, value in scored:
            if move == primary_name:
                kind = 'primary'
                source = ('human:black-three' if (white_two, move) in HUMAN_BLACK_THREE
                          else 'teacher:black-three')
            elif value is None:
                kind = 'unrated'
                source = 'teacher:black-three-root'
            else:
                kind = 'coverage'
                source = 'teacher:black-three'
            add(index, board, BLACK, point, kind, value,
                ('cp' if value is not None else None),
                ('black' if value is not None else None), source,
                '黑三 %s/%s(%s)' % (white_two, move, class_name), conflicts)
        primary_value = [v for m, _p, v in scored if m == primary_name][0]
        report_rows.append((
            3, '黑三', '白二 %s（%s）' % (white_two, class_name), primary_name,
            '人工指定' if (white_two, primary_name) in HUMAN_BLACK_THREE else '逐点评分最高',
            '未评分' if primary_value is None else '%d（黑方视角）' % primary_value,
            '%s；待评：%s' % ('、'.join(m for m, _p, _v in scored),
                            '、'.join(m for m, _p, v in scored if v is None) or '无')))

    # 白四 + 黑五：直接采用最终 74 条 history 作为白四覆盖依据。
    four_scores = {(w2, b3, row['move']): (row, point)
                   for w2, b3, row, point in four_candidates}
    five_cases = []
    seen_four_pairs = set()
    for case in black_five['cases']:
        white_two, black_three = case['white_two'], case['black_three']
        history = tuple(tuple(cell) for cell in case['history'])
        if len(history) != 4:
            raise SystemExit('黑五输入不是完整前四手：%r' % (history,))
        board_three = board_from_history(history[:3])
        board_four = board_from_history(history)
        four_move = history[3]
        if label((four_move[0], four_move[1])) != case['white_four']:
            raise SystemExit('白四坐标与历史不符：%r' % (case,))
        branch = sorted(name for (w2, b3, name) in final_four
                        if (w2, b3) == (white_two, black_three))
        if case['white_four'] not in branch:
            raise SystemExit('白四分支不在候选集合内：%r' % (case['white_four'],))

        # 白四主着按整组候选决定：人工基准/对应支优先，否则取同口径白方视角最高分。
        # 同组的多个分支共享同一张 3 子盘面，主着只标记一次，不随后写分支变化。
        designated = HUMAN_WHITE_FOUR.get((white_two, black_three))
        if designated is not None:
            if designated[0] not in branch:
                raise SystemExit('人工首选不在候选集合内：%r' % ((white_two, black_three),))
            primary_name = designated[0]
        else:
            rated = [(name, four_scores[(white_two, black_three, name)][0]['white_eval'])
                     for name in branch
                     if four_scores[(white_two, black_three, name)][0]['white_eval'] is not None]
            if not rated:
                raise SystemExit('白四整组无评分，无法确定暂定主着：%r'
                                 % ((white_two, black_three),))
            primary_name = rated[0][0]
            best_value = rated[0][1]
            for name, value in rated[1:]:
                if value > best_value:
                    primary_name, best_value = name, value

        pair_name = (white_two, black_three)
        if pair_name not in seen_four_pairs:
            seen_four_pairs.add(pair_name)
            primary_eval = four_scores[(white_two, black_three, primary_name)][0]['white_eval']
            report_rows.append((
                4, '白四', '白二 %s / 黑三 %s' % (white_two, black_three), primary_name,
                '人工合并首选' if designated is not None else '逐点评分最高',
                '未评分' if primary_eval is None else '%d（白方视角）' % primary_eval,
                '候选：%s；待评：%s' % (
                    '、'.join(branch),
                    '、'.join(n for n in branch
                             if four_scores[(white_two, black_three, n)][0]['white_eval'] is None) or '无')))

        for name in branch:
            row, point = four_scores[(white_two, black_three, name)]
            origin = '白四 %s/%s/%s' % (white_two, black_three, name)
            if row.get('label_key'):
                record = labels.get(row['label_key'], origin)
                check = row.get('black_eval_raw')
                if check and raw_magnitude(record['evaluation']['raw']) != raw_magnitude(check['raw']):
                    raise SystemExit('%s 标签评分与候选评分不一致' % origin)
            if name == primary_name:
                kind = 'primary'
                source = ('human:white-four-merge' if designated is not None
                          else 'teacher:white-four')
            elif row['white_eval'] is None:
                kind = 'unrated'
                source = 'teacher:white-four'
            else:
                kind = 'coverage'
                source = 'teacher:white-four'
            add(index, board_three, WHITE, point, kind, row['white_eval'],
                ('cp' if row['white_eval'] is not None else None),
                ('white' if row['white_eval'] is not None else None), source,
                origin, conflicts)

        # 黑五：先收集；八种旋转/镜像可能把多个绝对前四手并到同一规范盘面，
        # 必须统一归并后再定主着，不允许后写覆盖前写。
        candidates = case['candidates']
        if len(candidates) < 3:
            raise SystemExit('黑五候选少于保底前三：%r' % (case['white_four'],))
        origin = '黑五 %s/%s/%s' % (white_two, black_three, case['white_four'])
        parsed = []
        for position, row in enumerate(candidates):
            move = tuple(row['point'])
            if label(move) != row['move']:
                raise SystemExit('%s 第%d条坐标与标签不符' % (origin, position))
            evaluation = row['evaluation_black']
            if evaluation['kind'] == 'cp':
                value, score_kind = int(evaluation['raw']), 'cp'
            elif evaluation['kind'] == 'mate':
                value, score_kind = raw_magnitude(evaluation['raw']), 'mate'
            else:
                raise SystemExit('未知黑五评分类型：%r' % (evaluation['kind'],))
            record = labels.get(row['label_key'], origin)
            if raw_magnitude(record['evaluation']['raw']) != value:
                raise SystemExit('%s 标签评分与候选评分不一致' % origin)
            if score_kind == 'cp' and int(record['evaluation']['raw']) != -value:
                raise SystemExit('%s 标签评分视角不是黑方取反' % origin)
            parsed.append((move, value, score_kind, position == 0))
        five_cases.append((origin, board_four, parsed))

    # 黑五归并：八种旋转/镜像会把多个绝对前四手并到同一规范盘面。
    # 先各自规范化，再按规范键取并集，主着在并集上按筛选规则重算。
    # 同一规范点若被两组给了不同评分，保留文件顺序在先者并记录冲突，
    # 不允许后写覆盖前写，也不静默丢弃。
    five_groups = {}
    five_conflicts = []
    five_duplicates = 0
    for origin, board_four, parsed in five_cases:
        for move, value, score_kind, is_top in parsed:
            key_four, point = canonical_entry(board_four, BLACK, move)
            group = five_groups.setdefault(key_four, {'entries': {}, 'tops': [], 'origins': []})
            if origin not in group['origins']:
                group['origins'].append(origin)
            entry = ('coverage', value, score_kind, 'black', 'teacher:black-five')
            previous = group['entries'].get(point)
            if previous is not None:
                if previous[1:] != entry[1:]:
                    five_conflicts.append((origin, key_four, point, previous, entry))
                else:
                    five_duplicates += 1
                continue
            group['entries'][point] = entry
            if is_top:
                group['tops'].append(point)
    collisions = sorted(tuple(sorted(group['origins'])) for group in five_groups.values()
                        if len(group['origins']) > 1)
    for key_four, group in five_groups.items():
        # 主着规则：正杀棋优先于普通分；同为杀棋时步数少者更强，普通分则高者更强；
        # 再按规范坐标稳定排序。同一规范盘面合并后仍按并集重算，不随后写变化。
        primary = min(group['entries'],
                      key=lambda p: (0 if group['entries'][p][2] == 'mate' else 1,
                                     group['entries'][p][1] if group['entries'][p][2] == 'mate'
                                     else -group['entries'][p][1], p))
        group['entries'][primary] = ('primary',) + group['entries'][primary][1:]
        origin = group['origins'][0]
        for point, entry in group['entries'].items():
            add_canonical(index, key_four, point, entry, origin, conflicts)

    # 黑五主着按各绝对前四手回写坐标，便于逐组核对。
    for origin, board_four, parsed in five_cases:
        key_four = canonical_entry(board_four, BLACK, parsed[0][0])[0]
        group = five_groups[key_four]
        primary_point = [p for p, entry in group['entries'].items() if entry[0] == 'primary'][0]
        symmetry = canonical_position(board_four, BLACK)[1]
        absolute = inverse_point(primary_point[0], primary_point[1], symmetry)
        entry = group['entries'][primary_point]
        merged = '（与 %d 组同形合并）' % (len(group['origins']) - 1) if len(group['origins']) > 1 else ''
        report_rows.append((
            5, '黑五', origin[len('黑五 '):], label(absolute),
            '杀棋标签优先' if entry[2] == 'mate' else '逐点评分最高',
            '%d（黑方视角）' % entry[1],
            '本组输入 %d 条%s' % (len(parsed), merged)))

    counts = {'黑三': 0, '白四': 0, '黑五': 0}
    black_three_sources = ('teacher:black-three', 'teacher:black-three-root',
                           'human:black-three')
    white_four_sources = ('teacher:white-four', 'human:white-four-merge')
    for bucket in index.values():
        for entry, _origin in bucket.values():
            if entry[4] in black_three_sources:
                counts['黑三'] += 1
            elif entry[4] in white_four_sources:
                counts['白四'] += 1
            elif entry[4] == 'teacher:black-five':
                counts['黑五'] += 1
    five_rows = sum(len(parsed) for _o, _b, parsed in five_cases)
    if counts['黑三'] != 27 or counts['白四'] != 74:
        raise SystemExit('黑三/白四确认条数与任务要求不符：%r' % (counts,))
    if five_rows != 429:
        raise SystemExit('黑五输入条数不是 429：%r' % (five_rows,))
    if counts['黑五'] != sum(len(group['entries']) for group in five_groups.values()):
        raise SystemExit('黑五写入条数与归并结果不符：%r' % (counts['黑五'],))
    if five_rows != counts['黑五'] + five_duplicates + len(five_conflicts):
        raise SystemExit('黑五归并账目不平：输入 %d，唯一 %d，重复 %d，冲突 %d' % (
            five_rows, counts['黑五'], five_duplicates, len(five_conflicts)))
    if conflicts:
        for item in conflicts[:10]:
            print('冲突：', item)
        raise SystemExit('同一完整盘面键出现不同候选，共 %d 处' % len(conflicts))

    # ---- 去重与盘面统计 ----
    stats = {}
    for key in index:
        stones = len(key[1]) + len(key[2])
        stats[stones] = stats.get(stones, 0) + 1
    for bucket in index.values():
        primaries = [point for point, (entry, _o) in bucket.items() if entry[0] == 'primary']
        if len(primaries) > 1:
            raise SystemExit('同一盘面出现多个主着：%r' % (primaries,))

    records = []
    order = {'primary': 0, 'policy': 0, 'coverage': 1, 'unrated': 2}
    for key in sorted(index, key=lambda k: (len(k[1]) + len(k[2]), k[0], k[1], k[2])):
        bucket = index[key]
        for point in sorted(bucket, key=lambda p: (order[bucket[p][0][0]],
                                                   -(bucket[p][0][1] or 0), p)):
            entry, _origin = bucket[point]
            records.append((key[0], key[1], key[2], point[0], point[1]) + entry)

    provenance = {
        'rule': 'freestyle15',
        'scope': '整盘第1-5手（双方合计）；第6手起不接管',
        'first_move': 'H8',
        'positions': len(index),
        'candidate_moves': len(records),
        'primary_moves': sum(1 for row in records if row[5] == 'primary'),
        'policy_moves': sum(1 for row in records if row[5] == 'policy'),
        'unrated_moves': sum(1 for row in records if row[5] == 'unrated'),
        'positions_by_stones': dict(sorted(stats.items())),
        'white_two_classes': {name: {'offset': list(offset), 'selectable': selectable,
                                     'family': family}
                              for name, offset, selectable, family in WHITE_TWO_CLASSES},
        'black_three_confirmed': 27,
        'white_four_full_candidates': 75,
        'white_four_confirmed': 74,
        'white_four_dropped': '%s/%s/%s' % HUMAN_WHITE_FOUR_MERGE['dropped_branch'],
        'black_five_original': 820,
        'black_five_confirmed': 429,
        'black_five_unique': sum(len(group['entries']) for group in five_groups.values()),
        'black_five_duplicates': five_duplicates,
        'black_five_merged_positions': [list(item) for item in collisions],
        'black_five_score_conflicts': [
            {'origin': origin, 'point': list(point),
             'kept': [previous[1], previous[2]], 'ignored': [entry[1], entry[2]]}
            for origin, _key, point, previous, entry in five_conflicts],
        'teacher': identity,
        'python': sys.version.split()[0],
        'sources_sha256': {
            'black-five': sha256_file(EXP / BLACK_FIVE_DIR / 'filtered-top3-or-over400.json'),
            'black-five-completed': sha256_file(completed),
            'white-four': sha256_file(EXP / WHITE_FOUR_DIR / 'result.json'),
            'white-four-identity': sha256_file(EXP / WHITE_FOUR_DIR / 'identity.json'),
            'white-four-merge': sha256_file(EXP / WHITE_FOUR_MERGE_DIR / 'result.json'),
            'black-three-five-class': sha256_file(EXP / FIVE_CLASS_DIR / 'result.json'),
            'black-three-knight': sha256_file(EXP / KNIGHT_DIR / 'result.json'),
            'black-three-far-diagonal': sha256_file(EXP / FAR_DIAGONAL_DIR / 'result.json'),
        },
        'proven': False,
    }
    return provenance, records, report_rows


def render(provenance, records):
    lines = ['# -*- coding: utf-8 -*-',
             '"""整盘前5手稀疏开局库的纯数据清单；由 scripts/build_opening_front_book.py 生成。',
             '',
             '每条记录：(行棋方, 黑子, 白子, 行, 列, 种类, 分数, 分数类型, 视角, 来源)。',
             '种类：primary=暂定主着；policy=白二正邻/斜邻的每局视角策略；',
             'coverage=已评分保留候选；unrated=待评候选，不得当作最高分或零分。',
             '本清单只作用于整盘第1-5手；第6手起不再接管，出谱立即交回原候选与深推。',
             '"""',
             'PROVENANCE = %r' % (provenance,),
             '',
             'RECORDS = (']
    for row in records:
        lines.append('    %r,' % (row,))
    lines.append(')')
    lines.append('')
    return '\n'.join(lines)


def render_report(provenance, report_rows):
    lines = ['# 整盘前5手稀疏开局库：暂定主着汇总', '',
             '- 盘面 %d 个，候选 %d 条，主着 %d 条，白二策略 %d 条，待评 %d 条' % (
                 provenance['positions'], provenance['candidate_moves'],
                 provenance['primary_moves'], provenance['policy_moves'],
                 provenance['unrated_moves']),
             '- 黑五输入 %d 条，八种旋转/镜像去重后唯一 %d 条，重复 %d 条，同形合并 %d 处，'
             '同点评分冲突 %d 处' % (
                 provenance['black_five_confirmed'], provenance['black_five_unique'],
                 provenance['black_five_duplicates'], len(provenance['black_five_merged_positions']),
                 len(provenance['black_five_score_conflicts'])),
             '- 主着依据不构成五态硬证明，也未做棋力对照。', '',
             '| 手数 | 阶段 | 局面 | 暂定主着 | 依据 | 评分 | 备注 |',
             '|---|---|---|---|---|---|---|']
    for row in report_rows:
        lines.append('| %d | %s | %s | %s | %s | %s | %s |' % row)
    if provenance['black_five_merged_positions']:
        lines += ['', '## 黑五同形合并', '']
        for item in provenance['black_five_merged_positions']:
            lines.append('- ' + ' ↔ '.join(item))
    if provenance['black_five_score_conflicts']:
        lines += ['', '## 黑五同点评分冲突（保留先写，未覆盖）', '']
        for item in provenance['black_five_score_conflicts']:
            lines.append('- %s 规范点 %r：保留 %r，忽略 %r，来源 %s' % (
                '', item['point'], item['kept'], item['ignored'], item['origin']))
    lines.append('')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='生成整盘前5手稀疏开局库数据清单')
    parser.add_argument('--check', action='store_true', help='只校验并打印摘要，不写文件')
    parser.add_argument('--report', metavar='PATH', help='把暂定主着汇总表写入指定 markdown 文件')
    args = parser.parse_args()

    provenance, records, report_rows = build()
    text = render(provenance, records)
    if args.check:
        if not OUTPUT.exists():
            raise SystemExit('数据文件尚不存在：%s' % OUTPUT.relative_to(ROOT))
        if OUTPUT.read_text(encoding='utf-8') != text:
            raise SystemExit('数据文件与生成结果不一致，需要重新生成')
        print('数据文件与生成结果一致')
    else:
        OUTPUT.write_text(text, encoding='utf-8')
        print('已写入 %s' % OUTPUT.relative_to(ROOT))
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(render_report(provenance, report_rows), encoding='utf-8')
        print('已写入汇总表 %s' % args.report)
    print('盘面 %d 个，候选 %d 条，主着 %d 条，白二策略 %d 条，待评 %d 条' % (
        provenance['positions'], provenance['candidate_moves'],
        provenance['primary_moves'], provenance['policy_moves'],
        provenance['unrated_moves']))
    print('按石子数分布：', provenance['positions_by_stones'])
    print('黑五：输入 %d，唯一 %d，重复 %d，同形合并 %d 处，同点评分冲突 %d 处' % (
        provenance['black_five_confirmed'], provenance['black_five_unique'],
        provenance['black_five_duplicates'], len(provenance['black_five_merged_positions']),
        len(provenance['black_five_score_conflicts'])))
    return 0


if __name__ == '__main__':
    sys.exit(main())
