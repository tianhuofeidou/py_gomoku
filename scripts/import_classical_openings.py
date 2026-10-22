# -*- coding: utf-8 -*-
"""离线整理人类棋谱：明确推荐供接管，未评级研究线仅供根候选。

只读取用户提供的 RenLib 3.0 数据及人工整理清单，不下载、运行外部程序，
不使用求解器评价选主着。输出必须显式指定，不会自动覆盖正式库。
格式参考：https://github.com/gomoku/RenLib （LibraryFile/MoveNode/RenLibDoc）。
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gomoku.core.opening_book import canonical_position, transform_point
from gomoku.game import check_win


def read_library(data):
    """解析严格的 3.0 二进制前序树；保留注释节点，拒绝截断和非法回退。

    DOWN 标记之后还有兄弟，RIGHT 结束当前分支；两者不是子节点数量。
    字符串按两个字节读取，终止符之后的同一对填充字节也必须消耗。
    pass 和非落子节点可被解析，但不作为正式棋盘路径导入。
    """
    if len(data) < 20 or data[:10] != b'\xffRenLib\xff\x03\0':
        raise ValueError('expected RenLib 3.0 header')
    offset = 20
    nodes, path, forks = [], [], []

    def pair():
        nonlocal offset
        if offset + 2 > len(data):
            raise ValueError('truncated pair at %d' % offset)
        value = data[offset:offset + 2]
        offset += 2
        return value

    def read_text(old=False):
        value = bytearray()
        while True:
            for byte in pair():
                if old:
                    value.append(byte & 127)
                    if byte & 128:
                        return bytes(value)
                elif byte == 0:
                    return bytes(value)
                else:
                    value.append(byte)

    while offset < len(data):
        start = offset
        pos, flags = pair()
        info = flags
        if flags & 1:
            high, low = pair()
            info |= high << 16 | low << 8
        comment = read_text(bool(flags & 32)) if flags & 40 else b''
        label = read_text() if info & 256 else b''
        point = (pos // 16, pos % 16 - 1) if pos else None
        if point and not all(0 <= x < 15 for x in point):
            raise ValueError('invalid point at %d' % start)
        root = not nodes and point is None
        node = {'id': len(nodes), 'offset': start, 'point': point, 'info': info,
                'comment': comment, 'label': label,
                'parent': path[-1] if path else None}
        nodes.append(node)
        if not root:
            path.append(node['id'])
        if flags & 128:
            forks.append(len(path))
        if flags & 64:
            if forks:
                path = path[:forks.pop() - 1]
            elif offset != len(data):
                raise ValueError('branch end without fork at %d' % start)
    if forks:
        raise ValueError('unclosed branches')
    return nodes


def history_board(history):
    """恢复有序、合法且未终局的前缀；连珠禁手结论不参与判定。"""
    board = [[0] * 15 for _ in range(15)]
    for index, point in enumerate(history):
        if point is None or len(point) != 2 or not all(type(x) is int and 0 <= x < 15 for x in point):
            raise ValueError('unsupported or invalid move')
        r, c = point
        if board[r][c]:
            raise ValueError('occupied move')
        player = index % 2 + 1
        board[r][c] = player
        if check_win(board, r, c, player):
            raise ValueError('terminal history')
    return board


def opening_record(history, move):
    board = history_board(history)
    r, c = move
    if not (0 <= r < 15 and 0 <= c < 15) or board[r][c]:
        raise ValueError('invalid recommendation')
    key, symmetry = canonical_position(board, len(history) % 2 + 1)
    return (*key, *transform_point(r, c, symmetry))


NEGATIVE = re.compile(r'\?|\b(?:bad|wrong|lose|loss|lousy|no win|not good|weak)\b|弱|败|错误', re.I)
POSITIVE = re.compile(r'\b(?:best|strongest|only|good|right|normal|need)\b|最强|正招|第一胜点', re.I)
FOUL = re.compile(r'禁手|三三|四四|长连|\b(?:foul|overline|forbidden)\b', re.I)


def study_candidates(nodes, min_stones=6, max_stones=11):
    """不把树的遍历顺序当推荐次序：只取明确正向注释或唯一未标坏的续着。

    唯一续着不等于最强，返回的数据只能用于本方第四手以后的根候选深推。
    多个未评级续着留空；包含禁手依赖注释、pass/标记或终局的前缀整支跳过。
    """
    histories, safe, children = {}, {}, defaultdict(list)
    for node in nodes:
        parent = node['parent']
        text = node['comment'].decode('gb18030', errors='replace')
        history = histories.get(parent, ())
        root = node['id'] == 0 and node['point'] is None
        histories[node['id']] = history if root else history + (node['point'],)
        safe[node['id']] = (safe.get(parent, True) and node['point'] is not None
                            and not node['info'] & 2 and not FOUL.search(text))
        if root:
            safe[node['id']] = True
        if parent is not None:
            children[parent].append(node)
    for parent, variants in children.items():
        history = histories[parent]
        if not safe[parent] or not min_stones <= len(history) <= max_stones:
            continue
        rated = []
        for node in variants:
            text = node['comment'].decode('gb18030', errors='replace')
            if safe[node['id']] and not NEGATIVE.search(text):
                rated.append((node, bool(POSITIVE.search(text))))
        preferred = [node for node, positive in rated if positive]
        if preferred:
            selected, basis = min(preferred, key=lambda n: n['point']), 'explicit-comment'
        elif len(variants) == 1 and len(rated) == 1:
            selected, basis = rated[0][0], 'single-study-continuation'
        else:
            continue
        try:
            record = opening_record(history, selected['point'])
        except ValueError:
            continue
        yield record, {'node': selected['id'], 'offset': selected['offset'], 'basis': basis}


def build_book(manifest, library_dir):
    records, alternatives, candidates, evidence = {}, {}, {}, {}
    for entry in manifest['recommendations']:
        if entry['source'] not in manifest['sources'] or not entry.get('basis'):
            raise ValueError('recommendation without provenance')
        row = opening_record(entry['history'], entry['move'])
        key = row[:3]
        target = alternatives if entry.get('alternative') else records
        if key in target and target[key] != row:
            raise ValueError('conflicting human recommendations')
        target[key] = row
        evidence[str(row)] = {'source': entry['source'], 'basis': entry['basis']}
    for key, row in alternatives.items():
        if key not in records or records[key] == row:
            raise ValueError('alternative without distinct primary')
    counts = {}
    for source_id, source in manifest['sources'].items():
        if source.get('format') != 'renlib3':
            continue
        data = (library_dir / source['file']).read_bytes()
        if hashlib.sha256(data).hexdigest() != source['sha256']:
            raise ValueError('source hash mismatch: ' + source_id)
        nodes = read_library(data)
        counts[source_id] = len(nodes)
        for row, proof in study_candidates(nodes):
            key = row[:3]
            if key not in records and key not in candidates:
                candidates[key] = row
                evidence[str(row)] = dict(proof, source=source_id)
    provenance = {'rule': 'freestyle15', 'source_rule': 'renju', 'selection': 'human-curated',
                  'positions': len(records), 'alternative_moves': len(alternatives),
                  'candidate_positions': len(candidates), 'proven': False,
                  'source_nodes': counts, 'sources': manifest['sources'],
                  'recommended_by_stones': dict(sorted(Counter(len(k[1]) + len(k[2]) for k in records).items())),
                  'candidates_by_stones': dict(sorted(Counter(len(k[1]) + len(k[2]) for k in candidates).items()))}
    return records, alternatives, candidates, provenance, evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--library-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    raw = args.manifest.read_bytes()
    result = build_book(json.loads(raw), args.library_dir)
    records, alternatives, candidates, provenance, evidence = result
    provenance['manifest_sha256'] = hashlib.sha256(raw).hexdigest()
    text = '# -*- coding: utf-8 -*-\n"""人类经典应手与研究线；研究线仅作根候选，不提供胜败证明。"""\n'
    text += 'PROVENANCE = ' + repr(provenance) + '\n'
    for name, rows in [('RECORDS', records), ('ALTERNATIVES', alternatives), ('CANDIDATES', candidates)]:
        text += '\n' + name + ' = (\n'
        text += ''.join('    ' + repr(row) + ',\n' for row in sorted(rows.values()))
        text += ')\n'
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'opening_book_data.py').write_text(text, encoding='utf-8')
    (args.output_dir / 'classical_audit.json').write_text(
        json.dumps({'provenance': provenance, 'evidence': evidence}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: provenance[key] for key in
                      ('positions', 'alternative_moves', 'candidate_positions',
                       'recommended_by_stones', 'candidates_by_stones')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
