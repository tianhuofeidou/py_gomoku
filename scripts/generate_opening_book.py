# -*- coding: utf-8 -*-
"""用用户自行提供的 Rapfi 250615 离线生成开局建议；不下载、不安装、不自动启用。

最多新分析 300 个局面（保留旧库深记录另计）、每局面 20 万节点/5 秒、单线程，总预算 600 秒。
不读取评测开局集；以人类26种三手棋形及参考应手安排覆盖，按自由规则分析。
白二的正邻、斜邻两种相邻应手均保留，其余推荐着法由 Rapfi 选择。
输出 analysis.json 与纯数据模块，
人工审查生成结果后才能替换正式数据；数据代表有限搜索建议，不是必胜证明。
"""
def main():
    import argparse
    import ast, sys, json, subprocess, threading, queue, re, time, hashlib
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from gomoku.core.opening_book import canonical_position, transform_point, inverse_point
    from gomoku.game import check_win
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', required=True, help='Rapfi 250615 可执行文件，配套 config/权重放在同目录')
    parser.add_argument('--output-dir', required=True, help='分析记录与待审查数据模块的输出目录')
    parser.add_argument('--source-url', default='https://github.com/dhbloo/rapfi/releases/tag/250615')
    parser.add_argument('--seeds', type=Path, default=Path(__file__).resolve().parents[1] / 'gomoku/data/human_opening_seeds.json')
    parser.add_argument('--base-book', type=Path, default=Path(__file__).resolve().parents[1] / 'gomoku/core/opening_book_data.py', help='保留其中六子及以上的已有记录，不重新分析')
    args = parser.parse_args()
    OUT = Path(args.output_dir).resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    ENGINE = Path(args.engine).resolve()
    ROOT = ENGINE.parent
    engine_sha256 = hashlib.sha256(ENGINE.read_bytes()).hexdigest()
    weight_sha256 = hashlib.sha256((ROOT/'mix9svqfreestyle_bsmix.bin.lz4').read_bytes()).hexdigest()
    seed_bytes = args.seeds.read_bytes()
    seeds = json.loads(seed_bytes)
    if seeds['rule'] != 'freestyle15' or len(seeds['openings']) != 26:
        raise ValueError('expected 26 freestyle analysis seeds')
    NODE_LIMIT = 200000
    MAX_POSITIONS = 300
    DEADLINE = time.monotonic() + 600
    base_bytes = args.base_book.read_bytes()
    base_records = next(ast.literal_eval(node.value) for node in ast.parse(base_bytes).body
                        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'RECORDS' for t in node.targets))
    records = {(row[0], tuple(row[1]), tuple(row[2])): list(row)
               for row in base_records if len(row[1]) + len(row[2]) >= 6}
    retained_positions = len(records)
    alternatives = {}
    audit = []
    p = subprocess.Popen([str(ENGINE)], cwd=ROOT,
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding='utf-8', creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    q = queue.Queue()
    def reader():
        for line in p.stdout:
            q.put(line.strip())
        q.put(None)
    threading.Thread(target=reader, daemon=True).start()
    def send(line):
        p.stdin.write(line + '\n'); p.stdin.flush()
    def until(pattern, seconds=12):
        lines = []
        end = time.monotonic() + seconds
        while True:
            line = q.get(timeout=max(.01, end-time.monotonic()))
            if line is None:
                raise RuntimeError('Rapfi exited: '+repr(lines))
            lines.append(line)
            if line.startswith(('ERROR', 'UNKNOWN')):
                raise RuntimeError(line)
            if re.fullmatch(pattern, line):
                return line, lines
    def board_of(history):
        board = [[0]*15 for _ in range(15)]
        for i, (r,c) in enumerate(history):
            assert board[r][c] == 0
            board[r][c] = i%2+1
        return board
    def save():
        data = {'engine': 'Rapfi 250615 / 0.43.01 AVX2', 'rule': 'freestyle15',
                'node_limit': NODE_LIMIT, 'per_position_ms':5000, 'threads':1,
                'max_positions':MAX_POSITIONS, 'analysis_budget_seconds':600,
                'selection':'human 26 shapes; best plus up to two reference white-four replies; best plus one nearby black-five deviation; through white six',
                'seed_sha256':hashlib.sha256(seed_bytes).hexdigest(),
                'human_source_url':seeds['shape_source_url'],
                'base_book_sha256':hashlib.sha256(base_bytes).hexdigest(),
                'retained_positions':retained_positions,
                'records':list(records.values()), 'alternatives':list(alternatives.values()), 'audit':audit}
        data['source_url'] = args.source_url
        data['engine_sha256'] = engine_sha256
        data['freestyle_weight_sha256'] = weight_sha256
        (OUT/'analysis.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
        provenance = {key: data[key] for key in ('engine', 'source_url', 'engine_sha256', 'freestyle_weight_sha256', 'rule', 'node_limit', 'per_position_ms', 'threads', 'seed_sha256', 'human_source_url', 'selection', 'base_book_sha256', 'retained_positions')}
        provenance['analysis_sha256'] = hashlib.sha256((OUT/'analysis.json').read_bytes()).hexdigest()
        provenance.update(positions=len(records), alternative_moves=len(alternatives),
                          max_position_stones=max((len(k[1])+len(k[2]) for k in records), default=-1), proven=False)
        text = '# -*- coding: utf-8 -*-\n\"\"\"离线有限搜索的开局建议，不构成胜败证明。\"\"\"\n'
        text += 'PROVENANCE = ' + repr(provenance) + '\n\nRECORDS = (\n'
        for row in sorted(records.values(), key=lambda x: (x[0], tuple(x[1]), tuple(x[2]))):
            text += '    ' + repr((row[0], tuple(row[1]), tuple(row[2]), row[3], row[4])) + ',\n'
        text += ')\n\nALTERNATIVES = (\n'
        for row in alternatives.values():
            text += '    ' + repr((row[0], tuple(row[1]), tuple(row[2]), row[3], row[4])) + ',\n'
        (OUT/'opening_book_data.py').write_text(text + ')\n\nCANDIDATES = ()\n', encoding='utf-8')
    def store(history, move, alternate=False):
        board = board_of(history)
        key, symmetry = canonical_position(board, len(history)%2+1)
        r, c = transform_point(*move, symmetry)
        target = alternatives if alternate else records
        target[key] = [key[0], list(key[1]), list(key[2]), r, c]
    def analyze(history):
        if len(audit)>=MAX_POSITIONS or time.monotonic()>=DEADLINE:
            return None
        board=board_of(history);player=len(history)%2+1
        if any(check_win(board,r,c,i%2+1) for i,(r,c) in enumerate(history)):
            return None
        key,symmetry=canonical_position(board,player)
        if key in records:
            return inverse_point(*records[key][-2:], symmetry)
        send('RESTART');until('OK')
        send('BOARD')
        for i,(r,c) in enumerate(history):
            send(f'{c},{r},{1 if i%2+1==player else 2}')
        send('DONE')
        start=time.monotonic();line,lines=until(r'\d+,\d+')
        c,r=map(int,line.split(','));assert 0<=r<15 and 0<=c<15 and board[r][c]==0
        rr,cc=transform_point(r,c,symmetry)
        records[key]=[key[0],list(key[1]),list(key[2]),rr,cc]
        audit.append({'history':history,'player':player,'move':[r,c],
                      'milliseconds':round((time.monotonic()-start)*1000,2),'messages':lines[:-1]})
        save()
        if len(audit)%10==0:
            print('ANALYZED',len(audit),'ply',len(history),'remaining_seconds',round(DEADLINE-time.monotonic()),flush=True)
        return r,c
    def unique(histories):
        seen=set();out=[]
        for h in histories:
            key,_=canonical_position(board_of(h),len(h)%2+1)
            if key not in seen:
                seen.add(key);out.append(h)
        return out
    def alternatives_near(history,best,count):
        if not history:
            return [best]
        b=board_of(history)
        cells=[(r,c) for r in range(6,9) for c in range(6,9) if b[r][c]==0]
        cells.sort(key=lambda x:(max(abs(x[0]-best[0]),abs(x[1]-best[1])),x))
        options=unique([history+[x] for x in [best]+[x for x in cells if x!=best]])
        return [h[-1] for h in options[:count]]

    try:
        send('START 15');until('OK')
        for option in ('rule 0','max_memory 268435456','timeout_turn 5000','timeout_match 0',
                       'thread_num 1',f'max_node {NODE_LIMIT}'):
            send('INFO '+option)
        store([], (7, 7))
        white_second = [tuple(move) for move in seeds['white_second']]
        store([(7, 7)], white_second[0])
        store([(7, 7)], white_second[1], alternate=True)
        for move in white_second:
            analyze([(7, 7), move])
        # 先确保26种白四均有应手，再把预算花在后续变化；不按单一最优路径取样。
        fourth = []
        for item in seeds['openings']:
            history = [tuple(p) for p in item['history']]
            best = analyze(history)
            if best is not None:
                choices = [best] + [tuple(p) for p in item['reference_replies'] if tuple(p) != best][:2]
                fourth.extend(history + [move] for move in choices)
        fifth = []
        for history in unique(fourth):
            best = analyze(history)
            if best is not None:
                # 附加黑五仅作为对手可能的偏离，为白六准备应手；不会随机推荐它。
                fifth.extend(history + [move] for move in alternatives_near(history, best, 2))
        for history in unique(fifth):
            analyze(history)
        save()
        print('COMPLETE',len(records),'positions',flush=True)
    finally:
        try:
            send('END');p.wait(timeout=5)
        except (BrokenPipeError,subprocess.TimeoutExpired):
            p.kill();p.wait()


if __name__ == '__main__':
    main()
