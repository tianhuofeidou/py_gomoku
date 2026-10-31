"""离线训练任务：父进程调度，计算由隔离的 Python / 教师进程完成。"""
import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from gomoku.tools import teacher_tune as U, tune_params as T


class Jobs:
    """有界批次、按输入顺序归集；线程只等待外部进程，不注入引擎参数。"""
    def __init__(self, workers):
        self.workers = workers

    def map(self, function, items):
        items = list(items)
        results = []
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for start in range(0, len(items), self.workers):
                futures = [pool.submit(function, item)
                           for item in items[start:start + self.workers]]
                try:
                    results.extend(f.result() for f in futures)
                except BaseException:
                    for future in futures:
                        future.cancel()
                    raise
        return results


class Labels:
    """先去重和预留全局额度，再并行请求；每个键只有一个写入者。"""
    def __init__(self, teacher, jobs, reuse=()):
        self.teacher, self.jobs = teacher, jobs
        self.reuse = [Path(p) for p in reuse]

    def key(self, moves):
        return T.digest([self.teacher.identity, moves])

    def read(self, moves):
        key = self.key(moves)
        data = U.read_json(self.teacher.cache / (key + '.json'))
        checksum = data.pop('payload_sha256', None)
        if (data.get('key') != key or data.get('identity') != self.teacher.identity
                or checksum != T.digest(data)):
            raise ValueError('教师缓存指纹损坏')
        return data

    def prefetch(self, histories):
        unique = {self.key(moves): moves for moves in histories}
        missing = []
        for key, moves in sorted(unique.items()):
            target = self.teacher.cache / (key + '.json')
            if not target.exists():
                for directory in self.reuse:
                    source = directory / target.name
                    if source.exists():
                        data = U.read_json(source)
                        original = dict(data)
                        checksum = data.pop('payload_sha256', None)
                        if (data.get('key') != key or data.get('identity') != self.teacher.identity
                                or checksum != T.digest(data)):
                            raise ValueError('复用教师缓存指纹损坏')
                        U.write_json(target, original)
                        break
            if target.exists():
                self.read(moves)
            else:
                missing.append(moves)
        # 额度是实验目录全局额度，不是每个 worker 分到完整额度。
        remaining = self.teacher.max_queries - len(list(self.teacher.cache.glob('*.json')))
        allowed = missing[:max(0, remaining)]

        def query(moves):
            isolated = copy.copy(self.teacher)
            isolated.queries = 0
            isolated.max_queries = 1
            return isolated._query(moves)

        self.jobs.map(query, allowed)
        if len(allowed) != len(missing):
            raise U.BudgetStop('达到教师缓存总额度；已完成任务保留，未重复启动查询')

    def utility(self, position, move):
        board, _ = U.board_of(position['moves'])
        r, c = move
        if not (0 <= r < 15 and 0 <= c < 15) or board[r][c]:
            raise ValueError('待标注候选非法')
        board[r][c] = position['player']
        if U.check_win(board, r, c, position['player']):
            return 1.0
        if len(position['moves']) == 224:
            return 0.0
        return -self.read(position['moves'] + [[r, c, position['player']]])['evaluation']['utility']

    def child_history(self, position, move):
        board, _ = U.board_of(position['moves'])
        r, c = move
        board[r][c] = position['player']
        if U.check_win(board, r, c, position['player']) or len(position['moves']) == 224:
            return None
        return position['moves'] + [[r, c, position['player']]]
