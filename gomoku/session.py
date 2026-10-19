# -*- coding: utf-8 -*-
"""对局会话：把 Game 状态机 + 同步好的引擎实例包装成一次对局。

GUI（play.py）与 CLI（play_cli.py）共用：落子后同步引擎内部状态，
AI 轮次用 service.ai_move 计算并落子；人人对战可只分析当前局面而不落子。
"""
from gomoku.game import Game, BLACK, WHITE
from gomoku import service


class PlaySession:
    def __init__(self, sid='local'):
        self.sid = sid
        self.game = Game(human_player=BLACK, mode='ai', ai_mode='engine')
        self.game.register(sid)
        self.engine = service.create_engine()

    def reset(self, mode='ai', human_player=BLACK):
        history = self.game.history[:]
        self.game = Game(human_player=human_player, mode=mode, ai_mode='engine')
        self.game.history = history
        self.game.register(self.sid)
        self.engine.reset()

    def place(self, r, c, player):
        res = self.game.place(r, c, player)
        if res['ok']:
            self.engine.on_move(self.game.board(), r, c, player)
        return res

    def ai_play(self, player=None):
        game = self.game
        if game.winner is not None:
            return None
        if player is None:
            player = game.ai_player()
        if player != game.current_player():
            return None
        mv, ranked, net_used = service.ai_move(game.board(), game.moves, player, engine=self.engine)
        if mv is None:
            return None
        r, c = mv
        res = self.place(r, c, player)
        if not res['ok']:
            raise RuntimeError(res['error'])
        return (r, c), ranked, net_used

    def analyze(self, player=None):
        """分析当前行棋方并返回推荐点和候选评分，不改变棋局或落子历史。"""
        game = self.game
        if game.winner is not None:
            return None
        if player is None:
            player = game.current_player()
        if player != game.current_player():
            return None
        board = game.board()
        # 空盘没有棋型可供 fallback 评分，沿用引擎正式开局策略推荐天元，
        # 给出归一化展示分；从第二手起进入完整 analyze_turn 深推并返回多候选。
        if not game.moves:
            self.engine.deep_search.reset_leaf_eval_stats()
            self.engine.decision_net_used = False
            opening = self.engine.compute_move(board, player)
            if isinstance(opening, list) and opening:
                r, c = opening[0]
                ranked = [self.engine.deep_search._candidate_result(r, c, 100.0)]
                return (r, c), ranked, False
        result = self.engine.analyze_turn(board, player)
        part4 = result['part4_result']
        move = part4.get('move')
        ranked = part4.get('ranked') or []
        return move, ranked, self.engine.decision_net_used

    def undo(self, by='human'):
        if by == 'human':
            self.game.undo_human_move()
        elif by == 'ai':
            self.game.undo_ai_move()
        else:
            self.game.undo_last_move()
        # Game 层悔棋只回退 moves，引擎增量状态表不回退 →
        # 全盘重建，否则幽灵子残留干扰后续候选/必杀检测（漏杀根因）
        self.engine.reset(self.game.moves)
