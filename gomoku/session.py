# -*- coding: utf-8 -*-
"""对局会话：把 Game 状态机 + 同步好的引擎实例包装成一次对局。

GUI（play.py）与 CLI（play_cli.py）共用：落子后同步引擎内部状态，
AI 轮次用 service.ai_move 计算并落子。
"""
from gomoku.game import Game, BLACK, WHITE
from gomoku import service


class PlaySession:
    def __init__(self, sid='local'):
        self.sid = sid
        self.game = Game(human_player=BLACK, mode='ai', ai_mode='engine')
        self.game.register(sid)
        self.engine = service.get_engine()

    def reset(self, mode='ai', human_player=BLACK):
        self.game = Game(human_player=human_player, mode=mode, ai_mode='engine')
        self.game.register(self.sid)
        self.engine = service.get_engine()

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
        mv, ranked, net_used = service.ai_move(game.board(), game.moves, player, engine=self.engine)
        r, c = mv
        self.place(r, c, player)
        return (r, c), ranked, net_used

    def undo(self, by='human'):
        if by == 'human':
            self.game.undo_human_move()
        elif by == 'ai':
            self.game.undo_ai_move()
        else:
            self.game.undo_last_move()
