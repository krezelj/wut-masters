import numpy as np
from games.base_game import BaseMove
from games.othello import Othello
from players.base_player import BasePlayer


class OthelloHeuristicPlayer(BasePlayer):
    
    endgame_threshold = 0.8

    def __init__(self):
        super().__init__()

    def get_move(self, game: Othello) -> BaseMove:
        best_move = None
        best_evaluation = -np.inf
        for move in game.get_moves():
            game.make_move(move)
            # minus, since we are not evaluating from the perspective
            # of the opponent
            evaluation = -self.__get_evaluation(game)
            game.undo_move(move)

            if evaluation > best_evaluation:
                best_evaluation = evaluation
                best_move = move
        return best_move
    
    def __get_evaluation(self, game: Othello) -> float:
        if self.__check_endgame(game):
            return np.sum(game.player_board) - np.sum(game.opponent_board)
        return self._get_base_evaluation(game)

    def _get_base_evaluation(self, game: Othello) -> float:
        raise NotImplementedError()
    
    def __check_endgame(self, game: Othello) -> bool:
        occupied = np.sum(game.board, axis=0)
        if occupied.sum() / (game.size ** 2) >= self.endgame_threshold:
            return True
        for i in [0, -1]:
            for j in [0, -1]:
                # open corner
                if occupied[i, j] == 0:
                    return False 
        return True # all corners occupied


class OthelloPositionalPlayer(OthelloHeuristicPlayer):
    """
    Implemented according to https://doi.org/10.1016/j.cor.2006.10.004
    """

    def __init__(self):
        super(OthelloPositionalPlayer, self).__init__()
        self.weights = np.array([
            [100, -20, 10, 5, 5, 10, -20, 100],
            [-20, -50, -2, -2, -2, -2, -50, -20],
            [10, -2, -1, -1, -1, -1, -2, 10],
            [5, -2, -1, -1, -1, -1, -2, 5],
            [5, -2, -1, -1, -1, -1, -2, 5],
            [10, -2, -1, -1, -1, -1, -2, 10],
            [-20, -50, -2, -2, -2, -2, -50, -20],
            [100, -20, 10, 5, 5, 10, -20, 100],
        ])

    def get_move(self, game: Othello) -> BaseMove:
        return super(OthelloPositionalPlayer, self).get_move(game)
    
    def _get_base_evaluation(self, game) -> float:
        return np.sum(game.player_board * self.weights - game.opponent_board * self.weights)


class OthelloMobilityPlayer(OthelloHeuristicPlayer):
    """
    Implemented according to https://doi.org/10.1016/j.cor.2006.10.004
    """

    def __init__(self):
        super(OthelloMobilityPlayer, self).__init__()
        self.w1 = 10
        self.w2 = 1

    def get_move(self, game: Othello) -> BaseMove:
        return super(OthelloMobilityPlayer, self).get_move(game)
    
    def _get_base_evaluation(self, game: Othello) -> float:
        corner_diff = self.__get_corner_diff(game)
        mp = len(game.get_moves())

        game.black_to_move = not game.black_to_move
        mo = len(game.get_moves())
        game.black_to_move = not game.black_to_move

        return self.w1 * corner_diff + self.w2 * ((mp - mo) / (mp + mo))

    def __get_corner_diff(self, game: Othello) -> int:
        diff = 0
        for i in [0, -1]:
            for j in [0, -1]:
                if game.player_board[i, j]:
                    diff += 1
                elif game.opponent_board[i, j]:
                    diff -= 1
        return diff