

import numpy as np
from games.base_game import BaseMove
from games.othello import Othello
from players.base_player import BasePlayer


class OthelloPositionalPlayer(BasePlayer):
    """
    Implemented according to https://doi.org/10.1016/j.cor.2006.10.004
    """

    def __init__(self):
        super().__init__()
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
        self.endgame_threshold = 0.8

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
        return np.sum(game.player_board * self.weights - game.opponent_board * self.weights)

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