

from typing import Callable, Literal

import torch
from games.base_game import BaseGame, BaseMove
from players.base_player import BasePlayer


class MinimaxPlayer(BasePlayer):

    MAX_VAL = 10_000_000

    def __init__(self, depth: int, eval_func: Callable, sort_func: Callable,):
        super().__init__()
        self.depth = depth
        self.cum_nodes = 0
        self.__eval_func = eval_func
        self.__sort_func = sort_func

    def get_move(self, game: BaseGame) -> BaseMove:
        self.__nodes = 0
        self.game = game

        # for d in range(1, self.depth + 1):
        #     self.__search(d, 0, -self.MAX_VAL, self.MAX_VAL)
        self.__search(self.depth, 0, -self.MAX_VAL, self.MAX_VAL)
        self.cum_nodes += self.__nodes

        return self.best_move

    def __search(self, depth, ply, alpha, beta):
        self.__nodes += 1

        if depth == 0 or self.game.is_over:
            return self.__eval_func(self.game)
        isRoot = ply == 0
        
        moves = self.game.get_moves()
        self.__sort_func(self.game, moves) # sorts inplace

        current_value = -self.MAX_VAL
        for move in moves:
            self.game.make_move(move)
            newValue = -self.__search(depth - 1, ply + 1, -beta, -alpha)
            self.game.undo_move(move)

            if newValue > current_value:
                current_value = newValue
                alpha = max(current_value, alpha)
                if isRoot:
                    self.best_move = move
                if alpha >= beta:
                    break; # prune

        return current_value

    @classmethod
    def default_evaluation(cls, game: BaseGame) -> float:
        return game.evaluate()

    @classmethod
    def default_sorting(cls, game: BaseGame, moves: list[BaseMove]):
        return game.sort_moves(moves)
    
    @classmethod
    def get_model_evaluation_func(cls, model, obs_mode: Literal["flat", "image"]) -> Callable:
        def model_evaluation(game: BaseGame):
            with torch.inference_mode():
                obs = torch.tensor(game.get_obs(obs_mode=obs_mode)).unsqueeze(dim=0).to("cpu")
                v = model.policy.predict_values(obs).item()
            return -v
        return model_evaluation
    
    @classmethod
    def get_model_sorting_func(cls, model, obs_mode: Literal["flat", "image"]) -> Callable:
        def model_sorting(game: BaseGame, moves: list[BaseMove]):
            with torch.inference_mode():
                obs = torch.tensor(game.get_obs(obs_mode=obs_mode)).unsqueeze(dim=0).to("cpu")
                probs = model.policy.get_distribution(obs).distribution.probs
                probs = probs.detach().numpy().squeeze()
            moves.sort(key=lambda m: -probs[m.index])
        return model_sorting