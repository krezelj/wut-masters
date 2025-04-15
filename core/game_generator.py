from typing import Optional, Type

import numpy as np

from games.base_game import BaseGame


class GameGenerator:

    def __init__(self, 
                 game_type: Type[BaseGame],
                 n_games: int, 
                 mirror_games: bool, 
                 n_random_moves: int,
                 seed: int = 0):
        
        self.game_type = game_type
        self.n_games = n_games
        self.mirror_games = mirror_games
        self.n_random_moves = n_random_moves
        self._rng = np.random.default_rng(seed)

        self.n_games_generated = 0
        self.last_game: Optional[BaseGame] = None

    def get_next_game(self) -> tuple[Optional[BaseGame], bool]:
        if self.n_games_generated >= self.n_games:
            return None, False

        if self.mirror_games and self.last_game is not None:
            game_to_return = self.last_game
            self.last_game = None
            self.n_games_generated += 1
            return game_to_return, True
        else:
            game_to_return = self.__generate_game()
            self.last_game = game_to_return.copy() if self.mirror_games else None
            if not self.mirror_games:
                self.n_games_generated += 1
            return game_to_return, False
        
        # return game_to_return

    def __generate_game(self) -> BaseGame:
        game = self.game_type()
        for _ in range(self.n_random_moves):

            # despite BaseGame implementing get_random_move,
            # we use own random seed generator for consistensy
            moves = game.get_moves()
            move = self._rng.choice(moves)
            game.make_move(move)

        # TODO add game evaluation (using MCTS?)
        return game