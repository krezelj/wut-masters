from typing import Optional, Type, Union

import numpy as np

from games.base_game import BaseGame


class GameGenerator:

    def __init__(self, 
                 game_type: Type[BaseGame],
                 n_games: int, 
                 mirror_games: bool, 
                 n_random_moves: Union[int, tuple[int, int]],
                 game_kwargs: dict = {},
                 seed: int = 0,
                 **kwargs):
        
        self.game_type = game_type
        self.n_games = n_games
        self.mirror_games = mirror_games
        self.n_random_moves = n_random_moves
        self.game_kwargs = game_kwargs
        self._rng = np.random.default_rng(seed)

        self.n_games_generated = 0
        self.last_game: Optional[BaseGame] = None

    def get_next_game(self) -> tuple[Optional[BaseGame], bool]:
        # == and not >= to allow -1 to mean infinity
        if self.n_games_generated == self.n_games:
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
        if isinstance(self.n_random_moves, tuple):
            low = self.n_random_moves[0] // 2
            high = self.n_random_moves[1] // 2 + 1
            n = self._rng.integers(low, high) * 2
        else:
            n = self.n_random_moves
        assert(n % 2 == 0)

        # for large enough n it is possible that the game might finish
        # before returning, we need to handle this case and generate new
        # games until a valid game state is generated
        while True:
            game = self.game_type(**self.game_kwargs)
            for _ in range(n):

                # despite BaseGame implementing get_random_move,
                # we use own random seed generator for consistensy
                moves = game.get_moves()
                move = self._rng.choice(moves)
                game.make_move(move)
                if game.is_over:
                    break
            else: # finally, if all moves were valid, break out of the while loop
                break

        # TODO add game evaluation (using MCTS?)
        return game