from typing import Optional, Type
import warnings

import numpy as np

from core.game_generator import GameGenerator
from games.base_game import BaseGame, BaseMove
from players.base_player import BasePlayer


# status enums
STARTING = 0    #
READY = 1       # ready to start
RUNNING = 2     # currently running
WAITING = 3     # waiting for a response
RESUMING = 4    # got response, but not yet running
OVER = -1        # all games have finished 


class MatchManager:

    # __slots__ = ['current_player_idx']

    @property
    def current_player(self):
        return self.players[self.current_player_idx]

    def __init__(self, 
                 players: list[Optional[BasePlayer]],
                 game_type: Type[BaseGame], 
                 n_games: int = 1,
                 mirror_games: bool = False,
                 n_random_moves: int = 0,
                 seed: int = 0,
                 verbose: int = 0):
        
        self.status = STARTING
        self.verbose = verbose
        self.players = players
        self.current_player_idx = 0
        self.game_generator = GameGenerator(game_type, n_games, mirror_games, n_random_moves, seed)
        self.current_game: Optional[BaseGame] = None
        self.wins: list = []
        self.games_completed: int = 0
        self.status = READY

    def run(self) -> Optional[BaseGame]:
        if self.status == WAITING:
            raise RuntimeError("run method was called while the MatchManager is WAITING")
        if self.status == READY:
            self.status = RUNNING
    
        while self.status != OVER:

            if self.current_game is None:
                self.current_game, is_mirrored = self.game_generator.get_next_game()
                self.current_player_idx = 0

                if self.current_game is None:
                    self.status = OVER
                    break
                elif is_mirrored:
                    self.__advance_players()
                self.first_player_idx = self.current_player_idx

            self.__run_game()

            if self.status == WAITING:
                return self.current_game
            else:
                # TODO log current game status (who won, how many moves etc.)
                self.games_completed += 1
                self.__update_stats()
                if self.verbose > 0:
                    self.__print_stats()
                self.current_game = None

    def __run_game(self):
        while not self.current_game.is_over:
            move = None
            if self.current_player is None and self.status == RUNNING:
                self.status = WAITING
                return
            elif self.current_player is None and self.status == RESUMING:
                move = self.response_move
                self.status = RUNNING
            else:
                move = self.current_player.get_move(self.current_game)

            self.__make_move(move)

    def __make_move(self, move: BaseMove):
        # TODO log status and move
        self.current_game.make_move(move)
        self.__advance_players()

    def __advance_players(self):
        self.current_player_idx = (self.current_player_idx + 1) % len(self.players)

    def respond(self, move: BaseMove):
        self.response_move = move
        self.status = RESUMING

    def __update_stats(self):
        result = self.current_game.result
        if result < 0:
            self.wins.append(result)
        else:
            # TODO check how it scales for more than two players
            self.wins.append(abs(result - self.first_player_idx))

    def __print_stats(self):
        # from the perspective of the first player
        wins = np.sum(np.array(self.wins) == 0)
        losses = np.sum(np.array(self.wins) == 1)
        draws = np.sum(np.array(self.wins) == -1)
        print(f"Game {self.games_completed:>5} | {wins}-{draws}-{losses} |")