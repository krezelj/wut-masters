from typing import Optional, Type
import warnings

import numpy as np

from core.game_generator import GameGenerator
from games.base_game import BaseGame, BaseMove
from players.base_player import BasePlayer

class MatchManager:

    # status enums
    READY = 1       # ready to start the next game
    RUNNING = 2     # currently running a game
    WAITING = 3     # waiting for a response mid game
    RESUMING = 4    # got response, but not yet running a game
    OVER = -1       # all games have finished 

    @property
    def current_player(self):
        return self.players[self.current_player_idx]
    
    @property
    def draw_result(self):
        return self.current_game.n_possible_outcomes - 1

    def __init__(self, 
                 players: list[Optional[BasePlayer]],
                 game_type: Type[BaseGame], 
                 n_games: int = 1,
                 mirror_games: bool = False,
                 n_random_moves: int = 0,
                 pause_after_game: bool = False,
                 seed: int = 0,
                 verbose: int = 0):
        
        self.verbose = verbose
        self.pause_after_game = pause_after_game
        self.players = players

        self.current_game: Optional[BaseGame] = None
        self.results = np.zeros(shape=(game_type.n_possible_outcomes, len(players)))
        self.last_winner_idx : Optional[int] = None
        self.games_completed = 0

        self.game_generator = GameGenerator(game_type, n_games, mirror_games, n_random_moves, seed)

        self.status = self.READY

    def run(self):
        if self.status == self.WAITING:
            raise RuntimeError("run method was called while the MatchManager is WAITING")
        if self.status == self.READY:
            self.status = self.RUNNING
            self.current_game = None
    
        while self.status != self.OVER:

            if self.current_game is None:
                self.current_game, is_mirrored = self.game_generator.get_next_game()
                self.current_player_idx = 0

                if self.current_game is None:
                    self.status = self.OVER
                    break
                elif is_mirrored:
                    self.__advance_players()
                self.first_player_idx = self.current_player_idx

            self.__run_game()

            if self.status == self.WAITING:
                return
            else:
                # TODO log current game status (who won, how many moves etc.)
                self.games_completed += 1
                self.__update_stats()
                if self.verbose > 0:
                    self.__print_stats()
                # self.current_game = None
            if self.pause_after_game:
                self.status = self.READY
                return

    def __run_game(self):
        while not self.current_game.is_over:
            move = None
            if self.current_player is None and self.status == self.RUNNING:
                self.status = self.WAITING
                return
            elif self.current_player is None and self.status == self.RESUMING:
                move = self.response_move
                self.status = self.RUNNING
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
        self.status = self.RESUMING

    def __update_stats(self):
        result = self.current_game.result
        if result == self.draw_result:
            self.last_winner_idx = -1
        else:
            self.last_winner_idx = (self.first_player_idx - result) % len(self.players)
        self.results[result, self.first_player_idx] += 1

    def __print_stats(self):
        # from the perspective of the player with index 0
        non_draws = self.results[:-1, :]
        wins = non_draws.diagonal().sum().astype(np.int32)
        losses = np.sum(non_draws).astype(np.int32) - wins
        draws = np.sum(self.results[-1, :]).astype(np.int32)

        print(f"Game {self.games_completed:>5} | {wins}-{draws}-{losses}")