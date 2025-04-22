import time
import logging
from typing import Optional, Type

import numpy as np

from core.game_generator import GameGenerator
from games.base_game import BaseGame, BaseMove
from players.base_player import BasePlayer

class MatchManager:

    # status enums
    READY = 1       # ready to start the next game
    RUNNING = 2     # currently running games
    WAITING = 3     # waiting for a response mid game
    RESUMING = 4    # got response, but not yet running a game
    OVER = -1       # all games have finished

    # log levels
    __BASE_GAME_LL = 1
    __BASE_MOVE_LL = 2
    __BASE_MOVE_STATE_LL = __BASE_MOVE_LL + 1

    @property
    def current_player(self):
        return self.players[self.current_player_idx]
    
    @property
    def current_player_name(self):
        if self.player_names is not None:
            return self.player_names[self.current_player_idx]
        return self.current_player_idx
    
    @property
    def draw_result(self):
        return self.current_game.n_possible_outcomes - 1

    def __init__(self, 
                 players: list[Optional[BasePlayer]],
                 game_type: Type[BaseGame],
                 player_names: Optional[list[str]] = None,
                 n_games: int = 1,
                 mirror_games: bool = False,
                 n_random_moves: int = 0,
                 pause_after_game: bool = False,
                 seed: int = 0,
                 verbose: int = 0):
        
        self.verbose = verbose
        self.pause_after_game = pause_after_game
        self.players = players
        self.player_names = player_names

        self.current_game: Optional[BaseGame] = None
        self.results = np.zeros(shape=(game_type.n_possible_outcomes, len(players)))
        self.last_winner_idx : Optional[int] = None
        self.games_completed = 0
        self.moves_made = 0

        self.game_generator = GameGenerator(game_type, n_games, mirror_games, n_random_moves, seed)
        self.status = self.READY

    def run(self):
        if self.status == self.WAITING:
            msg = "`run` method was called while the MatchManager is WAITING"
            logging.error(msg)
            raise RuntimeError(msg)
        if self.status == self.READY:
            self.status = self.RUNNING
    
        while self.status != self.OVER:
            if self.status == self.RUNNING:
                self.current_game = None

            if self.current_game is None:
                self.__start_new_game()

            if self.current_game is None:
                self.status = self.OVER
                return
            
            self.__run_game()

            if self.status == self.WAITING:
                return
            else:
                self.__finish_game()

            if self.pause_after_game:
                self.status = self.READY
                return
            
    def respond(self, move: BaseMove):
        self.response_move = move
        self.status = self.RESUMING

    def __start_new_game(self):
        self.current_game, is_mirrored = self.game_generator.get_next_game()
        self.moves_made = 0
        self.current_player_idx = 0
        if self.current_game is None:
            return
        elif is_mirrored:
            self.__advance_players()
        self.first_player_idx = self.current_player_idx

    def __run_game(self):
        while not self.current_game.is_over:
            if self.status == self.RUNNING:
                t_start = time.time()

            # handle undefined player
            if self.current_player is None:
                if self.status == self.RUNNING:
                    self.status = self.WAITING
                    return
                elif self.status == self.RESUMING:
                    move = self.response_move
                    self.status = self.RUNNING
            else:
                move = self.current_player.get_move(self.current_game)

            self.__elapsed_ms = (time.time() - t_start) * 1000
            self.__make_move(move)

    def __finish_game(self):
        self.games_completed += 1
        self.__update_stats()
        if self.verbose >= self.__BASE_GAME_LL:
            self.__log_game()

    def __make_move(self, move: BaseMove):
        if self.verbose >= self.__BASE_MOVE_LL:
            self.__log_move(move)
        self.moves_made += 1
        self.current_game.make_move(move)
        self.__advance_players()

    def __advance_players(self):
        self.current_player_idx = (self.current_player_idx + 1) % len(self.players)

    def __update_stats(self):
        result = self.current_game.result
        if result == self.draw_result:
            self.last_winner_idx = -1
        else:
            self.last_winner_idx = (self.first_player_idx - result) % len(self.players)
        self.results[result, self.first_player_idx] += 1

    def __log_move(self, move: BaseMove):
        msg = f"\nGame {self.games_completed}, player {self.current_player_name} made move no. {self.moves_made}\n"
        if self.verbose >= self.__BASE_MOVE_STATE_LL:
            msg += f"\tState: {str(self.current_game)}\n"
        msg += f"\tMove: {str(move)}\n"
        msg += f"\tTime: {self.__elapsed_ms:.2f}ms"
        logging.info(msg)

    def __log_game(self):
        # from the perspective of the player with index 0
        non_draws = self.results[:-1, :]
        wins = non_draws.diagonal().sum().astype(np.int32)
        losses = np.sum(non_draws).astype(np.int32) - wins
        draws = np.sum(self.results[-1, :]).astype(np.int32)

        msg = f"Game {self.games_completed:>5} finished ({wins}-{draws}-{losses})"
        logging.info(msg)
        # print(f"Game {self.games_completed:>5} | {wins}-{draws}-{losses}")