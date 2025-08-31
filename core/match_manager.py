from copy import copy
import time
import logging
from typing import Optional, Type, Union
import csv

import numpy as np

from core.connection_manager import CMInstance, ConnectionManager
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
    __BASE_MATCH_LL = 1
    __BASE_GAME_LL = __BASE_MATCH_LL + 1
    __BASE_MOVE_LL = __BASE_GAME_LL + 1
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
                 n_random_moves: Union[int, tuple[int, int]] = 0,
                 pause_after_game: bool = False,
                 connection_manager: Optional['ConnectionManager'] = CMInstance,
                 allow_external_simulation: bool = True,
                 game_kwargs: dict = {},
                 seed: int = 0,
                 csv_filename: Optional[str] = None,
                 verbose: int = 0,
                 **kwargs):
        if connection_manager is None:
            self.connection_manager = ConnectionManager(verbose=True)
        else:
            self.connection_manager = connection_manager
        game_kwargs = copy(game_kwargs)
        if "connection_manager" not in game_kwargs:
            game_kwargs['connection_manager'] = self.connection_manager

        can_simulate_externally = all(map(lambda p: hasattr(p, "hash_name"), players))
        if allow_external_simulation and can_simulate_externally:
            self.simulate_externally = True
        else:
            if allow_external_simulation:
                logging.warning("External simulation is not possible.")
            self.simulate_externally = False

        self.verbose = verbose
        self.pause_after_game = pause_after_game
        self.players = players
        self.game_type = game_type
        self.player_names = player_names
        self.n_games = n_games
        self.mirror_games = mirror_games
        self.n_random_moves = n_random_moves

        self.current_game: Optional[BaseGame] = None
        self.results = np.zeros(shape=(game_type.n_possible_outcomes, len(players)))
        self.times = None
        self.last_winner_idx : Optional[int] = None
        self.games_completed = 0
        self.moves_made = 0

        self.game_generator = GameGenerator(
            game_type, 
            n_games, 
            mirror_games, 
            n_random_moves, 
            game_kwargs,
            seed,
            **kwargs)

        if csv_filename is not None:
            self.log_csv_data = True
            if not csv_filename.endswith(".csv"):
                csv_filename += ".csv"
            self.csv_log = open(csv_filename, mode='w', newline='')
            self.csv_writer = csv.writer(self.csv_log, delimiter=';')
        else:
            self.log_csv_data = False
            self.csv_log = None
            self.csv_writer = None

        self.status = self.READY

    def __del__(self):
        pass
        # if self.csv_log is not None:
        #     self.csv_log.close()

    def run(self):
        if self.status == self.WAITING:
            msg = "`run` method was called while the MatchManager is WAITING"
            logging.error(msg)
            raise RuntimeError(msg)
        if self.status == self.READY:
            self.status = self.RUNNING
    
        while self.status != self.OVER:
            if self.status == self.RUNNING:
                self.__prepare_for_next_game()

            if self.current_game is None:
                self.__start_new_game()

            if self.current_game is None:
                self.__finish_match()
                return
            
            if self.simulate_externally:
                self.__run_external_game()
            else:
                self.__run_game()

            if self.status == self.WAITING:
                return
            else:
                self.__finish_game()

            if self.pause_after_game:
                self.status = self.READY
                return

    def run_external(self):
        response = self.connection_manager.run_match(
            game_type=self.game_type.name,
            players=self.players,
            n_games=self.n_games,
            mirror_games=self.mirror_games,
            n_random_moves=self.n_random_moves
        )

        results, *times = response.split(';')
        results = results.split(',')
        self.times = times
        iterator = 0
        for i in range(self.results.shape[0]):
            for j in range(self.results.shape[1]):
                self.results[i, j] = int(results[iterator])
                iterator += 1
        
        self.__finish_match()
        return self.results, self.times

    def respond(self, move: BaseMove):
        self.response_move = move
        self.status = self.RESUMING

    def __run_game(self):
        while not self.current_game.is_over:
            if self.status == self.RUNNING:
                self.t_start = time.process_time_ns()

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

            self.__elapsed_ms = (time.process_time_ns() - self.t_start) / 1e6
            self.__make_move(move)

    def __run_external_game(self):
        debug_response = self.connection_manager.run_game(self.current_game, self.players, self.first_player_idx)
        logging.debug(debug_response)

    def __prepare_for_next_game(self):
        if self.current_game is not None:
            self.current_game.close()
        self.current_game = None

    def __start_new_game(self):
        self.current_game, is_mirrored = self.game_generator.get_next_game()
        self.moves_made = 0
        self.current_player_idx = 0
        if self.current_game is None:
            return
        elif is_mirrored:
            self.__advance_players()
        self.first_player_idx = self.current_player_idx

    def __finish_game(self):
        self.games_completed += 1
        self.__update_stats()
        if self.verbose >= self.__BASE_GAME_LL:
            self.__log_game()

    def __finish_match(self):
        self.status = self.OVER
        if self.verbose >= self.__BASE_MATCH_LL:
            self.__log_match()

    def __make_move(self, move: BaseMove):
        if self.verbose >= self.__BASE_MOVE_LL:
            self.__log_move(move)
        if self.log_csv_data:
            self.__log_csv_data(move)
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
        msg += f"\tTime: {self.__elapsed_ms:.5f}ms"
        logging.info(msg)

    def __log_csv_data(self, move: BaseMove):
        self.csv_writer.writerow([
            str(self.current_game), 
            str(move.index), 
            self.current_player_name, 
            np.round(self.__elapsed_ms, 2)]
        )

    def __log_game(self):
        # from the perspective of the player with index 0
        non_draws = self.results[:-1, :]
        wins = non_draws.diagonal().sum().astype(np.int32)
        losses = np.sum(non_draws).astype(np.int32) - wins
        draws = np.sum(self.results[-1, :]).astype(np.int32)

        msg = f"Game {self.games_completed:>5} finished ({wins}-{draws}-{losses})"
        logging.info(msg)

    def __log_match(self):
        if len(self.players) == 2:
            wins = int(self.results[0, 0] + self.results[1, 1])
            losses = int(self.results[0, 1] + self.results[1, 0])
            draws = int(self.results[2, 0] + self.results[2, 1])
            logging.info(f"Result: {wins}-{draws}-{losses}")
        logging.info(self.results)
        if self.times is not None:
            logging.info(self.times)

        