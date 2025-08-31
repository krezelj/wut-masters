


import itertools
import logging
from typing import Optional, Union

import numpy as np
from typing import Type

from core.match_manager import MatchManager
from games.base_game import BaseGame
from players.base_player import BasePlayer


class Tournament:

    # log levels
    __BASE_MATCH_LL = 1
    __BASE_MATCH_RESULTS_LL = __BASE_MATCH_LL + 1

    def __init__(self,
                 players: list[BasePlayer],
                 game_type: Type[BaseGame],
                 player_names: Optional[list[str]] = None,
                 n_games: int = 1,
                 n_random_moves: Union[int, tuple[int, int]] = 0,
                 count_draws_as_half_wins: bool = False,
                 include_both_sides: bool = False,
                 skip_selfplay: bool = False,
                 seed: int = 0,
                 verbose: int = 0,
                 **kwargs):
        
        self.players = players
        self.game_type = game_type
        self.player_names = player_names
        self.n_games = n_games
        self.n_random_moves = n_random_moves
        self.count_draws_as_half_wins = count_draws_as_half_wins
        self.include_both_sides = include_both_sides
        self.skip_selfplay = skip_selfplay
        self.seed = seed
        self.verbose = verbose
        self.mm_verbose = kwargs.get("mm_verbose", 0)

        self.win_matrix = np.zeros(shape=(len(self.players), len(self.players)))
        self.times = np.zeros(shape=(len(self.players,)))

    def run(self):
        indices = list(range(len(self.players)))
        idx_pairs =  list(itertools.product(indices, repeat=2))
        for idx_pair in idx_pairs:
            if self.skip_selfplay and idx_pair[0] == idx_pair[1]:
                continue
            if idx_pair[0] > idx_pair[1]:
                continue
            if self.verbose >= self.__BASE_MATCH_LL:
                logging.info(
                    f"Running match between {self.__get_player_name(idx_pair[0])} and {self.__get_player_name(idx_pair[1])}")
            results, times = self.__run_match(idx_pair)
            self.__update_stats(results, times, idx_pair)
            if self.verbose >= self.__BASE_MATCH_RESULTS_LL:
                logging.info(results)

        # normalise stats
        self.win_matrix /= self.n_games
        self.times /= len(self.players)

        return self.win_matrix, self.times

    def __run_match(self, idx_pair):
        i, j = idx_pair
        mm = MatchManager(
            players=[self.players[i], self.players[j]],
            game_type=self.game_type,
            n_games=self.n_games,
            mirror_games=True,
            n_random_moves=self.n_random_moves,
            allow_external_simulation=True,
            seed=self.seed,
            verbose=self.mm_verbose
        )
        mm.run_external()
        return mm.results, mm.times
    
    def __update_stats(self, results, times, idx_pair):
        i, j = idx_pair
        self.win_matrix[i, j] = results[0, 0]
        self.win_matrix[j, i] = results[0, 1]
        if self.include_both_sides:
            self.win_matrix[i, j] += results[1, 1]
            self.win_matrix[j, j] += results[1, 0]

        if self.count_draws_as_half_wins:
            draws = results[2, :].sum()
            self.win_matrix[i, j] += draws / 2
            self.win_matrix[j, j] += draws / 2

        self.times[i] += float(times[0].replace(',', '.'))
        self.times[j] += float(times[1].replace(',', '.'))

    def __get_player_name(self, idx):
        if self.player_names is None:
            return idx
        return self.player_names[idx]