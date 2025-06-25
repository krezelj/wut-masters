from typing import Literal, Optional
import numpy as np
import numpy.typing as npt

from core.connection_manager import CMInstance, ConnectionManager
from games.base_external_game import BaseExternalGame, BaseExternalMove
from games.utils import *

BLACK = 0
WHITE = 1
# BOARD_SIZE = 8
OBS_HEIGHT = 8
OBS_WIDTH = 4

class ExternalCheckersMove(BaseExternalMove):

    @property
    def index(self):
        # TODO Maybe optimise this one day
        return int(self.move_data.split(',')[0])
    
    @index.setter
    def index(self, value):
        pass

    def __init__(self, move_data: str):
        super().__init__(move_data)

    def __str__(self):
        return self.move_data

class ExternalCheckers(BaseExternalGame):

    move_type = ExternalCheckersMove
    name = 'checkers'
    n_possible_outcomes = 3
    n_actions = 129
    obs_shape = (3, OBS_HEIGHT, OBS_WIDTH)

    @property
    def state(self):
        if self._state_cache is None:
            self._state_cache = self.connection_manager.get_string(self)
        return self._state_cache
        
    @property
    def player_idx(self):
        return int(self.state[-1])
    
    @player_idx.setter
    def player_idx(self, value):
        pass
    
    @property
    def is_over(self):
        return self.connection_manager.is_over(self) == "True"
    
    @is_over.setter
    def is_over(self, value):
        pass

    @property
    def result(self):
        self._state_cache = None
        if not self.is_over:
            return None
        return int(self.connection_manager.result(self))
    
    @result.setter
    def result(self, value):
        pass

    def __init__(self, 
                 hash_name: Optional[str] = None, 
                 use_zobrist: bool = True,
                 connection_manager: ConnectionManager = CMInstance):
        super().__init__(hash_name, use_zobrist, connection_manager)

    def sort_moves(self, moves: list[ExternalCheckersMove]):
        pass # no sorting for this game, sorry!

    def render(self, show_legal_moves: bool=True):
        raise NotImplementedError()

    def copy(self) -> 'ExternalCheckers':
        new_hash_name = self.connection_manager.copy(self)
        return ExternalCheckers(new_hash_name, use_zobrist=self.use_zobrist, connection_manager=self.connection_manager)
    
    def get_obs(self, obs_mode: Literal["flat", "image"]) -> npt.NDArray:
        # raise NotImplementedError()
        board = np.zeros(shape=(3, OBS_HEIGHT, OBS_WIDTH))
        state = self.state
        for i in range(OBS_HEIGHT):
            for j in range(OBS_WIDTH):
                char = state[i * OBS_WIDTH + j]
                if char == 'x' or char == 'X':
                    board[0, i, j] = 1
                elif char == 'o' or char == 'O':
                    board[1, i, j] = 1
                if char == 'X' or char == 'O':
                    board[2, i, j] = 1
        
        player_board = board[self.player_idx, :, :]
        opponent_board = board[1 - self.player_idx, :, :]
        kings = board[2, :, :]

        if self.player_idx == WHITE:
            player_board = player_board.flatten()[::-1].reshape(OBS_HEIGHT, OBS_WIDTH)
            opponent_board = opponent_board.flatten()[::-1].reshape(OBS_HEIGHT, OBS_WIDTH)
            kings = kings.flatten()[::-1].reshape(OBS_HEIGHT, OBS_WIDTH)

        obs = np.stack([player_board, opponent_board, kings])
        if obs_mode == "flat":
            return obs.flatten().astype(np.float32)
        if obs_mode == "image":
            return obs.astype(np.uint8) * 255

    def get_move_from_action(self, action: int) -> ExternalCheckersMove:
        if self._moves_cache is None:
            raise ValueError("move cache is cleared")
        for move in self._moves_cache:
            if move.index == action:
                return move
        raise ValueError("no matching move was found!")

    def get_move_from_user_input(self, user_input: str) -> ExternalCheckersMove:
        raise NotImplementedError()

if __name__ == '__main__':
    pass