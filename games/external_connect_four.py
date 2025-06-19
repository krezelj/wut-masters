from typing import Literal, Optional
import numpy as np
import numpy.typing as npt

from core.connection_manager import CMInstance, ConnectionManager
from games.base_external_game import BaseExternalGame, BaseExternalMove
from games.utils import *

WIDTH = 7
HEIGHT = 6

class ExternalConnectFourMove(BaseExternalMove):

    @property
    def index(self):
        return int(self.move_data)
    
    @index.setter
    def index(self, value):
        pass

    def __init__(self, move_data: str):
        super().__init__(move_data)

    def __str__(self):
        return self.move_data

class ExternalConnectFour(BaseExternalGame):

    move_type = ExternalConnectFourMove
    name = 'connect_four'
    n_possible_outcomes = 3
    n_actions = WIDTH # TODO GameEnv should handle this 
    obs_shape = (2, HEIGHT, WIDTH) # TODO GameEnv should handle this

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

    def sort_moves(self, moves: list[ExternalConnectFourMove]):
        pass # no sorting for this game, sorry!

    def render(self, show_legal_moves: bool=True):
        raise NotImplementedError()

    def copy(self) -> 'ExternalConnectFour':
        new_hash_name = self.connection_manager.copy(self)
        return ExternalConnectFour(new_hash_name, use_zobrist=self.use_zobrist, connection_manager=self.connection_manager)
    
    def get_obs(self, obs_mode: Literal["flat", "image"]) -> npt.NDArray:
        board = np.zeros(shape=(2, HEIGHT, WIDTH))
        state= self.state
        for i in range(HEIGHT):
            for j in range(WIDTH):
                char = state[i * WIDTH + j]
                if char == 'X':
                    board[0, i, j] = 1
                elif char == 'O':
                    board[1, i, j] = 1
        
        player_board = board[self.player_idx, :, :]
        opponent_board = board[1 - self.player_idx, :, :]

        obs = np.stack([player_board, opponent_board])
        if obs_mode == "flat":
            return obs.flatten().astype(np.float32)
        if obs_mode == "image":
            return obs.astype(np.uint8) * 255

    def get_move_from_action(self, action: int) -> ExternalConnectFourMove:
        return ExternalConnectFourMove(str(action))

    def get_move_from_user_input(self, user_input: str) -> ExternalConnectFourMove:
        raise NotImplementedError()

if __name__ == '__main__':
    pass