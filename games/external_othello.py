from typing import Literal, Optional
import numpy as np
import numpy.typing as npt

from core.connection_manager import CMInstance, ConnectionManager
from games.base_external_game import BaseExternalGame, BaseExternalMove
from games.utils import *

BOARD_SIZE = 8

class ExternalOthelloMove(BaseExternalMove):

    null_move_idx = 64

    @property
    def algebraic(self):
        if self.index < 0:
            return "null"
        position = (self.index // BOARD_SIZE, self.index % BOARD_SIZE)
        return f"{chr(position[1] + ord('a'))}{position[0] + 1}"
    
    @algebraic.setter
    def algebraic(self, value):
        pass

    @property
    def index(self):
        # TODO Maybe optimise this one day
        return int(self.move_data.split(',')[0])
    
    @index.setter
    def index(self, value):
        pass

    def __init__(self, move_data: str):
        self.move_data = move_data

    @classmethod
    def get_null_move(cls, game: 'ExternalOthello') -> 'ExternalOthelloMove':
        return ExternalOthelloMove(f"{ExternalOthelloMove.null_move_idx},{game.null_moves},0")

    def __str__(self):
        return self.move_data

class ExternalOthello(BaseExternalGame):

    weights = np.array([
            [100, -20, 10, 5, 5, 10, -20, 100],
            [-20, -50, -2, -2, -2, -2, -50, -20],
            [10, -2, -1, -1, -1, -1, -2, 10],
            [5, -2, -1, -1, -1, -1, -2, 5],
            [5, -2, -1, -1, -1, -1, -2, 5],
            [10, -2, -1, -1, -1, -1, -2, 10],
            [-20, -50, -2, -2, -2, -2, -50, -20],
            [100, -20, 10, 5, 5, 10, -20, 100],
    ]).flatten()

    move_type = ExternalOthelloMove
    name = 'othello'
    n_possible_outcomes = 3
    n_actions = BOARD_SIZE * BOARD_SIZE + 1 # TODO GameEnv should handle this 
    obs_shape = (2, BOARD_SIZE, BOARD_SIZE) # TODO GameEnv should handle this

    @property
    def state(self):
        if self._state_cache is None:
            self._state_cache = self.connection_manager.get_string(self)
        return self._state_cache
    
    @property
    def null_moves(self):
        return int(self.state[-1])
    
    @property
    def material_diff(self):
        return self.state.count('X') - self.state.count('O')

    @property
    def player_idx(self):
        return int(self.state[-2])
    
    @player_idx.setter
    def player_idx(self, value):
        pass
    
    @property
    def is_over(self):
        is_full = '.' not in self.state
        return is_full or self.null_moves == 2
    
    @is_over.setter
    def is_over(self, value):
        pass

    @property
    def result(self):
        self._state_cache = None
        if not self.is_over:
            return None
        if self.material_diff > 0:
            return 0 # black idx
        if self.material_diff < 0:
            return 1 # white idx
        return 2 # draw
    
    @result.setter
    def result(self, value):
        pass

    def __init__(self, 
                 hash_name: Optional[str] = None, 
                 use_zobrist: bool = True,
                 connection_manager: ConnectionManager = CMInstance):
        super().__init__(hash_name, use_zobrist, connection_manager)

    def sort_moves(self, moves: list[ExternalOthelloMove]):
        moves.sort(key=lambda m: -self.weights[m.index])

    def render(self, show_legal_moves: bool=True):
        if show_legal_moves:
            moves = [move.index for move in self.get_moves()]

        top_row = f"  {'  '.join([chr(ord('a') + i) for i in range(BOARD_SIZE)])} "
        print(top_row)
        for i in range(BOARD_SIZE):
            print(f"{i+1} ", end="")
            for j in range(BOARD_SIZE):
                char = self.state[i * BOARD_SIZE + j]
                if char == 'X':
                    print('X  ', end="")
                    continue
                elif char == 'O':
                    print('O  ', end="")
                    continue
                elif show_legal_moves and i * BOARD_SIZE + j in moves:
                    print('.  ', end="")
                    continue
                print('   ', end="")
            print("")

    def copy(self) -> 'ExternalOthello':
        new_hash_name = self.connection_manager.copy(self)
        return ExternalOthello(new_hash_name, use_zobrist=self.use_zobrist, connection_manager=self.connection_manager)
    
    def get_obs(self, obs_mode: Literal["flat", "image"]) -> npt.NDArray:
        board = np.zeros(shape=(2, BOARD_SIZE, BOARD_SIZE))
        state = self.state
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                char = state[i * BOARD_SIZE + j]
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
    
    def get_move_from_action(self, action: int) -> ExternalOthelloMove:
        if action == self.n_actions - 1:
            return ExternalOthelloMove.get_null_move(self)        
        return ExternalOthelloMove(f"{action},{self.null_moves},0")

    def get_move_from_user_input(self, user_input: str) -> ExternalOthelloMove:
        raise NotImplementedError()

if __name__ == '__main__':
    pass