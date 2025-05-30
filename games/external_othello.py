from typing import Literal, Optional
import numpy as np
import numpy.typing as npt

from core.connection_manager import CMInstance, ConnectionManager
from games.base_game import BaseGame, BaseMove
from games.utils import *

BOARD_SIZE = 8

class ExternalOthelloMove(BaseMove):

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

class ExternalOthello(BaseGame):

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

    name = 'othello'
    n_possible_outcomes = 3
    n_actions = BOARD_SIZE * BOARD_SIZE + 1 # TODO GameEnv should handle this 
    obs_shape = (2, BOARD_SIZE, BOARD_SIZE) # TODO GameEnv should handle this

    @property
    def state(self):
        if self.__state_cache is None:
            self.__state_cache = self.connection_manager.get_string(self)
        return self.__state_cache
    
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
        self.__state_cache = None
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
        self.connection_manager = connection_manager
        self.use_zobrist = use_zobrist
        if hash_name is None:
            self.hash_name = self.connection_manager.add_game(name='othello', use_zobrist=use_zobrist)
        else:
            self.hash_name = hash_name
        self.__state_cache = None
        self.__moves_cache = None

    def close(self):
        self.connection_manager.remove_game(self)

    def get_moves(self) -> list[ExternalOthelloMove]:
        if self.__moves_cache is not None:
            return self.__moves_cache
        moves_data = self.connection_manager.get_moves(self).split(';')
        self.__moves_cache = []
        for md in moves_data:
            self.__moves_cache.append(ExternalOthelloMove(md))
        return self.__moves_cache

    def get_random_move(self) -> ExternalOthelloMove:
        move_data = self.connection_manager.get_random_move(self)
        return ExternalOthelloMove(move_data)

    def get_move_from_index(self, index: int) -> ExternalOthelloMove:
        raise NotImplementedError()

    def sort_moves(self, moves: list[BaseMove]):
        moves.sort(key=lambda m: -self.weights[m.index])

    def make_move(self, move: ExternalOthelloMove):
        move_data = self.connection_manager.make_move(self, move)
        move.move_data = move_data
        self.__state_cache = None
        self.__moves_cache = None

    def undo_move(self, move: ExternalOthelloMove):
        self.connection_manager.undo_move(self, move)
        self.__state_cache = None
        self.__moves_cache = None

    def evaluate(self) -> float:
        return float(self.connection_manager.evaluate(self))

    def render(self, show_legal_moves: bool=True):
        raise NotImplementedError()

    def copy(self) -> 'ExternalOthelloMove':
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
    
    def action_masks(self, with_moves: bool = False) -> list[bool]:
        mask = [False] * (BOARD_SIZE ** 2 + 1)
        moves = self.get_moves()
        for move in moves:
            mask[move.index] = True
        mask[-1] = not np.any(mask[:-1])
        if with_moves:
            return mask, moves
        return mask

    def get_move_from_action(self, action: int) -> BaseMove:
        if action == self.n_actions - 1:
            return ExternalOthelloMove.get_null_move(self)        
        return ExternalOthelloMove(f"{action},{self.null_moves},0")

    def get_move_from_user_input(self, user_input: str) -> BaseMove:
        raise NotImplementedError()

    def get_move_from_move_data(self, move_data: str) -> BaseMove:
        return ExternalOthelloMove(move_data)

    def __str__(self) -> str:
        return self.state

if __name__ == '__main__':
    pass