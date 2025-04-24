from typing import Literal, Optional
import numpy as np
import numpy.typing as npt

from core.connection_manager import CMInstance
from games.base_game import BaseGame, BaseMove
from games.utils import *

BOARD_SIZE = 8

class ExternalOthelloMove(BaseMove):

    @property
    def algebraic(self):
        if self.index < 0:
            return "null"
        position = (self.index // BOARD_SIZE, self.index % BOARD_SIZE)
        return f"{chr(position[1] + ord('a'))}{position[0] + 1}"

    @property
    def index(self):
        # TODO Maybe optimise this one day
        return int(self.move_data.split(',')[0])

    def __init__(self, move_data: str):
        self.move_data = move_data
        # self.index = int(self.move_data.split(',')[0])
        # self.null_moves = int(self.move_data[1])
        # self.flip_mask = int(self.move_data[2])

    def __str__(self):
        # return self.algebraic
        # return f"{self.index},{self.null_moves},{self.flip_mask}"
        return self.move_data

class ExternalOthello(BaseGame):

    n_possible_outcomes = 3
    n_actions = BOARD_SIZE * BOARD_SIZE + 1 # TODO GameEnv should handle this 
    obs_shape = (2, BOARD_SIZE, BOARD_SIZE) # TODO GameEnv should handle this

    @property
    def state(self):
        if self.__state_cache is None:
            self.__state_cache = CMInstance.get_string(self)
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
    
    @property
    def is_over(self):
        is_full = '.' not in self.state
        return is_full or self.null_moves == 2

    @property
    def result(self):
        if not self.is_over:
            return None
        if self.material_diff > 0:
            return 0 # black idx
        if self.material_diff < 0:
            return 1 # white idx
        return 2 # draw

    def __init__(self, hash_name: Optional[str] = None):
        if hash_name is None:
            self.hash_name = CMInstance.add_game(name='othello')
        else:
            self.hash_name = hash_name
        self.__state_cache = None

    def get_moves(self) -> list[ExternalOthelloMove]:
        moves_data = CMInstance.get_moves(self).split(';')
        moves = []
        for md in moves_data:
            moves.append(ExternalOthelloMove(md))
        return moves

    def get_random_move(self) -> ExternalOthelloMove:
        move_data = CMInstance.get_random_move(self)
        return ExternalOthelloMove(move_data)

    def get_move_from_index(self, index: int) -> ExternalOthelloMove:
        raise NotImplementedError()

    def make_move(self, move: ExternalOthelloMove):
        move_data = CMInstance.make_move(self, move)
        move.move_data = move_data
        self.__state_cache = None

    def undo_move(self, move: ExternalOthelloMove):
        CMInstance.undo_move(self, move)
        self.__state_cache = None

    def evaluate(self) -> float:
        return float(CMInstance.evaluate(self))

    def render(self, show_legal_moves: bool=True):
        raise NotImplementedError()

    def copy(self) -> 'ExternalOthelloMove':
        new_hash_name = CMInstance.copy(self)
        return ExternalOthello(new_hash_name)
    
    def get_obs(self, obs_mode: Literal["flat", "image"]) -> npt.NDArray:
        raise NotImplementedError()
    
    def action_masks(self) -> list[bool]:
        mask = [False] * (self.size ** 2 + 1)
        for move in self.get_moves():
            mask[move.index] = True
        mask[-1] = not np.any(mask[:-1])
        return mask

    def get_move_from_action(self, action: int) -> BaseMove:
        raise NotImplementedError()

    def get_move_from_user_input(self, user_input: str) -> BaseMove:
        raise NotImplementedError()
        # if len(user_input) != 2:
        #     raise ValueError("Invalid move")
        # j = ord(user_input[0]) - ord('a')
        # i = int(user_input[1]) - 1
        # if not is_in_limits(i, j, self.shape):
        #     raise ValueError("Invalid move")
        # try:
        #     move = next(filter(lambda move: move.position == (i, j), self.get_moves()))
        # except:
        #     raise ValueError("Invalid move")
        # return move

    def get_move_from_move_data(self, move_data: str) -> BaseMove:
        return ExternalOthelloMove(move_data)

    def __str__(self) -> str:
        return self.state

if __name__ == '__main__':
    pass