from abc import ABC
from typing import Literal

import numpy.typing as npt

class BaseMove(ABC):

    __slots__ = ['index']

    def __init__(self):
        super().__init__()

class BaseGame(ABC):

    __slots__ = ['name', 'hash_name', 'player_idx', 'is_over', 'result', 
                 'n_possible_outcomes', 'obs_shape', 'n_actions']
    

    def __init__(self):
        super().__init__()

    def close(self):
        pass # used to cleanup after the game has finished, but not required
    
    def get_moves(self) -> list[BaseMove]:
        raise NotImplementedError()

    def get_random_move(self) -> BaseMove:
        raise NotImplementedError()

    def get_move_from_index(self, index: int) -> BaseMove:
        raise NotImplementedError()

    def sort_moves(self, moves: list[BaseMove]):
        raise NotImplementedError()

    def make_move(self, move: BaseMove):
        idxs = [m.index for m in self.get_moves()]
        assert(move.index in idxs)

    def undo_move(self, move):
        raise NotImplementedError()

    def evaluate(self) -> float:
        raise NotImplementedError()

    def copy(self) -> 'BaseGame':
        raise NotImplementedError()
    
    def render(self):
        raise NotImplementedError()

    def get_obs(self, obs_mode: Literal["flat", "image"]) -> npt.NDArray:
        raise NotImplementedError()
    
    def action_masks(self, with_moves: bool = False) -> list[bool]:
        raise NotImplementedError()
    
    def get_move_from_action(self, action: int) -> BaseMove:
        raise NotImplementedError()
    
    def get_move_from_move_data(self, move_data: str) -> BaseMove:
        raise NotImplementedError()
    
    def get_move_from_user_input(self, user_input: str) -> BaseMove:
        raise NotImplementedError()