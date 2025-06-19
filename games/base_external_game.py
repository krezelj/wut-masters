from typing import Any, Literal, Optional, Union
from abc import abstractmethod

import numpy as np
import numpy.typing as npt

from core.connection_manager import CMInstance, ConnectionManager
from games.base_game import BaseGame, BaseMove
from games.utils import *

class BaseExternalMove(BaseMove):

    __slots__ = ['index', 'move_data']

    def __init__(self, move_data: str):
        self.move_data = move_data

class BaseExternalGame(BaseGame):

    @property
    def state(self):
        if self._state_cache is None:
            self._state_cache = self.connection_manager.get_string(self)
        return self._state_cache
        
    def __init__(self, 
                 hash_name: Optional[str] = None, 
                 use_zobrist: bool = True,
                 connection_manager: ConnectionManager = CMInstance):
        self.connection_manager = connection_manager
        self.use_zobrist = use_zobrist
        self._state_cache = None
        self._moves_cache = None
        if hash_name is None:
            self.hash_name = self.connection_manager.add_game(name=self.name, use_zobrist=use_zobrist)
        else:
            self.hash_name = hash_name
        

    def close(self):
        self.connection_manager.remove_game(self)

    def get_moves(self) -> list[Any]:
        if self._moves_cache is not None:
            return self._moves_cache
        moves_data = self.connection_manager.get_moves(self).split(';')
        self._moves_cache = []
        for md in moves_data:
            self._moves_cache.append(self.move_type(md))
        return self._moves_cache

    def get_random_move(self) -> BaseExternalMove:
        move_data = self.connection_manager.get_random_move(self)
        return self.move_type(move_data)

    def get_move_from_index(self, index: int) -> BaseExternalMove:
        raise NotImplementedError()

    @abstractmethod
    def sort_moves(self, moves: list[BaseExternalMove]):
        raise NotImplementedError()

    def make_move(self, move: BaseExternalMove):
        move_data = self.connection_manager.make_move(self, move)
        move.move_data = move_data
        self._state_cache = None
        self._moves_cache = None

    def undo_move(self, move: BaseExternalMove):
        self.connection_manager.undo_move(self, move)
        self._state_cache = None
        self._moves_cache = None

    def evaluate(self) -> float:
        return float(self.connection_manager.evaluate(self))

    @abstractmethod
    def render(self, show_legal_moves: bool=True):
        raise NotImplementedError()

    @abstractmethod
    def copy(self) -> 'BaseExternalGame':
        pass
    
    @abstractmethod
    def get_obs(self, obs_mode: Literal["flat", "image"]) -> npt.NDArray:
        pass
    
    def action_masks(self, with_moves: bool = False) -> Union[list[bool], tuple[list[bool, list[BaseExternalMove]]]]:
        mask = [False] * self.n_actions
        moves = self.get_moves()
        for move in moves:
            mask[move.index] = True
        if with_moves:
            return mask, moves
        return mask

    @abstractmethod
    def get_move_from_action(self, action: int) -> BaseExternalMove:
        pass

    @abstractmethod
    def get_move_from_user_input(self, user_input: str) -> BaseExternalMove:
        pass

    def get_move_from_move_data(self, move_data: str) -> BaseExternalMove:
        return self.move_type(move_data)

    def __str__(self) -> str:
        return self.state

if __name__ == '__main__':
    pass