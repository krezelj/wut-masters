from abc import ABC
from games.base_game import BaseGame, BaseMove


class BasePlayer(ABC):

    __slots__ = ['hash_name']

    def __init__(self):
        super().__init__()

    def get_move(self, game: BaseGame) -> BaseMove:
        raise NotImplementedError()