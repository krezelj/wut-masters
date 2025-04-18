from abc import ABC
from games.base_game import BaseGame


class BasePlayer(ABC):

    def __init__(self):
        super().__init__()

    def get_move(self, game: BaseGame) -> int:
        raise NotImplementedError()