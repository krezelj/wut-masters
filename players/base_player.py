from abc import ABC
from games.base_game import BaseGame


class BasePlayer(ABC):

    def __init__(self):
        super().__init__()

    def get_move(game: BaseGame) -> int:
        pass