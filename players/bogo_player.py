

from typing import Optional
import numpy as np
from games.base_game import BaseGame, BaseMove
from players.base_player import BasePlayer


class BogoPlayer(BasePlayer):

    def __init__(self, seed: Optional[int] = None):
        super().__init__()
        self._rng = np.random.default_rng(seed)

    def get_move(self, game: BaseGame) -> BaseMove:
        return self._rng.choice(game.get_moves())