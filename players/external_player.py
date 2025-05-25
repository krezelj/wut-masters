import subprocess
import logging
from typing import Literal

from core.connection_manager import CMInstance
from games.base_game import BaseGame, BaseMove
from players.base_player import BasePlayer

class ExternalPlayer(BasePlayer):

    def __init__(self, 
                 algorithm: Literal["bogo", "minimax", "mcts", "agent", "mctsBatch"],
                 log_info: bool = False,
                 **kwargs):
        
        self.log_info = log_info
        self.hash_name = CMInstance.add_algorithm(algorithm, **kwargs)

    def __del__(self):
        CMInstance.remove_algorithm(self)

    def get_move(self, game: BaseGame) -> BaseMove:
        response = CMInstance.get_move(game, self)
        move_data, debug_msg = response.split(';')
        move = game.get_move_from_move_data(move_data)
        
        if self.log_info:
            logging.info(debug_msg)

        return move