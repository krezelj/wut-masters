import subprocess
import logging
from typing import Literal

from games.base_game import BaseGame
from players.base_player import BasePlayer

EXE_PATH = './external/MastersAlgorithms.exe'

class ExternalPlayer(BasePlayer):

    def __init__(self, 
                 game: Literal["othello", "connect_four"], 
                 algorithm: Literal["minimax", "mcts"],
                 log_debug: bool = False,
                 **kwargs):
        
        self.log_debug = log_debug
        args = ["--game", game, "--algorithm", algorithm]
        if self.log_debug:
            args.append("--verbose")
        for k, v in kwargs.items():
            args.append(f'--{k}')
            args.append(f'{v}')

        self.process = subprocess.Popen(
            [EXE_PATH, *args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

    def __del__(self):
        self.process.terminate()
        self.process.wait()

    def get_move(self, game: BaseGame) -> int:
        if self.process.poll() is not None:
            error_message = self.process.stderr.read()
            raise Exception(f"Process has terminated. Error: {error_message}")
        
        self.process.stdin.write(str(game) + "\n")
        self.process.stdin.flush()

        response = self.process.stdout.readline()
        move_index_str, debug_msg = response.split(';')
        move_index = int(move_index_str)
        move = game.get_move_from_index(move_index)
        
        if self.log_debug:
            # remove \n character
            logging.debug(debug_msg[:-1])

        return move