import subprocess
from typing import Literal

from games.base_game import BaseGame
from players.base_player import BasePlayer

EXE_PATH = './external/MastersAlgorithms.exe'

class ExternalPlayer(BasePlayer):

    def __init__(self, 
                 game: Literal["othello", "connect_four"], 
                 algorithm: Literal["minimax", "mcts"], 
                 **kwargs):
        
        args = ["--game", game, "--algorithm", algorithm]
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
        move_index = int(self.process.stdout.readline())
        move = game.get_move_from_index(move_index)
        
        return move