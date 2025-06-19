import subprocess
import logging
from typing import Literal

from games.base_game import BaseGame, BaseMove
from players.base_player import BasePlayer

EXE_PATH = './external/MastersAlgorithms.exe'


class ConnectionManager:

    EXCEPTION_RESPONSE = "exception"
    END_RESPONSE = "end\n"
    EXIT_MESSAGE = "exit"

    def __init__(self, verbose: bool = False, **kwargs):
        self.verbose = verbose
        args = []
        if self.verbose:
            args.append("--verbose")
        for k, v in kwargs.items():
            args.append(f'--{k}')
            args.append(f'{v}')

        self.process = subprocess.Popen(
            [EXE_PATH, *args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8'
        )

    def __del__(self):
        self.process.terminate()
        self.process.wait()

    def __parse_command(self, command_args: dict) -> str:
        string_command = ""
        for k, v in command_args.items():
            if v is True:
                string_command += f"--{k} "
            elif v is False:
                continue
            else:
                string_command += f"--{k} {v} "
        assert(len(string_command) > 0)
        return string_command[:-1] + '\n'
    
    def __handle_exception(self):
        error_message = self.process.stdout.readline()
        stack_trace = []
        for line in iter(self.process.stdout.readline,""):
            line = line.lstrip()
            if line == self.END_RESPONSE:
                break
            stack_trace.append(line)
        stack_trace = ''.join(stack_trace)

        self.process.stdin.write(self.EXIT_MESSAGE)
        self.process.stdin.flush()
        HIGHLIGHT_START = '\033[91m\033[1m'
        HIGHLIGHT_END = '\033[0m'
        raise Exception(f"{HIGHLIGHT_START}The subprocess has raised an exception: {error_message}{HIGHLIGHT_END}"
                        + f"Stack Trace:\n{stack_trace}")

    def __send_command(self, command_args: dict):
        if self.process.poll() is not None:
            error_message = self.process.stderr.read()
            raise Exception(f"Process has terminated. Error: {error_message}")
        
        command = self.__parse_command(command_args)
        logging.debug(command[:-1])

        self.process.stdin.write(command)
        self.process.stdin.flush()

        response = self.process.stdout.readline()
        response = response[:-1] # remove \n at the end
        if response == self.EXCEPTION_RESPONSE:
            self.__handle_exception()
        logging.debug(f"response: {response}")

        return response
    
    def __to_camel_case(self, words: str):
        words = words.split('_')
        cap_words = map(lambda x: x.lower().capitalize(), words[1:])
        return words[0].lower() + ''.join(cap_words)

    def __add_kwargs(self, command_args: dict, **kwargs):
        for k, v in kwargs.items():
            # convert to camelCase
            k = self.__to_camel_case(k)
            command_args[k] = v

    def add_game(self, name: str, **kwargs) -> str:
        command_args = {
            'command': 'addGame',
            'name': self.__to_camel_case(name),
        }
        self.__add_kwargs(command_args, **kwargs)
        return self.__send_command(command_args)
    
    def remove_game(self, game: BaseGame) -> str:
        command_args = {
            'command': 'removeGame',
            'hashName': game.hash_name
        }
        return self.__send_command(command_args)

    def add_algorithm(self, name: Literal["minimax", "mcts"], **kwargs) -> str:
        command_args = {
            'command': 'addAlgorithm',
            'name': name,
        }
        self.__add_kwargs(command_args, **kwargs)
        return self.__send_command(command_args)

    def remove_algorithm(self, player: BasePlayer) -> str:
        command_args = {
            'command': 'removeGame',
            'hashName': player.hash_name
        }
        return self.__send_command(command_args)

    def get_move(self, game: BaseGame, player: BasePlayer) -> str:
        command_args = {
            'command': 'getMove',
            'algorithm': player.hash_name,
        }
        if hasattr(game, 'hash_name'):
            command_args['game'] = game.hash_name
        else:
            command_args['name'] = game.name
            command_args['state'] = str(game)
            # since we do not know whether the player requires
            # zobrist, we must include it
            command_args['useZobrist'] = True

        return self.__send_command(command_args)

    def get_moves(self, game: BaseGame) -> str:
        command_args = {
            'command': 'getMoves',
            'game': game.hash_name,
        }
        return self.__send_command(command_args)

    def get_random_move(self, game: BaseGame) -> str:
        command_args = {
            'command': 'getRandomMove',
            'game': game.hash_name,
        }
        return self.__send_command(command_args)

    def make_move(self, game: BaseGame, move: BaseMove) -> str:
        command_args = {
            'command': 'makeMove',
            'game': game.hash_name,
            'move': str(move)
        }
        return self.__send_command(command_args)

    def undo_move(self, game: BaseGame, move: BaseMove) -> str:
        command_args = {
            'command': 'undoMove',
            'game': game.hash_name,
            'move': str(move)
        }
        return self.__send_command(command_args)
    
    def is_over(self, game: BaseGame) -> str:
        command_args = {
            'command': 'isOver',
            'game': game.hash_name
        }
        return self.__send_command(command_args)
    
    def result(self, game: BaseGame) -> str:
        command_args = {
            'command': 'result',
            'game': game.hash_name
        }
        return self.__send_command(command_args)

    def evaluate(self, game: BaseGame) -> str:
        command_args = {
            'command': 'evaluate',
            'game': game.hash_name,
        }
        return self.__send_command(command_args)

    def get_string(self, game: BaseGame) -> str:
        command_args = {
            'command': 'getString',
            'game': game.hash_name,
        }
        return self.__send_command(command_args)

    def copy(self, game: BaseGame) -> str:
        command_args = {
            'command': 'copy',
            'game': game.hash_name,
        }
        return self.__send_command(command_args)
    
    def run_game(self, game: BaseGame, players: list[BasePlayer], first_player_idx: int):
        command_args = {
            'command': 'runGame',
            'game': game.hash_name,
            'players': ";".join(map(lambda p: p.hash_name, players)),
            "firstPlayerIdx": str(first_player_idx)
        }
        return self.__send_command(command_args)
    
    def run_match(self, 
                  game_type: str,
                  players: list[BasePlayer],
                  n_games: int,
                  mirror_games: bool,
                  n_random_moves: int):
        command_args = {
            "command": "runMatch",
            "gameType": self.__to_camel_case(game_type),
            "players": ";".join(map(lambda p: p.hash_name, players)),
            "nGames": str(n_games),
            "mirrorGames": mirror_games,
            "nRandomMoves": str(n_random_moves)
        }
        return self.__send_command(command_args)

CMInstance = ConnectionManager(verbose=True)