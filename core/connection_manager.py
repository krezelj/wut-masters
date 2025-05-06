import struct
import subprocess
import logging
import win32file
from typing import Literal

from games.base_game import BaseGame, BaseMove
from players.base_player import BasePlayer

EXE_PATH = './external/MastersAlgorithms.exe'


class ConnectionManager:

    def __init__(self, verbose: bool = False, **kwargs):
        self.verbose = verbose
        args = []
        if self.verbose:
            args.append("--verbose")
        for k, v in kwargs.items():
            args.append(f'--{k}')
            args.append(f'{v}')

        # self.process = subprocess.Popen(
        #     [EXE_PATH, *args],
        #     stdin=subprocess.PIPE,
        #     stdout=subprocess.PIPE,
        #     stderr=subprocess.PIPE,
        #     text=True
        # )

        # self.file_handle = win32file.CreateFile(
        #     r"\\\\.\\pipe\\communication", 
        #     win32file.GENERIC_READ | win32file.GENERIC_WRITE, 
        #     0, 
        #     None, 
        #     win32file.OPEN_EXISTING, 
        #     0, 
        #     None)

        self.f = open(r'\\.\pipe\communication', 'r+b', 0)

    def __del__(self):
        pass
        #self.process.terminate()
        #self.process.wait()

    def __parse_command(self, command_args: dict) -> str:
        string_command = ""
        for k, v in command_args.items():
            string_command += f"--{k} {v} "
        assert(len(string_command) > 0)
        return string_command[:-1] + '\n'

    def __send_command(self, command_args: dict):
        # if self.process.poll() is not None:
        #     error_message = self.process.stderr.read()
        #     raise Exception(f"Process has terminated. Error: {error_message}")
        
        command = self.__parse_command(command_args)[:-1].encode('ascii')
        # logging.debug(command[:-1])

        self.f.write(struct.pack('I', len(command)) + command)   # Write str length and str
        self.f.seek(0) 

        # win32file.WriteFile(self.file_handle, command, None)
        # win32file.FlushFileBuffers(self.file_handle)

        # self.process.stdin.write(command)
        # self.process.stdin.flush()


        n = struct.unpack('I', self.f.read(4))[0]    # Read str length
        response = self.f.read(n).decode('ascii')           # Read str
        self.f.seek(0)   
        # left, response = win32file.ReadFile(self.file_handle, 4096)

        # response = self.process.stdout.readline()
        # response = response[:-1] # remove \n at the end
        # logging.debug(f"response: {response}")

        return response
    
    def __add_kwargs(self, command_args: dict, **kwargs):
        for k, v in kwargs.items():
            command_args[k] = v

    def add_game(self, name: Literal["othello", "connect_four"], **kwargs) -> str:
        command_args = {
            'command': 'addGame',
            'name': name,
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
            command_args['zobrist'] = "True"

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

    

CMInstance = ConnectionManager(verbose=True)