from typing import Literal
import numpy as np
import numpy.typing as npt

from games.base_game import BaseGame, BaseMove
from games.utils import *

class OthelloMove(BaseMove):

    @property
    def algebraic(self):
        if self.position is None:
            return "null"
        return f"{chr(self.position[1] + ord('a'))}{self.position[0] + 1}"
    
    @property
    def index(self):
        if self.position is None:
            return np.prod(self.board_shape)
        return self.position[0] * self.board_shape[0] + self.position[1]

    def __init__(self, 
                 position: tuple[int, int], 
                 captures: list[tuple[int, int]], 
                 black_to_move: bool, 
                 null_moves: int, 
                 board_shape: tuple[int, int]):
        self.position = position
        self.black_to_move = black_to_move
        self.null_moves = null_moves
        self.board_shape = board_shape
        self.flip_mask = np.zeros(shape=board_shape, dtype=np.bool)
        for line in captures:
            for i, j in line:
                self.flip_mask[i, j] = True

    @classmethod
    def get_null_move(cls, game: 'Othello') -> 'OthelloMove':
        return OthelloMove(None, [], game.black_to_move, game.null_moves, game.shape)

    def __str__(self):
        return self.algebraic


class Othello(BaseGame):

    n_possible_outcomes = 3
    n_actions = 65 # TODO GameEnv should handle this 
    obs_shape = (2, 8, 8) # TODO GameEnv should handle this

    @property
    def player_idx(self):
        return 0 if self.black_to_move else 1
    
    @player_idx.setter
    def player_idx(self, value): 
        pass
        
    @property
    def player_board(self):
        return self.board[self.player_idx, :, :]
        
    @property
    def opponent_board(self):
        return self.board[1 - self.player_idx, :, :]
    
    @property
    def empty_positions(self):
        return zip(*np.where(np.all(~self.board, axis=0)))
    
    @property
    def is_over(self):
        return len(list(self.empty_positions)) == 0 or self.null_moves >= 2
    
    @is_over.setter
    def is_over(self, value):
        pass
    
    @property
    def result(self):
        if not self.is_over:
            return None
        if self.material_diff > 0:
            return 0 # black idx
        if self.material_diff < 0:
            return 1 # white idx
        return 2 # draw
    
    @result.setter
    def result(self, value):
        pass
    
    @property
    def material_diff(self):
        return np.sum(self.board[0, :, :]) - np.sum(self.board[1, :, :])

    def __init__(self, size: int = 8):
        self.size = size
        self.shape = (size, size)
        
        self.board = np.zeros(shape=(2, *self.shape), dtype=np.bool)
        center_idx = self.size // 2 - 1

        # initial white tokens
        self.board[1, center_idx, center_idx] = True
        self.board[1, center_idx + 1, center_idx + 1] = True

        # initial black tokens
        self.board[0, center_idx + 1, center_idx] = True
        self.board[0, center_idx, center_idx + 1] = True

        self.black_to_move = True
        self.null_moves = 0

        self.obs_shape = self.board.shape
        self.n_actions = size * size + 1

    def __get_captures_from_position(self, i: int, j: int) -> list[Positions]:
        captures = []
        directions = get_neighbor_diffs(i, j, self.shape)
        for di, dj in directions:
            position = (i + di, j + dj)
            if not self.opponent_board[position]:
                continue

            capture = [position]
            while True:
                position = (position[0] + di, position[1] + dj)

                if not is_in_limits(*position, self.shape):
                    break
                if not self.opponent_board[position]:
                    break

                capture.append(position)                    
            
            if is_in_limits(*position, self.shape) and self.player_board[position]:
                captures.append(capture)

        return captures

    def get_moves(self) -> list[OthelloMove]:
        moves = []

        for i, j in self.empty_positions:
            captures = self.__get_captures_from_position(i, j)
            if len(captures) == 0:
                continue
            moves.append(OthelloMove((i, j), captures, self.black_to_move, self.null_moves, self.shape))

        if len(moves) == 0:
            moves.append(OthelloMove.get_null_move(self))

        return moves

    def get_random_move(self) -> OthelloMove:
        # TODO use seeded _rng
        # raise NotImplementedError()
        return np.random.choice(self.get_moves())

    def get_move_from_index(self, index: int) -> OthelloMove:
        if index < 0:
            return OthelloMove(None, [], self.black_to_move, self.null_moves, self.shape)
        i = index // self.size
        j = index % self.size
        captures = self.__get_captures_from_position(i, j)
        return OthelloMove((i, j), captures, self.black_to_move, self.null_moves, self.shape)

    def make_move(self, move: OthelloMove):
        if move.position is not None:
            assert(move.black_to_move == self.black_to_move)
            assert(not self.player_board[move.position] and not self.opponent_board[move.position])
            self.player_board[move.position] = True
            self.player_board[move.flip_mask] = True
            self.opponent_board[move.flip_mask] = False
            self.null_moves = 0
        else:
            self.null_moves += 1
        self.black_to_move = not self.black_to_move

    def undo_move(self, move: OthelloMove):
        if move.position is not None:
            assert(move.black_to_move != self.black_to_move)
            self.opponent_board[move.position] = False
            self.opponent_board[move.flip_mask] = False
            self.player_board[move.flip_mask] = True
            self.null_moves = move.null_moves
        else:
            self.null_moves -= 1
        self.black_to_move = not self.black_to_move

    def evaluate(self) -> float:
        value = self.material_diff * (1 if self.black_to_move else -1)
        if self.is_over:
            return np.sign(value)
        
        return value / np.prod(self.shape)

    def render(self, show_legal_moves: bool=True):
        if show_legal_moves:
            moves = [move.position for move in self.get_moves()]

        top_row = f"  {'  '.join([chr(ord('a') + i) for i in range(self.size)])} "
        print(top_row)
        for i in range(self.size):
            print(f"{i+1} ", end="")
            for j in range(self.size):
                if self.board[0, i, j]:
                    print('X  ', end="")
                    continue
                if self.board[1, i, j]:
                    print('O  ', end="")
                    continue
                if show_legal_moves and (i, j) in moves:
                    print('.  ', end="")
                    continue
                print('   ', end="")

            print("")

    def copy(self) -> 'Othello':
        new_game = Othello(self.size)
        new_game.board = np.copy(self.board)
        new_game.black_to_move = self.black_to_move
        new_game.null_moves = self.null_moves
        return new_game
    
    def get_obs(self, obs_mode: Literal["flat", "image"]) -> npt.NDArray:
        # always return from the perspective of the current player
        obs = np.stack([self.player_board, self.opponent_board]).astype(np.float32)
        if obs_mode == "flat":
            obs = obs.flatten()
        return obs
    
    def action_masks(self) -> list[bool]:
        mask = [False] * (self.size ** 2 + 1)
        for move in self.get_moves():
            mask[move.index] = True
        mask[-1] = not np.any(mask[:-1])
        return mask

    def get_move_from_action(self, action: int) -> BaseMove:
        if action == self.n_actions - 1:
            return OthelloMove.get_null_move(self)
        move = next(filter(lambda move: move.index == action, self.get_moves()))
        return move

    def get_move_from_user_input(self, user_input: str) -> BaseMove:
        if len(user_input) != 2:
            raise ValueError("Invalid move")
        j = ord(user_input[0]) - ord('a')
        i = int(user_input[1]) - 1
        if not is_in_limits(i, j, self.shape):
            raise ValueError("Invalid move")
        try:
            move = next(filter(lambda move: move.position == (i, j), self.get_moves()))
        except:
            raise ValueError("Invalid move")
        return move


    def __str__(self):
        s = ""
        for i in range(self.size):
            for j in range(self.size):
                if self.board[0, i, j]:
                    s += 'X'
                elif self.board[1, i, j]:
                    s += 'O'
                else:
                    s += '.'
        s += str(self.player_idx)
        s += str(self.null_moves)
        return s


if __name__ == '__main__':
    pass