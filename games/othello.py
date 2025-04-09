import numpy as np

from games.base_game import BaseMove
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

    def __init__(self, position, capture_lines, black_to_move, null_moves, board_shape):
        self.position = position
        self.black_to_move = black_to_move
        self.null_moves = null_moves
        self.board_shape = board_shape
        self.flip_mask = np.zeros(shape=board_shape, dtype=np.bool)
        for line in capture_lines:
            for i, j in line:
                self.flip_mask[i, j] = True


class Othello:

    @property
    def player_idx(self):
        return 0 if self.black_to_move else 1
    
    @property
    def player_tokens(self):
        return self.board[self.player_idx, :, :]
        
    @property
    def opponent_tokens(self):
        return self.board[1 - self.player_idx, :, :]
    
    @property
    def empty_positions(self):
        return zip(*np.where(np.all(~self.board, axis=0)))
    
    @property
    def is_over(self):
        return len(list(self.empty_positions)) == 0 or self.null_moves >= 2
    
    @property
    def material_diff(self):
        return np.sum(self.board[0, :, :]) - np.sum(self.board[1, :, :])

    def __init__(self, size: int = 8):
        self.size = size
        self.shape = (size, size)
        
        self.board = np.zeros(shape=(2, *self.shape), dtype=np.bool)
        center_idx = self.size // 2 - 1

        # initial black tokens
        self.board[0, center_idx, center_idx] = True
        self.board[0, center_idx + 1, center_idx + 1] = True

        # initial white tokens
        self.board[1, center_idx + 1, center_idx] = True
        self.board[1, center_idx, center_idx + 1] = True

        self.black_to_move = True
        self.null_moves = 0

    def get_moves(self) -> list[OthelloMove]:
        moves = []

        for i, j in self.empty_positions:
            capture_lines = []

            has_opponent_token = lambda i, j: self.opponent_tokens[i, j]
            directions = get_neighbor_diffs(i, j, self.shape)

            for di, dj in directions:
                position = (i + di, j + dj)
                if has_opponent_token[*position]:
                    continue

                capture_line = [position]
                while True:
                    position = (position[0] + di, position[1] + dj)

                    if not is_in_limits(*position, self.shape):
                        break
                    if not self.opponent_tokens[position]:
                        break

                    capture_line.append(position)                    
                
                if is_in_limits(*position, self.shape) and self.player_tokens[position]:
                    capture_lines.append(capture_line)
            
            if len(capture_lines) == 0:
                continue

            moves.append(OthelloMove((i, j), capture_lines, self.black_to_move, self.null_moves, self.shape))

        if len(moves) == 0:
            moves.append(OthelloMove(None, capture_lines, self.black_to_move, self.null_moves, self.shape))

        return moves

    def make_move(self, move: OthelloMove):
        if move.position is not None:
            assert(move.black_to_move == self.black_to_move)
            self.player_tokens[move.position] = True
            self.player_tokens[move.flip_mask] = True
            self.opponent_tokens[move.flip_mask] = False
            self.null_moves = 0
        else:
            self.null_moves += 1
        self.black_to_move = not self.black_to_move

    def undo_move(self, move: OthelloMove):
        if move.position is not None:
            assert(move.black_to_move != self.black_to_move)
            self.opponent_tokens[move.position] = False
            self.opponent_tokens[move.flip_mask] = False
            self.player_tokens[move.flip_mask] = True
            self.null_moves = move.null_moves
        else:
            self.null_moves -= 1
        self.black_to_move = not self.black_to_move

    def evaluate(self) -> float:
        value = self.material_diff * (1 if self.black_to_move else -1)
        if self.is_over:
            return np.sign(value)
        
        return value / np.prod(self.shape)

    def render(self, show_legal_moves=False):
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

    def copy(self):
        new_game = Othello(self.size)
        new_game.board = np.copy(self.board)
        new_game.black_to_move = self.black_to_move
        new_game.null_moves = self.null_moves

if __name__ == '__main__':
    pass