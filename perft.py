import time
from games.base_game import BaseGame
from games.othello import Othello


class Perft:

    def __init__(self, max_depth):
        self.max_depth = max_depth

    def run(self, game: BaseGame):
        self.game = game
        for depth in range(1, self.max_depth + 1):
            self.leaf_nodes = 0
            start = time.time()
            self.search(depth)
            ms_elapsed = (time.time() - start) * 1000

            kns = self.leaf_nodes / max(1, ms_elapsed)
            print(f"depth {depth:>2} | {self.leaf_nodes:>11} | {ms_elapsed:>8.0f}ms | {kns:>6.2f}kN/s |")

    def search(self, depth: int):
        if depth == 0 or self.game.is_over:
            self.leaf_nodes += 1
            return
        
        moves = self.game.get_moves()
        for move in moves:
            self.game.make_move(move)
            self.search(depth - 1)
            self.game.undo_move(move)


def main():
    pass


if __name__ == '__main__':
    othello = Othello(8)
    perft = Perft(6)
    perft.run(othello)