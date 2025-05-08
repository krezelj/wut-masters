

from typing import Callable, Optional

import numpy as np
from games.base_game import BaseGame
from players.base_player import BasePlayer


class MCTSPlayer(BasePlayer):

    class Node:

        @property
        def is_terminal(self):
            return self.game.is_over

        def __init__(self, game: BaseGame, parent: Optional['MCTSPlayer.Node'] = None):
            self.game = game
            self.parent = parent
            self.children = None
            self.expanded = False
            self.visit_count = 0
            self.value_sum = 0

        def expand(self):
            self.expanded = True
            moves = self.game.get_moves()

            self.children = []
            for move in moves:
                new_game = self.game.copy()
                new_game.make_move(move)
                self.children.append(MCTSPlayer.Node(new_game, self))

        def get_best_child_index(self, estimator: Callable) -> int:
            max_value = -np.inf
            best_idx = 0
            for i, child in enumerate(self.children):
                value = estimator(child)
                if value > max_value:
                    max_value = value
                    best_idx = i
            return best_idx

        def get_best_child(self, estimator: Callable) -> 'MCTSPlayer.Node':
            return self.children[self.get_best_child_index(estimator)]

    def __init__(self, max_iters: int, estimator: Callable):
        self.max_iters = max_iters
        self.estimator = estimator

    def get_move(self, game: BaseGame):
        self.__nodes = 0

        self.root = MCTSPlayer.Node(game, None)
        self.root.expand()
        self.build_tree()

        best_idx = self.root.get_best_child_index(MCTSPlayer.visit_count_estimator)
        # value = best_child.value_sum / best_child.visit_count
        return game.get_moves()[best_idx]

    def build_tree(self):
        for _ in range(self.max_iters):
            current = self.select()
            if current == None:
                raise ValueError("Selected node is null!")

            if current.visit_count == 2 and not current.is_terminal:
                current.expand()
                current = current.get_best_child(self.estimator)
                current.visit_count += 1

            terminal_state = self.rollout(current.game.copy())
            value = terminal_state.evaluate()
            if terminal_state.player_idx == current.game.player_idx:
                value = -value
            value = np.sign(value)
            
            self.backtrack(current, value)
                
    def select(self) -> 'MCTSPlayer.Node':
        current = self.root
        while current.expanded:
            self.__nodes += 1
            current.visit_count += 1
            current = current.get_best_child(self.estimator)
        current.visit_count += 1
        return current

    def rollout(self, game: BaseGame) -> BaseGame:
        while not game.is_over:
            self.__nodes += 1
            move = game.get_random_move()
            game.make_move(move)
        return game

    def backtrack(self, current: 'MCTSPlayer.Node', value: float):
        while current is not None:
            current.value_sum += value
            value = -value
            current = current.parent
    
    @classmethod
    def visit_count_estimator(cls, node: 'MCTSPlayer.Node'):
        return node.visit_count
    
    @classmethod
    def ucb_estimator(cls, node: 'MCTSPlayer.Node'):
        if node.visit_count == 0:
            return np.inf

        ucb = node.value_sum / node.visit_count + np.sqrt(2 * np.log(node.parent.visit_count) / node.visit_count)
        return ucb