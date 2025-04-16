from abc import ABC

class BaseMove(ABC):

    __slots__ = ['index']

    def __init__(self):
        super().__init__()

class BaseGame(ABC):

    __slots__ = ['player_idx', 'is_over', 'result', 'n_possible_outcomes']

    def __init__(self):
        super().__init__()
    
    def get_moves(self) -> list[BaseMove]:
        raise NotImplementedError()

    def get_random_move(self) -> BaseMove:
        raise NotImplementedError()

    def get_move_from_index(self, index: int) -> BaseMove:
        raise NotImplementedError()

    def make_move(self, move):
        raise NotImplementedError()

    def undo_move(self, move):
        raise NotImplementedError()

    def evaluate(self) -> float:
        raise NotImplementedError()

    def copy(self) -> 'BaseGame':
        raise NotImplementedError()
    
    def render(self):
        raise NotImplementedError()

    def swap_players(self):
        raise NotImplementedError()