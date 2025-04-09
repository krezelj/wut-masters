from abc import ABC

class BaseMove(ABC):

    __slots__ = ['index']

    def __init__(self):
        super().__init__()

class BaseGame(ABC):

    def __init__(self):
        super().__init__()
    
    def get_moves(self) -> list[BaseMove]:
        pass

    def get_random_move(self) -> BaseMove:
        pass

    def get_move_from_index(self, index: int) -> BaseMove:
        pass

    def evaluate(self) -> float:
        pass