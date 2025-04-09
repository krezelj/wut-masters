from abc import ABC


class BaseGame(ABC):

    def __init__(self):
        super().__init__()
    
    def get_moves(self):
        pass