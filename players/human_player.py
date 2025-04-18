

from games.base_game import BaseGame, BaseMove
from players.base_player import BasePlayer


class HumanPlayer(BasePlayer):

    def __init__(self):
        super().__init__()

    def get_move(self, game: BaseGame) -> BaseMove:
        game.render()
        while True:
            try:
                user_input = input("Your move: ")
                move = game.get_move_from_user_input(user_input)
                break
            except ValueError:
                print("Invalid move! ", end="")
        return move
