from stable_baselines3.common.base_class import BaseAlgorithm

from games.base_game import BaseGame, BaseMove
from players.base_player import BasePlayer


class RLPlayer(BasePlayer):

    def __init__(self, model: BaseAlgorithm, deterministic: bool = False):
        super().__init__()
        self.model = model
        self.deterministic = deterministic

    def get_move(self, game: BaseGame) -> BaseMove:
        action_masks = game.action_masks()
        action, _ = self.model.predict(game.get_obs("flat"), action_masks=action_masks, deterministic=self.deterministic)
        return game.get_move_from_action(action)