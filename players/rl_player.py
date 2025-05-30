from typing import Literal
from stable_baselines3.common.base_class import BaseAlgorithm
import torch

from games.base_game import BaseGame, BaseMove
from players.base_player import BasePlayer


class RLPlayer(BasePlayer):

    def __init__(self, model: BaseAlgorithm, obs_mode: Literal["flat", "image"], deterministic: bool = True):
        super().__init__()
        self.model = model
        self.obs_mode = obs_mode
        self.deterministic = deterministic

    def get_move(self, game: BaseGame) -> BaseMove:
        action_masks = game.action_masks()
        with torch.no_grad():
            action, _ = self.model.predict(game.get_obs(self.obs_mode), action_masks=action_masks, deterministic=self.deterministic)
        return game.get_move_from_action(action)