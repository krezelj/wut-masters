from typing import Literal, Optional, Type

import numpy as np
import numpy.typing as npt
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register

from core.match_manager import MatchManager
from games.base_game import BaseGame
from players.base_player import BasePlayer

register(
    id="GameEnv-v0",
    entry_point="core.game_env:GameEnv"
)


class GameEnv(gym.Env):
    
    def __init__(self, 
                 opponent: BasePlayer, 
                 game_type: Type[BaseGame],
                 obs_mode = Literal["flat", "image"],
                 seed: Optional[int] = None,
                 **kwargs):
        super().__init__()
        self._rng = np.random.default_rng(seed=seed)

        self.action_space = spaces.Discrete(game_type.n_actions)
        self.obs_mode = obs_mode
        if obs_mode == "flat":
            self.observation_space = spaces.Box(low=0, high=1, shape=(np.prod(game_type.obs_shape), ), dtype=np.float32)
        elif obs_mode == "image":
            self.observation_space = spaces.Box(low=0, high=255, shape=game_type.obs_shape, dtype=np.uint8)

        self.mm = MatchManager(
            players=[None, opponent], 
            game_type=game_type, 
            n_games=-1,
            mirror_games=True,
            pause_after_game=True,
            allow_external_simulation=False,
            seed=seed,
            **kwargs)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[npt.NDArray, dict]:
        if self.mm.status != self.mm.WAITING:
            self.mm.run()
        return self._get_obs(), self._get_info()

    def step(self, action: int) -> tuple[npt.NDArray, float, bool, bool, dict]:
        # this is possible if a lot of random moves are used
        if self.mm.status != self.mm.WAITING:
            info = self._get_info()
            info['is_success'] = False
            return self._get_obs(), 0, True, False, info
        
        # assert(self.mm.status == self.mm.WAITING)
        move = self.mm.current_game.get_move_from_action(action)
        self.mm.respond(move)
        self.mm.run()

        terminated = False
        reward = 0
        info = self._get_info()
        if self.mm.status in [self.mm.OVER, self.mm.READY]:
            if self.mm.last_winner_idx == 0:
                reward = 1
            elif self.mm.last_winner_idx < 0:
                reward = 0
            else:
                reward = -1
            info['is_success'] = self.mm.last_winner_idx == 0
            terminated = True

        truncated = False
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> npt.NDArray:
        return self.mm.current_game.get_obs(self.obs_mode)

    def _get_info(self) -> dict:
        return {
            'game': self.mm.current_game
            }

    def render(self):
        self.mm.current_game.render()

    def action_masks(self) -> list[bool]:
        return self.mm.current_game.action_masks()