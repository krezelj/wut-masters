from copy import deepcopy

import numpy as np
import numpy.typing as npt
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn
from stable_baselines3.common.vec_env import DummyVecEnv

from core.game_env import GameEnv


global_opponent: MaskablePPO

def set_global_opponent(opponent: MaskablePPO):
    global global_opponent
    global_opponent = opponent


def get_unready_envs_idxs(envs: list[GameEnv]) -> list[int]:
    unready_envs = []
    for idx, env in enumerate(envs):
        if env.is_ready():
            continue
        unready_envs.append(idx)
    return unready_envs


def handle_opponent(envs: list[GameEnv], rollout_data: list[tuple], update_rollout: bool):
    unready_envs_idxs = get_unready_envs_idxs(envs)
    while len(unready_envs_idxs) > 0:

        batch_obs = []
        action_masks = []
        for env_idx in unready_envs_idxs:
            if update_rollout:
                obs = rollout_data[env_idx][0]
            else:
                obs = envs[env_idx]._get_obs()
            batch_obs.append(np.expand_dims(obs, axis=0))
            action_masks.append(envs[env_idx].action_masks())

        opponent_actions, _ = global_opponent.predict(np.concat(batch_obs), action_masks=action_masks)

        for i, env_idx in enumerate(unready_envs_idxs):
            updated_data = envs[env_idx].step(opponent_actions[i])
            if update_rollout:
                rollout_data[env_idx] = updated_data

        unready_envs_idxs = get_unready_envs_idxs(envs)


def batch_step_wait(self: DummyVecEnv) -> VecEnvStepReturn:
    """
    Custom step_wait function that delays collecting rollout data until after
    all environments have processed the opponent move. This allows for batching
    the opponent responses to make the training more efficient.

    This should only be used when the opponent is a different model (e.g. self-play),
    as it also assumes that the opponent handles exactly the same observation space.
    """
    unwrapped_envs: list[GameEnv] = [env.unwrapped for env in self.envs]

    # list of (obs, reward, terminated, truncated, info) tuples
    rollout_data = [None] * self.num_envs

    # assert all envs are ready (waiting for the agent i.e. player_idx=0)
    unready_envs_idxs = get_unready_envs_idxs(unwrapped_envs)
    assert(len(unready_envs_idxs) == 0)

    # step through each env with corresponding action
    # collect rollout data (it will be updated if necessary)
    for env_idx in range(self.num_envs):
        rollout_data[env_idx] = unwrapped_envs[env_idx].step(self.actions[env_idx])

    handle_opponent(unwrapped_envs, rollout_data, update_rollout=True)

    for env_idx in range(self.num_envs):
        obs, self.buf_rews[env_idx], terminated, truncated, self.buf_infos[env_idx] = rollout_data[env_idx]
        self.buf_dones[env_idx] = terminated or truncated
        self.buf_infos[env_idx]["TimeLimit.truncated"] = truncated and not terminated

        if self.buf_dones[env_idx]:
            self.buf_infos[env_idx]["terminal_observation"] = obs
            obs, self.reset_infos[env_idx] = self.envs[env_idx].reset()
        self._save_obs(env_idx, obs)

    # after the reset, it may be opponents turn so we must handle it once again
    # here we assume no additional reset is required, so n_random_moves must be sufficiently low
    # TODO consider using unbatched opponent handling in reset 
    # this will be a bit slower but more reliable and maintainable
    handle_opponent(unwrapped_envs, rollout_data, update_rollout=False)

    # assert all envs are ready (waiting for the agent i.e. player_idx=0)
    unready_envs_idxs = get_unready_envs_idxs(unwrapped_envs)
    assert(len(unready_envs_idxs) == 0)

    return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), deepcopy(self.buf_infos))