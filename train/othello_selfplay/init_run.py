import sys
import os

from sb3_contrib import MaskablePPO
import gymnasium as gym
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import logger_setup
from core import GameEnv
from games.external_othello import ExternalOthello
from players.bogo_player import BogoPlayer

def path(dir, *args):
    return os.path.abspath(os.path.join(dir, *args))

log_path = path(os.path.dirname(__file__), ".logs")
csv_path = path(os.path.dirname(__file__), ".data")
model_path = path(os.path.dirname(__file__), ".models")

print(log_path, csv_path, model_path)

logger_setup.setup(filename=path(log_path, "model_init"))

bogo = BogoPlayer(seed=0)
env = gym.make(
    'GameEnv-v0', 
    opponent=bogo, 
    game_type=ExternalOthello, 
    obs_mode="flat", 
    n_random_moves=0, 
    verbose=2,
    player_names=['agent', 'bogo'],
    csv_filename=path(csv_path, 'model_init')
)

policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[256, 256])
model = MaskablePPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

model.learn(100_000)
model.save(path(model_path, "model_0"))