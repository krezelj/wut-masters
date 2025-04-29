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

logger_setup.setup(filename=path(log_path, "model_init"))


def get_env(opponent, idx):
    return gym.make(
        'GameEnv-v0', 
        opponent=opponent, 
        game_type=ExternalOthello, 
        obs_mode="flat", 
        n_random_moves=0, 
        verbose=2,
        player_names=['agent', 'opponent'],
        csv_filename=path(csv_path, f'model_{idx + 1}')
    )

# find how many models already trained
min_model_idx = 0
while True:
    if os.path.isfile(path(model_path, f"model_{min_model_idx}")):
        min_model_idx += 1
    else:
        break
min_model_idx -= 1
assert(min_model_idx >= 0)

N = 20
for i in range(min_model_idx, N):
    model_frozen = MaskablePPO.load(path(model_path, f"model_{i}"))
    env = get_env(model_frozen)

    model = MaskablePPO.load(path(model_path, f"model_{i}"))
    model.learn(250_000)
    model.save(path(model_path, f"model_{i+1}"))
