import logging
import time

import torch
import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO

from core import GameEnv, MatchManager
from core.connection_manager import CMInstance
from games import Othello
from games.external_othello import ExternalOthello
from players import BogoPlayer, ExternalPlayer, HumanPlayer, OthelloMobilityPlayer, OthelloPositionalPlayer, RLPlayer
import logger_setup

logger_setup.setup(stream=True)
# # logger_setup.setup(filename=None, stream=False)
# # # mcts = ExternalPlayer('othello', 'mcts', maxiters=200)
# minimax = ExternalPlayer(algorithm='minimax', depth = 1, log_info=True)
# # # pos = OthelloPositionalPlayer()
# # # bogo = BogoPlayer(0)
# # # model_mlp = MaskablePPO.load('./models/test_agent_2')
# # # agent_mlp = RLPlayer(model_mlp, deterministic=True, obs_mode="flat")

# env = gym.make('GameEnv-v0', 
#                opponent=minimax, 
#                game_type=ExternalOthello, 
#                obs_mode="flat", 
#                n_random_moves=4,
#                verbose=0)

# # model = MaskablePPO.load('./models/test_agent_2', env=env)

# # # policy_kwargs = dict(
# # #     features_extractor_class=SimpleCNN,
# # #     features_extractor_kwargs=dict(features_dim=256),
# # #     activation_fn = torch.nn.ReLU
# # # )
# # # model = MaskablePPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

# policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[128, 128])
# model = MaskablePPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

# model

# # # print(model.policy)

# model.learn(200_000)
# # # model.save('./models/test_agent_2')





# # model_cnn = MaskablePPO.load('./models/cnn_test_agent')
model_mlp = MaskablePPO.load('./models/test_agent_2')

# # print(model_cnn)
# # n_params = sum(p.numel() for p in model_cnn.policy.parameters() if p.requires_grad)
# # print(n_params)

# # n_params = sum(p.numel() for p in model_mlp.policy.parameters() if p.requires_grad)
# # print(n_params)

# # agent_cnn = RLPlayer(model_cnn, deterministic=True, obs_mode="image")
agent_mlp = RLPlayer(model_mlp, deterministic=True, obs_mode="flat")
# pos = OthelloPositionalPlayer()
# mob = OthelloMobilityPlayer()
# bogo = BogoPlayer(0)
minimax = ExternalPlayer('minimax', depth = 6, log_info=False)
mcts = ExternalPlayer('mcts', maxiters=500, log_info=False)
# human = HumanPlayer()

mm = MatchManager(
    players=[mcts, mcts],
    player_names=['mcts_1', 'mcts_2'],
    game_type=ExternalOthello, 
    n_games=250,
    mirror_games=True,
    n_random_moves=4,
    verbose=2,
    seed=0)

print("start")

start = time.time()
mm.run()
print(time.time() - start)





# obs = torch.tensor(othello.get_obs(obs_mode="flat")).unsqueeze(dim=0).to('cuda')
# print(model_mlp.policy.predict_values(obs))