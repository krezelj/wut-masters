import numpy as np
from sb3_contrib import MaskablePPO

from core.connection_manager import CMInstance
from core.match_manager import MatchManager
from games.external_othello import ExternalOthello
from games.othello import Othello
import logger_setup
from players import ExternalPlayer
from players.bogo_player import BogoPlayer
from players.human_player import HumanPlayer
from players.othello_heuristics import OthelloPositionalPlayer, OthelloMobilityPlayer
from players.rl_player import RLPlayer

logger_setup.setup(stream=False)


minimax = ExternalPlayer(algorithm='minimax', depth=7, log_info=True)

model_mlp = MaskablePPO.load('./models/test_agent_2')
agent_mlp = RLPlayer(model_mlp, deterministic=True, obs_mode="flat")

mm = MatchManager(
    players=[agent_mlp, minimax],
    game_type=ExternalOthello,
    n_games=50,
    mirror_games=True,
    n_random_moves=4,
    seed=0,
    verbose=3
)
mm.run()
