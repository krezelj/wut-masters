import os
import time

import numpy as np
from sb3_contrib import MaskablePPO

from core.match_manager import MatchManager
from players.external_player import ExternalPlayer
from games.external_othello import ExternalOthello
from onnx_export import export

MIN_MODEL_ID = 129
MAX_MODEL_ID = 130
N_MODELS = MAX_MODEL_ID - MIN_MODEL_ID + 1

results = np.zeros(shape=(N_MODELS, 3, 2))

for i in range(MIN_MODEL_ID, MAX_MODEL_ID + 1):
    print(f"Evaluating model_{i}... ", end="", flush=True)
    start = time.time()

    # model = MaskablePPO.load(f'./models/othello_mlp_selfplay/model_{i}.zip', device="cpu")
    model = MaskablePPO.load(f'./train/othello_selfplay_cnn_batch/.models/model_{i}.zip', device="cpu")
    export(model, obs_shape=(2, 8, 8), separate=True)

    load_time = time.time()

    agent = ExternalPlayer('agent', modelDirectory="models", deterministic="True", unified=False)
    # readd minimax since it's using transposition tables in the engine and reusing it 
    # will lead to results very difficult to reproduce
    minimax = ExternalPlayer('minimax', depth=2, eval_func_name="standard")
    mm = MatchManager(
        players=[agent, minimax],
        player_names=['agent', 'opponent'],
        game_type=ExternalOthello, 
        n_games=1500,
        mirror_games=True,
        allow_external_simulation=True,
        n_random_moves=6,
        verbose=0,
        seed=0)
    
    del agent
    del minimax

    # mm.run_external()
    mm.run()
    results[i - MIN_MODEL_ID, :, :] = mm.results
    print(f"done ({(load_time - start):.2f}/{(time.time() - load_time):.2f}s)")

np.save(os.path.join(".logs", "results.npy"), results)
with open(os.path.join(".logs", "results.txt"), "w") as f:
    wins = results[:, 0, 0] + results[:, 1, 1]
    losses = results[:, 1, 0] + results[:, 0, 1]
    for i in range(len(wins)):
        f.write(f"{int(wins[i])},{int(losses[i])}\n")