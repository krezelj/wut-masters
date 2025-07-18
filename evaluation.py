import os
import time

import numpy as np
from sb3_contrib import MaskablePPO

from core.match_manager import MatchManager
from players.external_player import ExternalPlayer
from onnx_export import export

def evaluate(
        game_type,
        directory,
        opponent_func, 
        obs_shape,
        result_directory,
        result_name,
        unified,
        n_games,
        n_random_moves,
        min_model_id,
        max_model_id,
        model_step = 1):

    N_MODELS = (max_model_id - min_model_id) // model_step + 1

    results = np.zeros(shape=(N_MODELS, 3, 2))

    for i, model_id in enumerate(range(min_model_id, max_model_id + 1, model_step)):
        print(f"Evaluating model_{model_id}... ", end="", flush=True)
        start = time.time()

        # model = MaskablePPO.load(f'./train/othello_selfplay_mlp_mini/.models/model_{model_id}.zip', device="cpu")
        model = MaskablePPO.load(f'{directory}/model_{model_id}.zip', device="cpu")

        export(model, obs_shape=obs_shape, separate=not unified, unified=unified)
        load_time = time.time()

        agent = ExternalPlayer('agent', modelDirectory="models", deterministic=True, unified=unified)

        # readd minimax since it's using transposition tables in the engine and reusing it 
        # will lead to results very difficult to reproduce
        opponent = opponent_func()
        # opponent = ExternalPlayer('minimax', depth=2, eval_func_name="standard")

        mm = MatchManager(
            players=[agent, opponent],
            player_names=['agent', 'opponent'],
            game_type=game_type, 
            n_games=n_games,
            mirror_games=True,
            allow_external_simulation=True,
            n_random_moves=n_random_moves,
            verbose=0,
            seed=0)
        
        del agent
        del opponent

        mm.run()
        results[i, :, :] = mm.results
        print(f"done ({(load_time - start):.2f}/{(time.time() - load_time):.2f}s)")

    np.save(os.path.join(".results", result_directory, f"{result_name}_results.npy"), results)
    with open(os.path.join(".results", result_directory, f"{result_name}_results.txt"), "w") as f:
        wins = results[:, 0, 0] + results[:, 1, 1]
        losses = results[:, 1, 0] + results[:, 0, 1]
        for model_id in range(len(wins)):
            f.write(f"{int(wins[model_id])};{int(losses[model_id])}\n")