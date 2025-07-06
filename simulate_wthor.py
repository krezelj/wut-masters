import torch
from tqdm import tqdm

from games.external_othello import ExternalOthello, ExternalOthelloMove

data = []
with open('moves_data.txt', 'r') as f:
    for line in f.readlines():
        data.append(list(map(lambda x : int(x), line.split(','))))

all_obs = []
all_moves = []
all_results = []
for moves in tqdm(data, unit="game"):
    game = ExternalOthello()
    obs = []
    moves_made = []

    i = 0
    skip_game = False
    while not game.is_over:
        action_mask, legal_moves = game.action_masks(with_moves=True)
        if len(legal_moves) == 1 and legal_moves[0].index == ExternalOthelloMove.null_move_idx:
            move = legal_moves[0] # force the null move
        else:
            move = game.get_move_from_action(moves[i])
            if not action_mask[move.index]:
                skip_game = True
                break
            i += 1

        obs.append(torch.tensor(game.get_obs(obs_mode='image')))
        moves_made.append(torch.tensor(move.index))
        game.make_move(move)

    if skip_game:
        continue
    
    result = game.result
    if result == 0: # black won
        result = 1
    elif result == 1: # white won
        result = -1
    else: # draw
        result = 0
    for _ in range(len(moves_made)):
        all_results.append(result)
        result *= -1

    all_obs.extend(obs)
    all_moves.extend(moves_made)
    game.close()

torch.save(torch.stack(all_obs), 'obs.pt')
torch.save(torch.stack(all_moves), 'moves.pt')
torch.save(torch.tensor(all_results), 'result.pt')

