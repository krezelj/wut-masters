from typing import Callable, Literal, Optional

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.base_class import BaseAlgorithm
import torch

from games.base_game import BaseGame
from players.base_player import BasePlayer


class MCTSPlayer(BasePlayer):

    class Node:

        @property
        def is_terminal(self):
            return self.game.is_over

        def __init__(self, game: BaseGame, parent: Optional['MCTSPlayer.Node'], prior_probability: float):
            self.game = game
            self.parent = parent
            self.children = None
            self.expanded = False
            self.visit_count = 0
            self.value_sum = 0
            self.prior_probability = prior_probability

        def expand(self, prior_func: Callable):
            self.expanded = True
            moves = self.game.get_moves()
            probs = prior_func(self.game)

            self.children = []
            for move in moves:
                new_game = self.game.copy()
                new_game.make_move(move)
                self.children.append(MCTSPlayer.Node(new_game, self, prior_probability=probs[move.index]))

        def get_best_child_index(self, estimator: Callable) -> int:
            max_value = -np.inf
            best_idx = 0
            for i, child in enumerate(self.children):
                value = estimator(child)
                if value > max_value:
                    max_value = value
                    best_idx = i
            return best_idx

        def get_best_child(self, estimator: Callable) -> 'MCTSPlayer.Node':
            return self.children[self.get_best_child_index(estimator)]

    def __init__(self, 
                 max_iters: int,
                 simulation_policy: Callable,
                 prior_func: Callable,
                 rollout_policy: Callable,
                 value_estimator: Callable,
                 lam: float = 0.5,
                 ):
        self.max_iters = max_iters
        self.simulation_policy = simulation_policy
        self.prior_func = prior_func
        self.rollout_policy = rollout_policy
        self.value_estimator = value_estimator
        self.lam = lam

    def get_move(self, game: BaseGame):
        self.__nodes = 0

        self.root = MCTSPlayer.Node(game, None, 0)
        self.root.expand(self.prior_func)
        with torch.inference_mode():
            self.build_tree()

        best_idx = self.root.get_best_child_index(MCTSPlayer.visit_count_estimator)
        # value = best_child.value_sum / best_child.visit_count
        return game.get_moves()[best_idx]

    def build_tree(self):
        for _ in range(self.max_iters):
            current = self.select()
            if current == None:
                raise ValueError("Selected node is null!")

            if current.visit_count == 2 and not current.is_terminal:
                current.expand(self.prior_func)
                current = current.get_best_child(self.simulation_policy)
                current.visit_count += 1

            leaf_value = self.evaluate_leaf(current)
            self.backtrack(current, leaf_value)
                
    def select(self) -> 'MCTSPlayer.Node':
        current = self.root
        while current.expanded:
            self.__nodes += 1
            current.visit_count += 1
            current = current.get_best_child(self.simulation_policy)
        current.visit_count += 1
        return current

    def evaluate_leaf(self, leaf: 'MCTSPlayer.Node'):
        if self.lam == 0:
            rollout_value = 0
        else:
            terminal_state = self.rollout(leaf.game.copy())
            rollout_value = terminal_state.evaluate()
            if terminal_state.player_idx == leaf.game.player_idx:
                rollout_value = -rollout_value
            rollout_value = np.sign(rollout_value)
        
        estimator_value = self.value_estimator(leaf.game)
        value = (1 - self.lam) * estimator_value + self.lam * rollout_value
        return value

    def rollout(self, game: BaseGame) -> BaseGame:
        while not game.is_over:
            self.__nodes += 1
            move = self.rollout_policy(game)
            # move = game.get_random_move()
            game.make_move(move)
        return game

    def backtrack(self, current: 'MCTSPlayer.Node', value: float):
        while current is not None:
            current.value_sum += value
            value = -value
            current = current.parent

    @classmethod
    def visit_count_estimator(cls, node: 'MCTSPlayer.Node'):
        return node.visit_count
    
    @classmethod
    def ucb_estimator(cls, node: 'MCTSPlayer.Node'):
        if node.visit_count == 0:
            return np.inf

        ucb = node.value_sum / node.visit_count + np.sqrt(2 * np.log(node.parent.visit_count) / node.visit_count)
        return ucb
    
    @classmethod
    def random_rollout(cls, game: BaseGame):
        return game.get_random_move()
    
    @classmethod
    def get_default_mcts(cls) -> 'MCTSPlayer':
        return MCTSPlayer(
            max_iters=20,
            simulation_policy=MCTSPlayer.ucb_estimator,
            prior_func=lambda game: np.zeros(shape=(65)),
            rollout_policy=MCTSPlayer.random_rollout,
            value_estimator=lambda game: 0,
            lam=1
        )


class NetWrapper(torch.nn.Module):

    def __init__(self, policy, mode: Literal["actor", "critic"]):
        super().__init__()
        self.features_extractor = policy.features_extractor
        self.mlp_extractor = policy.mlp_extractor
        self.mode = mode

        if mode == "actor":
            self.output_head = policy.action_net
        else:
            self.output_head = policy.value_net

    def forward(self, obs):
        features = self.features_extractor(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        if self.mode == "actor":
            return self.output_head(latent_pi)
        else:
            return self.output_head(latent_vf)


class ModelController:

    def __init__(self, model: MaskablePPO, obs_mode: Literal["flat", "image"], obs_shape, c: float = 5):
        self.actor = torch.jit.trace(
            NetWrapper(model.policy, mode="actor"),
            example_inputs=torch.randn(1, *obs_shape)
        )
        self.actor.eval()

        self.critic = torch.jit.trace(
            NetWrapper(model.policy, mode="critic"),
            example_inputs=torch.randn(1, *obs_shape),
        )
        self.critic.eval()

        self.obs_mode = obs_mode
        self.c = c

    def simulation_policy(self, node: 'MCTSPlayer.Node'):
        Q = node.value_sum / (node.visit_count + 1)
        U = self.c * node.prior_probability * np.sqrt(node.parent.visit_count / (1 + node.visit_count))
        return Q + U
    
    def prior_func(self, game: BaseGame):
        action_masks = game.action_masks()
        obs = torch.tensor(game.get_obs(obs_mode=self.obs_mode)).unsqueeze(dim=0)

        logits = self.actor(obs)
        logits[~torch.tensor(action_masks).unsqueeze(dim=0)] = -torch.inf
        probs = torch.softmax(logits, dim=1).detach().squeeze()
        return probs
    
    def rollout_policy(self, game: BaseGame):
        action_masks = game.action_masks()
        obs = torch.tensor(game.get_obs(obs_mode=self.obs_mode)).unsqueeze(dim=0)

        logits = self.actor(obs)
        logits[~torch.tensor(action_masks).unsqueeze(dim=0)] = -torch.inf
        probs = torch.softmax(logits, dim=1).detach().squeeze()

        action = torch.distributions.Categorical(probs=probs).sample()
        return game.get_move_from_action(action)
    
    def value_estimator(self, game: BaseGame):
        obs = torch.tensor(game.get_obs(obs_mode=self.obs_mode)).unsqueeze(dim=0)
        v = self.critic(obs).item()
        return v