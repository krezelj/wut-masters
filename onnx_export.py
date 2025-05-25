from typing import Literal
from sb3_contrib import MaskablePPO
import torch

class SingleNet(torch.nn.Module):

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

class ActorCriticNet(torch.nn.Module):

    def __init__(self, policy):
        super().__init__()
        self.features_extractor = policy.features_extractor
        self.mlp_extractor = policy.mlp_extractor
        self.action_net = policy.action_net
        self.value_net = policy.value_net

    def forward(self, obs):
        features = self.features_extractor(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        logits = self.action_net(latent_pi)
        value = self.value_net(latent_vf)
        return logits, value
    

def export(model: MaskablePPO,
           obs_shape: tuple,
           unified: bool = False,
           separate: bool = False):
    
    example_input = torch.randn(1, *obs_shape)
    if unified:
        actor_critic = ActorCriticNet(model.policy)
        torch.onnx.export(
            actor_critic, 
            example_input, 
            "./models/actor_critic.onnx",
            input_names=["input"], 
            output_names=["logits", "value"], 
            opset_version=11,
            dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}, "value": {0: "batch_size"}}
        )
    
    if separate:
        actor = SingleNet(model.policy, mode="actor")
        critic = SingleNet(model.policy, mode="critic")

        torch.onnx.export(
            actor, 
            example_input, 
            "./models/actor.onnx",
            input_names=["input"], 
            output_names=["output"], 
            opset_version=11,
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        )
        torch.onnx.export(
            critic, 
            example_input, 
            "./models/critic.onnx",
            input_names=["input"], 
            output_names=["output"], 
            opset_version=11,
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        )


if __name__ == '__main__':
    model = MaskablePPO.load('./train/othello_selfplay_cnn_batch/.models/model_130.zip', device="cpu")
    # model = MaskablePPO.load('./models/othello_cnn_selfplay/model_72.zip', device="cpu")
    # model = MaskablePPO.load('./train/othello_selfplay_cnn_small_skip_no_pool/.models/model_50.zip', device="cpu")
    # model = MaskablePPO.load('./models/othello_mlp_selfplay/model_70.zip', device="cpu")
    export(model, obs_shape=(2, 8, 8), separate=True)