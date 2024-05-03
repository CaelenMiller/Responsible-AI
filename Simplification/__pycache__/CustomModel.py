
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from torch import nn
import torch

'''This is an example custom model for use with RLlib. Look at [TrainGridWorld.py] for example usage'''
class CustomModel(TorchModelV2, nn.Module):
    #The inputs need to be exactly this or RLlib will die. 
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.obs_size = kwargs["obs_size"]
        self.action_size = kwargs["action_size"]
        self.layers = kwargs["num_layers"]
        self.hidden_dim = kwargs["hidden_dim"]

        self.actor_input = nn.Linear(self.obs_size, self.hidden_dim)

        #Actor critic methods such as PPO or SAC need both to function
        self.actor = nn.Sequential(
            *[nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LeakyReLU(),
                nn.BatchNorm1d(self.hidden_dim)
            ) for _ in range(self.layers)]
        )
        self.actor_head = nn.Linear(self.hidden_dim, self.action_size)


        self.critic_input = nn.Linear(self.obs_size, self.hidden_dim)

        self.critic = nn.Sequential(
            *[nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LeakyReLU(),
                nn.BatchNorm1d(self.hidden_dim)
            ) for _ in range(self.layers)]
        )

        # Output layer
        self.critic_head = nn.Linear(self.hidden_dim, 1)

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        # Assuming the input dict contains 'obs' key with the observation
        obs = input_dict["obs"]

        self.value = self.critic_head(self.critic(self.critic_input(obs)))

        self.action = self.actor_head(self.actor(self.actor_input(obs)))

        return self.action, state
    
    @override(TorchModelV2)
    def value_function(self):
        assert self.value is not None, "must call forward first!"
        return torch.reshape(self.value, [-1]) #reshape into a single dimension

    