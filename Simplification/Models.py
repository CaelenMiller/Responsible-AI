import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class Koopman_net(BaseFeaturesExtractor):
    def __init__(self, observation_space, hidden_dim=32, action_dim=2, obs_dim=4,  hidden_layers=2):
        super(Koopman_net, self).__init__(observation_space, hidden_layers)

        self.g = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Sequential(
            *[nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.BatchNorm1d(hidden_dim)
            ) for _ in range(hidden_layers)]
        )
        )

        self.K = nn.Linear(hidden_dim, action_dim)

    def forward(self, observations):
        return self.K(self.g(observations))
    
    def g_forward(self, observation):
        return self.g(observation)
    
#Learned g inverse, maps from encoding/hidden to observation space
class G_inv(nn.Module):
    def __init__(self, hidden_dim=32, action_dim=2, obs_dim=4,  hidden_layers=2):
        super(G_inv, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.Sequential(
            *[nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.BatchNorm1d(hidden_dim)
            ) for _ in range(hidden_layers)]
        )
        )

    def forward(self, x):
        return self.network(x)
    

    