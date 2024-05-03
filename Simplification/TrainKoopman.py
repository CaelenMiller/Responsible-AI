from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
import gymnasium as gym

from Models import Koopman_net



# Create the environment
env = gym.make('CartPole-v1')

hidden_dim = 4
action_dim = 2
hidden_layers = 1
obs_dim = 4

# Create the policy with the custom network
policy_kwargs = dict(
    features_extractor_class=Koopman_net,
    features_extractor_kwargs=dict(hidden_dim=hidden_dim, action_dim=action_dim, obs_dim=obs_dim, hidden_layers=hidden_layers)  # Adjust dimensions according to your model
)

# Instantiate the agent with the custom policy
model = PPO(ActorCriticPolicy, env, verbose=1, policy_kwargs=policy_kwargs)

# Train the agent
model.learn(total_timesteps=100000)

model.save("./Models/ppo_koopman")

env.close()