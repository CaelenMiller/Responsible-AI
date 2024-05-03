import torch.nn as nn
from torch import optim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
import gymnasium as gym

from Models import G_inv


env = gym.make('CartPole-v1')
model = PPO.load("./Models/ppo_custom_env_model", env=env)

hidden_dim = 4
action_dim = 2
hidden_layers = 1
obs_dim = 4

# Define the model
g_inv = G_inv(hidden_dim=hidden_dim, action_dim=action_dim, obs_dim=obs_dim, hidden_layers=hidden_layers)

# Loss function
criterion = nn.MSELoss()

# Optimizer
optimizer = optim.Adam(g_inv.parameters(), lr=0.001)

#random actions, reduce overfitting to certain situations
random_actions = [env.action_space.sample() for _ in range(100)]
encodings = model.g_forward(random_actions)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    # Zero the gradients
    optimizer.zero_grad()

    # Forward pass: compute predicted outputs by passing inputs to the model
    outputs = g_inv(encodings)

    # Calculate the loss
    loss = criterion(outputs, random_actions)

    # Backward pass: compute gradient of the loss with respect to model parameters
    loss.backward()

    # Perform a single optimization step (parameter update)
    optimizer.step()

    # Print statistics
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
