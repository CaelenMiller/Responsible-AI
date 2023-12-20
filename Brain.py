import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torch.optim as optim
from collections import deque
import os


ACTOR_STATE_SIZE = 6
ENV_STATE_WIDTH = 7


class Brain:
    def __init__(self, model=None, optimizer_cls=optim.Adam, lr=0.001, batch_size=64):
        if model == "DQN":
            self.model = DQN()
            self.optimizer = optimizer_cls(self.model.parameters(), lr=lr)
        else:
            self.model = DQN()

        self.batch_size = batch_size
        self.optimizer = optimizer_cls(self.model.parameters(), lr=lr)

    def save_model(self, directory, filename):
        """
        Saves the model's state dictionary and the optimizer's state.

        Parameters:
        - directory (str): The path to the directory where to save the model.
        - filename (str): The name for the saved model file.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = os.path.join(directory, filename)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            file_path,
        )
        print(f"Model saved to {file_path}")

    def activate(self, inputs):
        outputs = self.model.forward(inputs.float().reshape(-1, 49 + ACTOR_STATE_SIZE))
        return outputs

    def init_explore(self):
        self.explore_outputs = [0, 0, 0]
        self.explore_outputs[random.randint(0, 2)] = 1
        if random.randint(0, 10) <= 3:
            self.explore_outputs[0] = 1
        self.explore_outputs = torch.tensor(self.explore_outputs).reshape(1, 3)

    def explore(self):
        return self.explore_outputs

    def learn(self, memory, reward=False, gamma=0.99):
        if self.model.name == "DQN":
            states, actions, rewards, next_states, _ = memory.sample(self.batch_size)
            states = (
                torch.stack(states)
                .reshape((-1, ENV_STATE_WIDTH * ENV_STATE_WIDTH + ACTOR_STATE_SIZE))
                .float()
            )
            actions = torch.stack(actions).reshape((-1, 3)).float()
            rewards = torch.stack(rewards).reshape((-1, 1)).float()
            if not reward:
                rewards = rewards.new_full(rewards.size(), reward)

            next_states = (
                torch.stack(next_states)
                .reshape((-1, ENV_STATE_WIDTH * ENV_STATE_WIDTH + ACTOR_STATE_SIZE))
                .float()
            )

            # Get the Q-values of the actions actually taken
            current_q_values = self.model(states)
            current_q_values = (current_q_values * actions).sum(dim=1).unsqueeze(1)

            # Compute the maximum Q-value, detach it from the graph to prevent backprop
            max_next_q_values = self.model(next_states).detach().max(1)[0].unsqueeze(1)

            # Compute the target Q-values
            target_q_values = rewards + (gamma * max_next_q_values)

            # Compute loss
            loss = F.mse_loss(current_q_values, target_q_values)

            # Zero gradients, perform a backward pass, and update the weights.
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.name = "DQN"

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)

        # Fully connected layers
        # We start with a 7x7 grid, after two 3x3 convolutions with stride 1, we end up with a 3x3 grid.
        # 32 channels and 3x3 grid makes it 32*3*3
        self.fc1 = nn.Linear(ACTOR_STATE_SIZE + ((ENV_STATE_WIDTH - 4) ** 2) * 32, 128)

        # Hidden layers
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 64)

        # Output layer
        self.fc5 = nn.Linear(64, 3)

    def forward(self, x):
        # print(x.shape)
        # Separate the spatial and non-spatial components of the state
        spatial_state = (
            x[:, :-ACTOR_STATE_SIZE]
            .view(-1, 1, ENV_STATE_WIDTH, ENV_STATE_WIDTH)
            .float()
        )
        non_spatial_state = x[:, -ACTOR_STATE_SIZE:].float()

        # print(f'spatial_state: {spatial_state.shape}')
        # print(f'non_spatial_state: {non_spatial_state.shape}')

        # Convolutional layers
        x = F.leaky_relu(self.conv1(spatial_state))
        x = F.leaky_relu(self.conv2(x))

        # # Flatten
        x = x.view(x.size(0), -1)

        # Concatenate with non-spatial state
        x = torch.cat((x, non_spatial_state), dim=1)

        # Fully connected layers
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))

        # Output layer
        x = self.fc5(x)
        x = torch.softmax(x, dim=1)

        return x


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


def get_gradient_stats(model):
    grad_stats = {"mean": [], "max": [], "min": [], "std": [], "norm": []}

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_stats["mean"].append(param.grad.data.mean())
            grad_stats["max"].append(param.grad.data.max())
            grad_stats["min"].append(param.grad.data.min())
            grad_stats["std"].append(param.grad.data.std())
            grad_stats["norm"].append(param.grad.data.norm(2))

    grad_stats["mean"] = torch.tensor(grad_stats["mean"]).mean().item()
    grad_stats["max"] = torch.tensor(grad_stats["max"]).max().item()
    grad_stats["min"] = torch.tensor(grad_stats["min"]).min().item()
    grad_stats["std"] = torch.tensor(grad_stats["std"]).mean().item()
    grad_stats["norm"] = torch.tensor(grad_stats["norm"]).sum().item()

    return grad_stats
