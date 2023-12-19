import gym
import numpy as np
import random
import torch

class Environment(gym.Env):
    def __init__(self, display_sim=True, brain=None):
       
        # Define action and observation space
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=1000, shape=(2,), dtype=np.float32)

        self.reset()


    def reset(self):
        self.target = 0
        self.current_angle = random.randint(0,360)
        self.actions = [0,0]
        self.done = False
        self.time = 0



    def get_reward(self): 
        reward = 6 - (min(abs(self.current_angle - self.target), abs(self.current_angle - self.target - 360)))/30
        return torch.from_numpy(np.array([reward/100]))


    def get_state(self):
        return torch.from_numpy(np.array([self.target, self.current_angle]))


    def step(self):
        self.time += 1
        if self.actions[0]:
            self.current_angle -= 1
        if self.actions[1]:
            self.current_angle == 1
        
        self.current_angle %= 360

        if abs(self.target - self.current_angle) == 1:
            self.done = True
        
        return self.get_state(), self.get_reward(), self.done
    

from Environment import *
from Brain import *

from collections import deque
import random


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


def train_on_game():
  # Initialize environment and agents
  brain = Brain(model="DQN")
  env = Environment(brain=brain)

  # Training settings
  num_episodes = 50
  batch_size = 32

  epsilon_start = 1.0
  epsilon_final = 0.01
  epsilon_decay = 50
  global_step = 0 
  epsilon = epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * global_step / epsilon_decay)


  for episode in range(num_episodes):
    # Initialize replay memory
    memory = ReplayBuffer(320)

    env.reset()

    while env.time < 100:  # Set a max number of steps per episode
        explore = random.random() < epsilon
        global_step += 1
        epsilon = epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * global_step / epsilon_decay)
        brain.init_explore()
        print(f'Exploring: {explore}')
        for i in range(8):

            prev_state = env.get_state()
            if explore:
                actions = torch.tensor(brain.explore())
            else:
                output = brain(prev_state)
                actions = (output > 0.4)
                if i == 7:
                    print(actions, output)
            env.actions = actions
            state, rewards, done = env.step()

            memory.push(prev_state, actions, rewards, state, done)

            if len(memory) > batch_size:
                env.actor.learn(memory)

            if done:
                print(f'Found correct position in {env.time}')
                break
        if done:
            break


train_on_game()

