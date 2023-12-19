"""The public goods game involves a public pool of resources that scales at each step (can easily be held at 1 step.)
Each agent may put as much as they want into the pool, but each recieves equally from it. The most optimal situation 
is when every agent puts everything into the pool. However, the nash equalibrium is no one contributing. """


import gym
import random


class ContinuousPublicGoodsGameEnv(gym.Env):
    def __init__(self, multiplier=1.5):
        super(ContinuousPublicGoodsGameEnv, self).__init__()
        
        self.game = ContinuousPublicGoodsGame(multiplier=multiplier)

    def reset(self): #Only the agents need to reset, not the sim
        return None
    
    def step(self, actions): #Not sure if we want multiple steps, so done might not be meaningful
        profit = self.game.step(actions)
        done = False        
        return profit, done


class ContinuousPublicGoodsGame:
    def __init__(self, multiplier=1.5):
        self.multiplier = multiplier

    #input: array of contributions
    #output: float of how much each person gets from the public pot
    def step(self, contributions):
        total_contribution = sum(contributions)
        public_goods_value = self.multiplier * total_contribution
        individual_profit = public_goods_value / len(contributions)
        return individual_profit
    

class Agent:
    def __init__(self):
        self.money = 20
        self.contribution = 0

    def contribute(self):
        self.contribution = random.random() * self.money
        self.money -= self.contribution
        return self.contribution

    def get_reward(self, profit):
        reward = (self.money + profit) / (self.money + self.contribution)
        self.money += profit
        return reward

    def reset(self): #might want to call this at each iteration
        self.money = 20


# Initialize environment and agent
num_agents=2
env = ContinuousPublicGoodsGameEnv(multiplier=1.5)
agents = [Agent() for i in range(num_agents)]


# Training loop
for episode in range(3):
    state = env.reset()
    for agent in agents:
        agent.reset()
    done = False
    
    while not done:
        actions = [agent.contribute() for agent in agents]
        print(f'contributions: {actions}')

        profit, done = env.step(actions)

        rewards = [agent.get_reward(profit) for agent in agents]
        print(f'rewards: {rewards}')
        
        done = True
        # Learning step; replace with agent.learn or equivalent
        # agent.learn(state, actions, rewards, next_state, done)

