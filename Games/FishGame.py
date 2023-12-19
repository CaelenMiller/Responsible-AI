"""The fishing game involves 2 fisheries that use the same source. They each need to decide how much they 
fish at each timestep, with the goal of maximizing intake. However, the more the lake is depleted, the fewer fish 
can be extracted. This may be too complicated for our needs."""


import numpy as np

class ContinuousFishingGame:
    def __init__(self, initial_fish=1000, regen_rate=0.1, max_fish=1000):
        self.current_fish = initial_fish
        self.regen_rate = regen_rate
        self.max_fish = max_fish

    def step(self, actions):
        """
        actions: a list of fishing rates for each agent (between 0 and 1)
        returns: a list of rewards for each agent
        """
        total_fishing_rate = sum(actions)
        total_fishing_amount = total_fishing_rate * self.current_fish
        
        rewards = [action * total_fishing_amount / total_fishing_rate for action in actions]

        # Update the environment
        self.current_fish = self.current_fish - total_fishing_amount + self.regen_rate * self.current_fish
        self.current_fish = min(max(0, self.current_fish), self.max_fish)
        
        return rewards

# Initialize the game
game = ContinuousFishingGame()

# Initialize agents' actions (feel free to replace these with RL algorithms)
# Agents choose a random fishing rate between 0 and 1
agent1_action = np.random.uniform(0, 1)
agent2_action = np.random.uniform(0, 1)

# Agents take actions
rewards = game.step([agent1_action, agent2_action])

print(f"Agent 1's action (fishing rate): {agent1_action}")
print(f"Agent 2's action (fishing rate): {agent2_action}")
print(f"Agent 1's reward: {rewards[0]}")
print(f"Agent 2's reward: {rewards[1]}")
print(f"Remaining fish in the lake: {game.current_fish}")
