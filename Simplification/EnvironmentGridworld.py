import gymnasium as gym
from gymnasium.envs.registration import EnvSpec
from gymnasium import spaces
import numpy as np

'''A simple gridworld environment, the goal is for the agent to get to a 2 in the grid'''
class CustomGridWorldEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, env_config={}):
        super(CustomGridWorldEnv, self).__init__()

        # EnvSpec is used for RLlib, if max_episode_steps is not defined it will throw more errors
        self.spec = EnvSpec
        self.spec.max_episode_steps = 50
        self.spec.id = "Grid-v0"

        self.actions = [(1, 0), (-1, 0), (0, -1), (0, 1), (0,0)] #NSWE

        self.gridWorld = GridWorld()

        self.display = Display(self.gridWorld.grid)

        obs_size = self.gridWorld.shape[0] * self.gridWorld.shape[1] + 2

        self.action_space = spaces.Discrete(5)  # 0: up, 1: down, 2: left, 3: right, 4: stay
        self.observation_space = spaces.Box(low=-15, high=15, shape=(obs_size,), dtype=np.float32)

        self.time = 0

        self.reset()


    def reset(self, *,seed=None, options=None):
        self.time = 0

        self.agent_position = self.gridWorld.get_random_location() #random valid location
        self.dist = self.gridWorld.find_distance_to_goal(self.agent_position[0], self.agent_position[1])

        #print(f'{max(self.embedding)} : {min(self.embedding)} - {command}')

        state = self.get_observation()

        #print(f'Goal: {self.current_target_type}, {command}')

        return state, {}

    def step(self, action):

        self.time+=1

        if self.time >= self.spec.max_episode_steps:
            truncated = True
        else:
            truncated = False

        self.apply_action(self.actions[action])

        # Calculate reward
        reward = self.calculate_reward()

        # Check if the agent reached one of the targets
        done = self.is_done()

        # Optional: return additional info
        info = {}

        return self.get_observation(), reward, done, truncated, info
    
    def apply_action(self, action): #takes in an action tuple from self.actions

        #enforce movement being in bounds
        next_x = max(0, min(self.agent_position[0] + action[0], self.gridWorld.shape[0]-1))
        next_y = max(0, min(self.agent_position[1] + action[1], self.gridWorld.shape[1]-1))

        if self.gridWorld.grid[next_y][next_x] != 1: #Not in a wall
            self.agent_position[0] = next_x
            self.agent_position[1] = next_y


    def render(self, render_mode='human'):
        if render_mode == "human":
            self.display.render(self.gridWorld.grid, self.agent_position)
            self.display.clock.tick(self.display.FPS)

    def calculate_reward(self):
        prev_dist = self.dist
        self.dist = self.gridWorld.find_distance_to_goal(self.agent_position[1], self.agent_position[0])
        if self.dist == 0:
            return 5
        else: 
            return prev_dist - self.dist

    def is_done(self):
        if self.gridWorld.grid[self.agent_position[1]][self.agent_position[0]] == 2:
            return True
        else:
            return False

    def get_observation(self):
        observation = np.concatenate((np.copy(self.gridWorld.grid).flatten(), np.array(self.agent_position, dtype=float)))

        return observation


from collections import deque

class GridWorld:
    def __init__(self, grid = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                        [0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1],
                                        [1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                                        [2, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1],
                                        [1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1],
                                        [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                                        [0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0],
                                        [0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1],
                                        [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 2, 1]])):
        self.grid = grid  # np.array of integers where 0 is path, 1 is wall, and 2 is exit
        self.shape = grid.shape

    def get_random_location(self):
        # Find the indices of all zeros in the grid
        zero_indices = np.argwhere(self.grid == 0)
        # Select a random index from the list of zero indices
        random_index = zero_indices[np.random.randint(0, len(zero_indices))]
        # Return the selected index as a list [x, y]
        return random_index.tolist()


    def find_distance_to_goal(self, start_x, start_y):
        # Check if starting point is a valid path
        if self.grid[start_x, start_y] == 1:  # 1 represents a wall
            return -1  # Return -1 or some indication that the start is invalid

        # Directions: up, down, left, right
        directions = [(1, 0), (-1, 0), (0, -1), (0, 1)]
        visited = set()  # Keep track of visited nodes
        queue = deque([(start_x, start_y, 0)])  # (x, y, distance)

        while queue:
            x, y, dist = queue.popleft()

            # If the current position is the goal, return the distance
            if self.grid[x, y] == 2:
                return dist

            # Add current node to visited
            visited.add((x, y))

            # Explore neighbors
            for dx, dy in directions:
                nx, ny = x + dx, y + dy

                # Check if next position is within grid bounds and not a wall
                if 0 <= nx < self.grid.shape[0] and 0 <= ny < self.grid.shape[1]:
                    if (nx, ny) not in visited and self.grid[nx, ny] != 1:
                        queue.append((nx, ny, dist + 1))

        # If no path is found
        return -1

import pygame 
import sys

class Display:
    def __init__(self, initial_grid, size=(500, 500)):
        pygame.init()
        self.screen = pygame.display.set_mode(size)
        self.size = size
        self.grid = initial_grid

        self.clock = pygame.time.Clock()
        self.FPS = 5

        # Generate a pygame color for each of the integers in the array
        self.colors = [
            pygame.Color(0, 100, 200),  # 0
            pygame.Color(50, 50, 50),  # 1
            pygame.Color(255, 0, 0),  # 2
        ]
        self.agent_color = pygame.Color(50,255,50)
        # Determine the size of the rectangles needed to fit all the squares on the screen
        self.rect_size = min(self.size) // max(self.grid.shape)

    def render(self, grid, agent_pos=[0,0]):
        self.grid = grid
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        self.screen.fill((0, 0, 0))  # Fill the background with black

        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                # Determine the color based on the integer in the grid
                color = self.colors[self.grid[i, j]]
                # Draw the rectangle
                pygame.draw.rect(self.screen, color, 
                                 (j*self.rect_size, i*self.rect_size, self.rect_size, self.rect_size))
                
        pygame.draw.rect(self.screen, self.agent_color, 
                                 (agent_pos[0]*self.rect_size + 3, agent_pos[1]*self.rect_size + 3, self.rect_size - 6, self.rect_size - 6))
                


        pygame.display.flip()

    