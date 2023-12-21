import numpy as np
import math
import pygame
import torch
import sys
import gym
from pygame.locals import *
from DisplayTools import *
from Map import *
from Actor import *


class Environment(gym.Env):
    def __init__(self, display_sim=True, brain=None, max_time = 1024, tile_matrix=np.array([])):
        pygame.init()
        if tile_matrix.any():
            self.map = Map(tile_matrix=tile_matrix)
        else:
            self.map = Map()
        self.actor = Actor(brain=brain) #initial setup and brain implant
        self.reset()
        self.time = 0
        self.display_sim = display_sim
        self.display = Display(self.map)

        self.clock = pygame.time.Clock()
        self.FPS = 75
        self.max_time = max_time

        # Define action and observation space
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=100, shape=(self.map.get_space_size() + 6,), dtype=np.float32)


    #Resets game. Can set if the game is displayed, and if the start is preset or not. 
    def reset(self, display=True, rand_start=False):
        if display:
            self.display_sim = True
        else:
            self.display_sim = False

        self.time = 0
        self.done = False
        self.actor = Actor(brain=self.actor.brain) #resets actor, except for brain
        if rand_start:
            self.actor.x, self.actor.y = self.map.random_valid_location()
        self.entities = [wall for wall in self.map.walls]
        self.entities.append(self.actor)
        for target in self.map.targets:
            self.entities.append(target)


    def get_reward(self): 
        on_target, min_distance, direction = self.map.target_intersections(self.actor.x, self.actor.y)
        self.actor.comp_distance.append(self.map.distance_map[int(self.actor.y//100)][int(self.actor.x//100)])
        reward_scale = 1.1 - self.time/self.max_time   
        
        if on_target: #Reward if on target
            self.done = True
            reward = 100
            
        else: #Reward based on tile_distance
            smoothed_dist, _ = self.map.get_smoothed_distance(int(self.actor.y),int(self.actor.x))
            self.actor.smoothed_distances.append(smoothed_dist)
            #print(f' {self.map.distance_map[int(self.actor.y//100)][int(self.actor.x//100)]}: {smoothed_dist}')
            reward = self.actor.compare_smooth_distances() * 10
            #reward = (self.map.max_dist - (reward / 100)) * 5 / self.map.max_dist
            
            

        if self.actor.time_still > 150: #Ends the run if stuck for 150 frames
            self.done = True
            #reward -= min((self.actor.time_still/200) ** 2 , 5)

        #reward for pointing in the right direction. Not continous to combat weird rotating behavior
        # angle_rewards = [4,2,1,0,-1,-2,-4]
        # diff = min(abs(direction - self.actor.angle_deg), 360 - abs(direction - self.actor.angle_deg))
        # reward += angle_rewards[int(diff//29)] / 6
        #reward += (4 - min(abs(direction - self.actor.angle_deg), 360 - abs(direction - self.actor.angle_deg))//30)/4

        #reward for moving away from current position
        #if len(self.actor.comp_positions) > self.actor.minimum_pos:
        #    reward += min(self.actor.compare_positions() - 1, 0)

        #reward for turning
        # if len(self.actor.comp_angles) > self.actor.minimum_pos:
        #     reward += self.actor.compare_angles() * 5

        #if len(self.actor.comp_distance) > self.actor.minimum_pos:
        #    reward += self.actor.compare_distances()

        #if reward > 0:
        #    reward *= reward_scale

        return torch.from_numpy(np.array([reward]))


    def get_state(self):
        #State includes tilemap and all relevant information from actor
        state = np.concatenate((self.map.get_local_map(self.actor.x, self.actor.y, 3).flatten(), np.array(self.actor.get_state())))
        return torch.from_numpy(state)


    def step(self):
        self.time+=1
        self.event_handler()
        self.process_elements()
        if self.display_sim:
            on_target, min_distance, direction = self.map.target_intersections(self.actor.x, self.actor.y)
            self.display.adjust_camera(self.actor.x, self.actor.y)
            self.display.render(self.entities, self.map.get_local_map(self.actor.x, self.actor.y, 3), \
                                [f'({round(self.actor.x, 3)}, {round(self.actor.y, 3)})',\
                                f'({round(self.actor.dx, 3)}, {round(self.actor.dy, 3)})',\
                                f'Time:  {self.time}',\
                                f'Reward:  {round(float(self.get_reward()), 3)}',\
                                f'Angle:  {self.actor.angle_deg}',\
                                f'Target Angle:  {direction}',\
                                f'Difference: {min(abs(self.actor.angle_deg - direction), abs(self.actor.angle_deg - direction - 360))}'])
        if self.display_sim:
            self.clock.tick(self.FPS)
        return self.get_state(), self.get_reward(), self.done


    def event_handler(self):
        for event in pygame.event.get() :
            if event.type == QUIT :
                pygame.quit()
                sys.exit()

        pressed = pygame.key.get_pressed()
        if not self.actor.brain:
            self.actor.apply_inputs(pressed[K_w], pressed[K_a], pressed[K_d])
        else:
            self.actor.use_brain(self.get_state())

        if pressed[K_r]:
            self.reset()

    def process_elements(self):
        self.collision_detection()
        for entity in self.entities:            
            entity.update()

    def collision_detection(self):
        #detect collisions between the actor and walls
        for wall in self.map.walls:
            collision, center, dx, dy = self.predict_actor_collision(wall)
            if collision:
                self.actor.x = center[0]
                self.actor.y = center[1]
                self.actor.dx = dx
                self.actor.dy = dy

        #detect if actor is trying to exit the map:
        collision, center, dx, dy = self.boundary_detection([self.actor.x, self.actor.y], \
                                                            self.actor.radius, self.actor.dx, self.actor.dy)
        if collision:
                self.actor.x = center[0]
                self.actor.y = center[1]
                self.actor.dx = dx
                self.actor.dy = dy

    def predict_actor_collision(self, wall):
        collision_x, center_x, dx, _ =  self.circle_rect_collision_x([self.actor.x, self.actor.y],\
                                                                 self.actor.radius,\
                                                                 wall.get_rect(), self.actor.dx, self.actor.dy)
        collision_y, center_y, _, dy =  self.circle_rect_collision_y([self.actor.x, self.actor.y],\
                                                                 self.actor.radius,\
                                                        wall.get_rect(), self.actor.dx, self.actor.dy)
        #collision_x, center_x, dx = False, [self.actor.x], self.actor.dx
        return collision_x or collision_y, [center_x[0], center_y[1]], dx, dy
        
    '''If there is a collision, returns the new position, dx, and dy for the circle'''
    def circle_rect_collision_x(self, circle_center, radius, rect, dx, dy):
        # Find the closest point to the circle within the rectangle
        closest_point = (max(rect.left, min(circle_center[0], rect.right)),\
                         max(rect.top, min(circle_center[1], rect.bottom)))
        

        # check if velocity is sufficient for collision
        if abs(closest_point[0] - circle_center[0]) > dx + radius * 2:
            return False, circle_center, dx, dy 
        
        # check if dy is sufficient for collision
        if (rect.top < circle_center[1] - radius and rect.bottom > circle_center[1] - radius) or (rect.top < circle_center[1] + radius and rect.bottom > circle_center[1] + radius):
            # check if direction could allow for collision (moving right)
            if circle_center[0] - radius < closest_point[0] and dx > 0:
                #check if collision will happen
                if circle_center[0] + radius + dx > closest_point[0]:
                    circle_center[0] = closest_point[0] - radius
                    return True, circle_center, 0, dy
                
            # check if direction could allow for collision (moving left)
            if circle_center[0] + radius > closest_point[0] and dx < 0:
                #check if collision will happen
                if circle_center[0] - radius + dx < closest_point[0]:
                    circle_center[0] = closest_point[0] + radius
                    return True, circle_center, 0, dy
        
        return False, circle_center, dx, dy
    
    def circle_rect_collision_y(self, circle_center, radius, rect, dx, dy):
        # Find the closest point to the circle within the rectangle
        closest_point = (max(rect.left, min(circle_center[0], rect.right)),\
                         max(rect.top, min(circle_center[1], rect.bottom)))
        

        # check if velocity is sufficient for collision
        if abs(closest_point[1] - circle_center[1]) > dy + radius * 2:
            return False, circle_center, dx, dy 
        
        # check if dx is sufficient for collision
        if (rect.left < circle_center[0] - radius and rect.right > circle_center[0] - radius) or (rect.left < circle_center[0] + radius and rect.right > circle_center[0] + radius):
            # check if direction could allow for collision (moving down)
            if circle_center[1] - radius < closest_point[1] and dy < 0:
                #check if collision will happen
                if circle_center[1] + radius - dy >= closest_point[1]:
                    circle_center[1] = closest_point[1] - radius
                    return True, circle_center, dx, 0
                
            # check if direction could allow for collision (moving up)
            if circle_center[1] + radius > closest_point[1] and dy > 0:
                #check if collision will happen
                if circle_center[1] - radius - dy <= closest_point[1]:
                    circle_center[1] = closest_point[1] + radius
                    return True, circle_center, dx, 0
        
        return False, circle_center, dx, dy
                
    def boundary_detection(self, circle_center, radius, dx, dy):
        left_boundary = 0
        right_boundary = 100 * len(self.map.tile_matrix[0])
        top_boundary = 0
        bottom_boundary = 100 * len(self.map.tile_matrix)
        collision = False
        #left
        if circle_center[0] - radius + dx < left_boundary:
            circle_center[0] =  radius + left_boundary
            dx = 0
            collision = True
        #right
        if circle_center[0] + radius + dx > right_boundary:
            circle_center[0] =  right_boundary - radius
            dx = 0
            collision = True
        #top
        if circle_center[1] - radius - dy < top_boundary:
            circle_center[1] =  top_boundary + radius
            dy = 0
            collision = True
        #bottom
        if circle_center[1] + radius - dy > bottom_boundary:
            circle_center[1] =  bottom_boundary - radius
            dy = 0
            collision = True

        return collision, circle_center, dx, dy

