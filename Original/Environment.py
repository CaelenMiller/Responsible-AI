import numpy as np
import math
import time
import pygame
import torch
import sys
import gym
from pygame.locals import *
from DisplayTools import *
from Map import *
from Actor import *


class Environment(gym.Env):
    def __init__(self, display_sim=True, brain=None, state_type = "sensor", 
                 max_time = 1024, max_stall_time = 60, tile_matrix=np.array([])):
        pygame.init()
        print("Initializing Environment")
        if tile_matrix.any():
            self.map = Map(tile_matrix=tile_matrix)
        else:
            self.map = Map()
        self.actor = Actor(brain=brain) #initial setup and brain implant
        self.reset()
        self.time = 0
        self.display_sim = display_sim
        self.display = Display(self.map)
        self.pause_menu = Pause_Menu(screen_height=self.display.HEIGHT, screen_width=self.display.WIDTH)
        self.pause_instructions = "none"
        self.state_type = state_type

        self.clock = pygame.time.Clock()
        self.FPS = 60
        self.max_time = max_time
        self.max_stall_time = max_stall_time

        # Define action and observation space
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
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
        self.actor.angle_deg = random.randint(0,360)
        if rand_start:
            self.actor.x, self.actor.y = self.map.random_valid_location()
        self.entities = [wall for wall in self.map.walls]
        self.entities.append(self.actor)
        for target in self.map.targets:
            self.entities.append(target)


    def get_reward(self): 
        on_target, _, _ = self.map.target_intersections(self.actor.x, self.actor.y)
        self.actor.comp_distance.append(self.map.distance_map[int(self.actor.y//100)][int(self.actor.x//100)])
        #reward_scale = (2 - self.time/self.max_time) / 2 # 1.0 to 0.5, encourages getting to goal faster
        
        if on_target: #Reward if on target
            self.done = True
            reward = 10

        else: #Reward based on tile_distance and facing the right direction
            smoothed_dist, right_direction = self.map.get_smoothed_distance(self.actor.y,self.actor.x, self.actor.angle_deg)
            self.actor.smoothed_distances.append(smoothed_dist)
            reward = self.actor.compare_smooth_distances() * 5
            #reward += right_direction / 10 #rewards moving in the right direction
            # if right_direction == -1:
            #     self.time = self.max_time #end early if going the wrong way
            
        reward -= 0.3 * self.actor.time_still/(self.max_stall_time) #between -0.5 and 0

        if self.actor.time_still > self.max_stall_time: #Ends the run if stuck for a time
            self.time = self.max_time
            #reward -= min((self.actor.time_still/200) ** 2 , 5)

        return torch.from_numpy(np.array([reward]))


    def get_state(self):
        print(self.lidar_detection(self.actor.angle_deg, 5, 180))

        #State includes tilemap and all relevant information from actor
        if self.state_type == "sensor":
            _, right_direction = self.map.get_smoothed_distance(int(self.actor.y),int(self.actor.x), self.actor.angle_deg)
            actor_state = np.array(self.actor.get_state())
            actor_state = np.append(actor_state, right_direction)
            env_state = self.map.get_local_map(self.actor.x, self.actor.y, 3).flatten()
            state = np.concatenate((env_state, actor_state))
            
        elif self.state_type == "visual":
            if self.display_sim:
                state = self.display.get_screen([28,28]) 
            else:
                raise TypeError(f'Cannot get visual data: Invalid state type: {self.state_type}')
        else:
            raise TypeError(f'Invalid state type: {self.state_type}')
        
        return torch.from_numpy(state)


    def step(self):
        self.time+=1

        self.event_handler()

        self.process_elements()

        state, reward = self.get_state(), self.get_reward()

        self.render()

        if self.display_sim:
            self.clock.tick(self.FPS)
        return state, reward, self.done

    def render(self):
        if self.display_sim:
            state, reward = self.get_state(), self.get_reward()
            #on_target, min_distance, direction = self.map.target_intersections(self.actor.x, self.actor.y)
            self.display.adjust_camera(self.actor.x, self.actor.y)
            if self.state_type == "sensor":
                self.display.render(self.entities, self.map.get_local_map(self.actor.x, self.actor.y, 3), \
                                    [f'({round(self.actor.x, 3)}, {round(self.actor.y, 3)})',\
                                    f'({round(self.actor.dx, 3)}, {round(self.actor.dy, 3)})',\
                                    f'Time:  {self.time}',\
                                    f'Time Still:  {self.actor.time_still}',\
                                    f'Reward:  {round(float(reward), 3)}',\
                                    f'Distance:  {self.actor.smoothed_distances[-1]}',
                                    f'Angle: {round(self.actor.angle_deg, 3)}'])
            elif self.state_type == "visual":
                self.display.render(self.entities)

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

        if pressed[K_ESCAPE] or pressed[K_p]:
            self.pause_instructions = self.pause_menu.activate(self.display, self.entities, self.display_sim)            

        if pressed[K_r]:
            self.reset()


    def get_pause_instruction(self):
        return self.pause_instructions
    
    def reset_instructions(self):
        self.pause_instructions = "none"

    def process_elements(self):
        self.collision_detection()
        for entity in self.entities:            
            entity.update()


    '''Base logic for the environment, various forms of collision and boundary detection'''
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

    def lidar_detection(self, center_angle, rays=15, angle_spread=90):
        x,y = self.actor.x, self.actor.y
        angles = [int((center_angle + angle_spread * ray / (rays - 1) - angle_spread/2 + 360) % 360) for ray in range(rays)]
        radian_angles = [math.radians(angle) for angle in angles]
        
        def cast_ray(self, angle, x, y):
            x_ray, y_ray = x,y
            dist = 0
            while x_ray//100 > 0 and y_ray//100 > 0 and x_ray//100 < len(self.map.tile_matrix) and y_ray//100 < len(self.map.tile_matrix):
                target_x = (math.ceil(x_ray / 100) * 100) if math.cos(angle) > 0 else (math.floor(x_ray / 100) * 100)
                target_y = (math.ceil(y_ray / 100) * 100) if math.sin(angle) > 0 else (math.floor(y_ray / 100) * 100)
                
                dx_to_100 = target_x - x_ray
                dy_to_100 = target_y - y_ray
                
                cos_theta = math.cos(angle)
                sin_theta = math.sin(angle)
                
                # Calculate magnitudes, avoiding division by zero
                d_x = dx_to_100 / cos_theta if cos_theta != 0 else float('inf')
                d_y = dy_to_100 / sin_theta if sin_theta != 0 else float('inf')
                
                # Choose the smallest positive magnitude
                mag = min(d_x, d_y) if min(d_x, d_y) > 0 else max(d_x, d_y)

                dist += mag

                if self.map.tile_matrix[int(x_ray//100)][int(y_ray//100)] != 0:
                    return self.map.tile_matrix[int(x_ray//100)][int(y_ray//100)], dist
                else:
                    x_ray += math.cos(angle) * mag
                    y_ray += math.sin(angle) * mag

            return 1, dist

        lidar_measurements = []
        for angle in radian_angles: #pass in randian angles to work properly with sin/cos calcs
            lidar_measurements.append(cast_ray(self, angle, x, y))

        return lidar_measurements



class Pause_Menu():
    def __init__(self, screen_width=700, screen_height=700):
        self.size = [200,200]
        self.color = (210, 180, 140)
        self.rect = pygame.Rect((screen_width - self.size[0]) // 2, (screen_height - self.size[1]) // 2, self.size[0], self.size[1])

    def activate(self, display, entities, display_sim = True):
        running = True
        if not display_sim:
            print("Menu is open. Press W to exit, \nD to display, \nX to end training, \nS to save model, \nA to toggle random starts.")
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

            pressed = pygame.key.get_pressed()
            
            if pressed[K_w]:
                return "none"
            
            if pressed[K_d]: #Force the next iteration to display
                return "force_display"

            if pressed[K_s]: #Save the current model
                return "force_save"

            if pressed[K_a]:
                return "toggle_random_start"
            
            if pressed[K_x]:
                return "end_training"
            
            if pressed[K_e]:
                return "display_loss"

            if display_sim:
                display.render(entities, pause_menu=self)
