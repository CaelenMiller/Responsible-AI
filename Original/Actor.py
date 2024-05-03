import pygame
import math
from collections import deque

'''What is trained. Contains a "brain" that determines its next action.
The brain should be a dynamic system like neural network.
x_k = current location + internal variables
x_k+1 = delta x is the change in location and some kind of internal stuff?
u_k = current situation
y_k = current location'''
class Actor():
    def __init__(self, brain=None, cords = [150,150], radius=20, color = (255,25, 70), top_speed=10.0, rot_velocity = 3, acceleration=1):

        self.brain = brain

        self.x = cords[0]
        self.y = cords[1]
        self.radius = radius
        self.color = color
        self.acceleration = acceleration
        self.velocity = 0
        self.top_speed = top_speed
        self.angle_deg = 270
        self.rot_velocity = rot_velocity
        self.dx, self.dy = 0, 0

        self.last_action = [False,False,False] #for info purposes
        self.last_x, self.last_y = self.x, self.y
        self.time_still = 0
        self.minimum_pos = 15
        self.comp_positions = deque(maxlen=30) #records last 30 positions
        self.comp_angles= deque(maxlen=30) #records last 30 angles
        self.comp_distance = deque(maxlen=30) #records last 30 distances
        self.smoothed_distances = deque(maxlen=3) # last 2 smoothed distance


    def apply_inputs(self, forward, Lrotation, Rrotation): # forward is between 0 and 1, rotation is between -1 (right) and 1 (left)
        clipped_forward = max(0, min(1, forward))
        clipped_Lrotation = max(0, min(1, Lrotation))
        clipped_Rrotation = max(0, min(1, Rrotation))
        self.velocity = min(self.velocity + self.acceleration * clipped_forward, self.top_speed)
        self.angle_deg += self.rot_velocity * clipped_Lrotation
        self.angle_deg -= self.rot_velocity * clipped_Rrotation

        #print(f'{round(clipped_forward,2)}, {round(clipped_Lrotation,2)}, {round(clipped_Rrotation,2)}')

        self.angle_deg %= 360

        self.dx = self.velocity * math.cos(math.radians(self.angle_deg))
        self.dy = self.velocity * math.sin(math.radians(self.angle_deg))

        self.last_action = [float(clipped_forward), float(clipped_Lrotation), float(clipped_Rrotation)]


    def apply_friction(self):
        if self.velocity > 0:
            self.velocity = max(self.velocity - self.acceleration/3, 0)
            
        
    def update(self):
        self.x += self.dx
        self.y -= self.dy
        self.apply_friction()
        self.comp_positions.append((self.x, self.y))
        self.comp_angles.append(self.angle_deg)
        if self.last_x == self.x and self.last_y == self.y:
            self.time_still += 1
        else:
            self.time_still = 0

        self.last_x = self.x
        self.last_y = self.y

    def get_state(self): #x, y, x in tile, y in tile, velocity, angle, dx, dy
        return [self.x/100, self.y/100, self.velocity, self.angle_deg/360, self.dx, self.dy]


    def use_brain(self, inputs, explore=False): #inputs can be a tuple of controls or the full input for the brain
        if self.brain != None:
            if not explore:
                outputs = self.brain.activate(inputs).reshape(3,-1)    
            else:
                outputs = self.brain.explore().reshape(3,-1)  
        else:
            outputs = inputs # if it has not brain, nothing changes
        return outputs


    def render(self, window, camera=None):
        if camera:
            adj_x, adj_y = camera.apply(self.x, self.y)
        else:
            adj_x = self.x
            adj_y = self.y
        pygame.draw.circle(window, self.color, (adj_x, adj_y), self.radius, width=0)

        # Draw the arrow
        angle_in_radians = math.radians(self.angle_deg)
        end_x = adj_x + math.cos(angle_in_radians) * self.radius
        end_y = adj_y - math.sin(angle_in_radians) * self.radius
        pygame.draw.line(window, (0,0,0), (adj_x, adj_y), (end_x, end_y), 3)

    def learn(self, memory, iterations): #TODO - consider if gamma should be passed into here
        return self.brain.learn(memory, iterations)

    def compare_positions(self): #distance between current position and oldest stored position
        return math.sqrt((self.x - self.comp_positions[0][0])**2 + (self.y - self.comp_positions[0][1])**2)
    
    def compare_angles(self): #distance between current and oldest stored angle
        return min(abs(self.angle_deg - self.comp_angles[0]), 360 - abs(self.angle_deg - self.comp_angles[0])) / 360
    
    def compare_distances(self): #distance between current position and oldest stored distance
        return self.comp_distance[0] - self.comp_distance[-1]
    
    def compare_smooth_distances(self): #distance between current position and oldest stored distance
        return self.smoothed_distances[0] - self.smoothed_distances[-1]
        
