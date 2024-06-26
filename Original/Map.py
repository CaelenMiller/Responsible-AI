import numpy as np
import pygame
import math
from collections import deque
import random

class Map:
    def __init__(self, size = [784,784], tile_matrix=np.array([[1,1,1,1,1,1,1,1,1,1,1,1,1],
                                                                [1,0,1,1,1,1,0,1,1,1,1,1,1],
                                                                [1,0,0,0,0,0,0,0,0,1,1,1,1],
                                                                [1,1,0,1,1,1,1,1,0,1,1,1,1],
                                                                [1,1,0,1,0,1,0,0,0,1,1,1,1],
                                                                [1,1,0,0,0,1,0,1,1,1,1,1,1],
                                                                [1,1,1,1,1,1,0,0,0,1,1,1,1],
                                                                [1,1,0,0,0,1,1,1,0,1,1,1,1],
                                                                [1,2,0,1,0,0,0,0,0,1,1,1,1],
                                                                [1,1,1,1,1,1,1,1,1,1,1,1,1]])):
        self.size=size
        self.tile_matrix=tile_matrix
        self.tile_size_x = 100
        self.tile_size_y = 100
        self.walls = []
        self.targets = []
        self.spawns = []
        self.generate_map(tile_matrix)
        self.distance_map = self.generate_distance_map()
        self.max_dist = self.distance_map.flatten()[np.isfinite(self.distance_map.flatten())].max()

        #self.show_distance_heatmap()

    #Uses a matrix to generate a map 
    def generate_map(self, tile_matrix):
        self.walls = []

        for y, row in enumerate(tile_matrix):
            for x, col in enumerate(row):
                x_adj = x * self.tile_size_x
                y_adj = y * self.tile_size_y
                if col == 1:
                    self.walls.append(Env_Object(x_adj, y_adj, self.tile_size_x, self.tile_size_y))
                if col == 2:
                    self.targets.append(Env_Object(x_adj, y_adj, self.tile_size_x, self.tile_size_y, obj_type="target"))
                if col == 3:
                    self.spawns.append(Env_Object(x_adj, y_adj, self.tile_size_x, self.tile_size_y, obj_type="spawn"))

    def target_intersections(self, x, y):
        min_distance = 1000
        for target in self.targets:
            distance_x = abs(x - (target.x + target.dim[0]//2))
            distance_y = abs(y - (target.y + target.dim[1]//2)) #manhattan
            distance = distance_x+distance_y
            delta_x = x - target.x
            delta_y = y - target.y
            angle_rad = math.atan2(delta_x, delta_y)  # Angle in radians
            angle_deg = math.degrees(angle_rad) + 90  # Convert to degrees
            if distance_x < target.dim[0]//2 and distance_y < target.dim[1]//2:
                return True, distance, angle_deg
            else:
                if distance < min_distance:
                    min_distance = distance
            
        return False, min_distance, angle_deg

    def get_space_size(self):
        return len(self.tile_matrix) * len(self.tile_matrix[0])
    
    #Fetches a minimap centered on the actor. If r=2, minimap is a 5x5 tilemap
    def get_local_map(self, x, y, r): #x and y are true coords
        map_x, map_y = x // self.tile_size_x, y // self.tile_size_y
        cols, rows = self.tile_matrix.shape
        
        # Initialize an empty array filled with 1
        vision = np.full((2 * r + 1, 2 * r + 1), 1)
        
        for i in range(-r, r + 1):
            for j in range(-r, r + 1):
                xi, yi = map_x + i, map_y + j
                
                if 0 <= xi and xi < rows and 0 <= yi and yi < cols:
                    vision[j + r][i + r] = self.tile_matrix[int(yi)][int(xi)]
                    
        return vision
    
    #Selects a random walkable location from the full map
    def random_valid_location(self):
        # Find the indices of zeros in the 2D array
        zero_indices = np.argwhere(self.tile_matrix == 0)
        
        # Randomly select one of the zero indices
        random_index = random.choice(zero_indices)
        return (random_index[1]+1/2) * self.tile_size_x, (random_index[0]+1/2) * self.tile_size_y
    
    def check_location_validity(self, x, y): #x and y are true coords
        if self.tile_matrix[x//self.tile_size_x][y//self.tile_size_y] == 0:
            return True
        else:
            return False
    
    #Performs BFS to generate map of integer distances from goal. AKA, number of tiles that must be crossed from
    #each tile in order to reach the closest goal. 
    def generate_distance_map(self):
        rows, cols = self.tile_matrix.shape
        distance_map = np.full((rows, cols), np.inf)  # Initialize with infinite distance

        # Initialize the queue for BFS
        queue = deque()

        # Initialize the distance for each target cell and add to queue
        for y, row in enumerate(self.tile_matrix):
            for x, val in enumerate(row):
                if val == 2:
                    queue.append((x, y))
                    distance_map[y][x] = 0

        # Breadth-First Search
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up

        while queue:
            x, y = queue.popleft()

            for dx, dy in directions:
                new_x, new_y = x + dx, y + dy

                if 0 <= new_x < cols and 0 <= new_y < rows:
                    if self.tile_matrix[new_y][new_x] != 1 and distance_map[new_y][new_x] == np.inf:
                        distance_map[new_y][new_x] = distance_map[y][x] + 1
                        queue.append((new_x, new_y))

        return distance_map
    
    #Gets a real valued distance from goal, using the distance map. If degree is input, determines if it is facing the correct direction
    def get_smoothed_distance(self, y, x, angle_deg = None): #x and y are in continous coord, get converted to int cords
        tile_x, tile_y = int(x//100), int(y//100)
        cur_distance = self.distance_map[tile_y][tile_x]
        smoothed_distance = cur_distance

        #Get the tile coords of each of the directions. Force them to be in bounds
        dir = {"up" : (max(tile_y-1, 0), tile_x),
                    "down" : (min(tile_y+1, len(self.tile_matrix)-1), tile_x),
                    "left" : (tile_y, max(tile_x-1, 0)),
                    "right" : (tile_y, min(tile_x+1, len(self.tile_matrix[0])-1))}

        #Dictionary. Key is which direction, -1 indicates wrong way, 0 if wall, 1 if right way
        direction_values = {"up" : cur_distance - self.distance_map[dir["up"][0]][dir["up"][1]] if self.tile_matrix[dir["up"][0]][dir["up"][1]] != 1 else 0,
                      "down" : cur_distance - self.distance_map[dir["down"][0]][dir["down"][1]] if self.tile_matrix[dir["down"][0]][dir["down"][1]] != 1 else 0,
                      "left"  : cur_distance - self.distance_map[dir["left"][0]][dir["left"][1]] if self.tile_matrix[dir["left"][0]][dir["left"][1]] != 1 else 0,
                      "right" : cur_distance - self.distance_map[dir["right"][0]][dir["right"][1]] if self.tile_matrix[dir["right"][0]][dir["right"][1]] != 1 else 0}
        

        #subtract the center of current tile from the actual position. 
        y_offset = y/100 - (tile_y + 0.5)#negative y is above center, negative x is left of center
        x_offset = x/100 - (tile_x + 0.5)#0.5 gets us to the center of the current tile

        current_side_y = "up" if y_offset < 0 else "down"
        current_side_x = "left" if x_offset < 0 else "right"

        #case: we're on the right side
        if direction_values[current_side_y] == 1: #we're on the correct vertical side
            smoothed_distance -= abs(y_offset)
        
        #case: we're on the wrong side
        if direction_values[current_side_y] == -1:
            #if not (direction_values["left"] == 0 and direction_values["right"] == 0):
                smoothed_distance += abs(y_offset)

        #case: we're on the right side
        if direction_values[current_side_x] == 1: #we're on the correct horizontal side
            smoothed_distance -= abs(x_offset)

        #case: we're on the wrong side
        if direction_values[current_side_x] == -1:
            #case: vertical position in tile does not matter
            #if (direction_values["up"] == 0 and direction_values["down"] == 0):
                smoothed_distance += abs(x_offset)

        # if direction_values[current_side_y] == 1 and direction_values[current_side_x] == 1:
        #     smoothed_distance += abs(x_offset) + abs(y_offset) - math.sqrt(abs(x_offset ** 2 + y_offset ** 2))
        if angle_deg is not None:
            right_direction = 0
            if angle_deg >= 45 and angle_deg < 135:
                right_direction = direction_values["up"]
            if angle_deg >= 135 and angle_deg < 225:
                right_direction = direction_values["left"]
            if angle_deg >= 225 and angle_deg < 315:
                right_direction = direction_values["down"]
            if angle_deg >= 315 or angle_deg < 45:
                right_direction = direction_values["right"]


            return smoothed_distance, right_direction
        #print(f'{cur_distance} : {smoothed_distance} - up: {direction_values["up"]}, down: {direction_values["down"]}, left: {direction_values["left"]}, right: {direction_values["right"]}')

        return smoothed_distance
    
    #TESTING THE MAP SMOOTHING - Displays a heatmap 
    def show_distance_heatmap(self):
        import matplotlib.pyplot as plt
        matrix = [[self.get_smoothed_distance(y,x) if self.tile_matrix[y//100][x//100] == 0 else -0.5 for x in range(999)] for y in range(999)]

        plt.imshow(matrix, cmap='hot', interpolation='nearest')

        plt.colorbar()

        plt.show()

    def generate_random_tilemap(self, height, width):
        # Initialize maze with walls (1's)
        tilemap = np.ones((height, width), dtype=np.int32)

        # Function to carve paths using DFS
        def carve_paths(x, y):
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            random.shuffle(directions)

            for dx, dy in directions:
                nx, ny = x + dx*2, y + dy*2
                if 1 <= nx < width-1 and 1 <= ny < height-1 and tilemap[ny, nx] == 1:
                    tilemap[ny, nx] = 0
                    tilemap[ny-dy, nx-dx] = 0
                    carve_paths(nx, ny)

        # Start carving paths from a random cell
        start_x, start_y = random.randrange(1, width-1, 2), random.randrange(1, height-1, 2)
        tilemap[start_y, start_x] = 0
        carve_paths(start_x, start_y)

        # Place the special cell (2)
        while True:
            x, y = random.randint(1, width-2), random.randint(1, height-2)
            if tilemap[y, x] == 0:
                tilemap[y, x] = 2
                break

        return tilemap

'''These operate as rectangular zones, including non passable walls, spawn locations, and target locations'''
class Env_Object:
    def __init__(self, minx, miny, dimx, dimy, obj_type="wall"):
        self.rect = pygame.Rect(minx, miny, dimx, dimy)
        self.obj_type = obj_type
        self.x = minx
        self.y = miny
        self.dim = [dimx, dimy]

        if self.obj_type=="wall":
            self.color=(0,205,255)
        else:
            self.color=(200, 15, 15)

    def get_rect(self):
        return self.rect
    
    def render(self, window, camera=None):
        if camera:
            left, top = camera.apply(self.rect.left, self.rect.top)
            adj_rect = pygame.Rect(left, top, self.rect.width, self.rect.height)
        else:
            adj_rect = self.rect
        
        pygame.draw.rect(window, self.color, adj_rect)

    def update(self):
        pass
    
    def __str__(self) -> str:
        return f"({self.obj_type}: {self.rect})"
    