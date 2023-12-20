import numpy as np
import pygame
import math
from collections import deque
import random


class Map:
    def __init__(
        self,
        size=[700, 700],
        tile_matrix=np.array(
            [
                [0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                [1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
                [1, 0, 0, 2, 0, 1, 0, 1, 0, 0, 0, 1, 0],
                [1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0],
            ]
        ),
    ):
        self.size = size
        self.tile_matrix = tile_matrix
        self.walls = []
        self.targets = []
        self.spawns = []
        self.generate_map(tile_matrix)
        self.distance_map = self.generate_distance_map()
        self.max_dist = self.distance_map.flatten()[
            np.isfinite(self.distance_map.flatten())
        ].max()

    def generate_map(self, tile_matrix):
        self.walls = []
        self.tile_size_x = 100
        self.tile_size_y = 100

        for y, row in enumerate(tile_matrix):
            for x, col in enumerate(row):
                x_adj = x * self.tile_size_x
                y_adj = y * self.tile_size_y
                if col == 1:
                    self.walls.append(
                        Env_Object(x_adj, y_adj, self.tile_size_x, self.tile_size_y)
                    )
                if col == 2:
                    self.targets.append(
                        Env_Object(
                            x_adj,
                            y_adj,
                            self.tile_size_x,
                            self.tile_size_y,
                            obj_type="target",
                        )
                    )
                if col == 3:
                    self.spawns.append(
                        Env_Object(
                            x_adj,
                            y_adj,
                            self.tile_size_x,
                            self.tile_size_y,
                            obj_type="spawn",
                        )
                    )

    def target_intersections(self, x, y):
        min_distance = 1000
        for target in self.targets:
            distance_x = abs(x - (target.x + target.dim[0] // 2))
            distance_y = abs(y - (target.y + target.dim[1] // 2))  # manhattan
            distance = distance_x + distance_y
            delta_x = x - target.x
            delta_y = y - target.y
            angle_rad = math.atan2(delta_x, delta_y)  # Angle in radians
            angle_deg = math.degrees(angle_rad) + 90  # Convert to degrees
            if distance_x < target.dim[0] // 2 and distance_y < target.dim[1] // 2:
                return True, distance, angle_deg
            else:
                if distance < min_distance:
                    min_distance = distance

        return False, min_distance, angle_deg

    def get_space_size(self):
        return len(self.tile_matrix) * len(self.tile_matrix[0])

    def get_local_map(self, x, y, r):  # x and y are true coords
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

    def random_valid_location(self):
        # Find the indices of zeros in the 2D array
        zero_indices = np.argwhere(self.tile_matrix == 0)

        # Randomly select one of the zero indices
        random_index = random.choice(zero_indices)
        return (random_index[1] + 1 / 2) * self.tile_size_x, (
            random_index[0] + 1 / 2
        ) * self.tile_size_y

    def generate_distance_map(self):
        rows, cols = self.tile_matrix.shape
        distance_map = np.full(
            (rows, cols), np.inf
        )  # Initialize with infinite distance

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
                    if (
                        self.tile_matrix[new_y][new_x] != 1
                        and distance_map[new_y][new_x] == np.inf
                    ):
                        distance_map[new_y][new_x] = distance_map[y][x] + 1
                        queue.append((new_x, new_y))

        return distance_map

    def get_smoothed_distance(self, x, y):  # x and y are in continous coord
        dis_x, dis_y = y // 100, x // 100
        cur_distance = self.distance_map[dis_y][dis_x]
        smoothed_distance = cur_distance * 100

        # figure out which direction is correct
        above = (max(dis_y - 1, 0), dis_x)
        below = (min(dis_y + 1, len(self.tile_matrix) - 1), dis_x)
        left = (dis_y, max(dis_x - 1, 0))
        right = (dis_y, min(dis_x + 1, len(self.tile_matrix[0]) - 1))

        directions = {
            "below": cur_distance < self.distance_map[above[0]][above[1]],
            "above": cur_distance < self.distance_map[below[0]][below[1]],
            "left": cur_distance < self.distance_map[left[0]][left[1]],
            "right": cur_distance < self.distance_map[right[0]][right[1]],
        }

        y_offset = y - (dis_y * 100 + 50)
        x_offset = x - (dis_x * 100 + 50)

        if y_offset > 0:
            if directions["below"]:
                smoothed_distance -= y_offset
            else:
                smoothed_distance += y_offset

        if y_offset < 0:
            if directions["above"]:
                smoothed_distance += y_offset
            else:
                smoothed_distance -= y_offset

        if x_offset > 0:
            if directions["right"]:
                smoothed_distance -= x_offset
            else:
                smoothed_distance += x_offset

        if x_offset < 0:
            if directions["left"]:
                smoothed_distance += x_offset
            else:
                smoothed_distance -= x_offset

        return smoothed_distance, directions


"""These operate as rectangular zones, including non passable walls, spawn locations, and target locations"""


class Env_Object:
    def __init__(self, minx, miny, dimx, dimy, obj_type="wall"):
        self.rect = pygame.Rect(minx, miny, dimx, dimy)
        self.obj_type = obj_type
        self.x = minx
        self.y = miny
        self.dim = [dimx, dimy]

        if self.obj_type == "wall":
            self.color = (0, 205, 255)
        else:
            self.color = (200, 15, 15)

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
