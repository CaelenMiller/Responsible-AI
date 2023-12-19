from pygame.locals import *
import pygame


class Display:
    def __init__(self, map):
        self.WIDTH = map.size[0]
        self.HEIGHT = map.size[1]
        self.WINDOW = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.BACKGROUND = (0, 90, 50)
        self.WALLS = (0,205,255)
        self.camera = Camera()

        self.walkable = Env_Object(0, 0, 100 * len(map.tile_matrix[0]), 100 * len(map.tile_matrix), obj_type="background")


    def render(self, entities, mini_map = [], info=[]):
            self.WINDOW.fill(self.WALLS)
            self.walkable.render(self.WINDOW, self.camera)
            for entity in entities:
                 entity.render(self.WINDOW, self.camera)
            if len(mini_map) > 0:
                self.draw_mini_map(mini_map)
            self.draw_text(info, self.WIDTH-20, 20)
            pygame.display.update()

    def draw_text(self, text, x, y, font_size=20, color=(255, 255, 255)):
        # Initialize a font
        my_font = pygame.font.SysFont(None, font_size)

        for i, line in enumerate(text):
            # Render the text
            text_surface = my_font.render(line, True, color)
            
            # Get text rectangle
            text_rect = text_surface.get_rect()
            
            # Positioning it at top-right
            text_rect.topright = (x, y+20*i)
        
            # Draw the text on screen
            self.WINDOW.blit(text_surface, text_rect)


    def draw_mini_map(self, mini_map):
        # Define your colors
        colors = {
            -1: (0, 0, 0),
            0: (0, 90, 50),
            1: (0, 205, 255),
            2: (200, 15, 15),
            3: (100, 0, 100)
        }

        # Number of rows/cols in the slice
        n = mini_map.shape[0]

        # Calculate cell dimensions
        cell_width = 50 // n
        cell_height = 50 // n

        # Coordinates where the mini display starts (bottom-right corner of screen)
        start_x = self.WIDTH - 50 - 2  # minus 2 for border
        start_y = self.HEIGHT - 50 - 2  # minus 2 for border

        # Draw the border
        pygame.draw.rect(self.WINDOW, (255, 255, 255), (start_x - 2, start_y - 2, 54, 54), 2)

        # Render the grid
        for i in range(n):
            for j in range(n):
                # Get the array value
                val = mini_map[i, j]
                
                # Lookup the color
                color = colors.get(val, (255, 255, 255))  # Default to white if value not found
                
                # Calculate the top-left corner coordinates for this cell
                x = start_x + j * cell_width
                y = start_y + i * cell_height

                pygame.draw.rect(self.WINDOW, color, (x, y, cell_width, cell_height))

    def adjust_camera(self, x, y):
        self.camera.adjust(x - self.WIDTH//2, y - self.HEIGHT//2)


class Camera():
    def __init__(self):
        self.xdif = 0
        self.ydif = 0

    def apply(self, x,y):
        return x + self.xdif, y + self.ydif

    def adjust(self, x, y):
        self.xdif = -x
        self.ydif = -y


class Env_Object:
    def __init__(self, minx, miny, dimx, dimy, obj_type="wall"):
        self.rect = pygame.Rect(minx, miny, dimx, dimy)
        self.obj_type = obj_type
        self.x = minx
        self.y = miny
        self.dim = [dimx, dimy]

        if self.obj_type=="wall":
            self.color=(0,205,255)
        if self.obj_type=="background":
            self.color = (0, 90, 50)
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

