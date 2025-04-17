import pygame
import numpy as np
from config import line_color1, line_color2, screen_size

def game_setup():
    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode(screen_size)
    return clock, screen

def get_origin():
    origin = [screen_size[0]/2, screen_size[1]/2]
    return origin

def did_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False  
        else:
            return True
        
def paint_rect(screen, wntd_pos: list):
    origin = get_origin()
    rect1 = pygame.Rect(wntd_pos[0] + origin[0], wntd_pos[1] + origin[1], 5, 5)
    pygame.draw.rect(screen, line_color1, rect1) 

def paint_arm(screen, vec1: list, vec2: list):
    origin = np.array(get_origin())
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    pygame.draw.line(screen, line_color1, origin, vec1 + origin, 5)
    pygame.draw.line(screen, line_color2, vec1 + origin, vec1 + vec2 + origin, 5)