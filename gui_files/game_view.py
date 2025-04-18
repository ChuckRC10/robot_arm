import pygame
import jax.numpy as jnp
from config import arm1color, arm2color, screenSize

def game_setup():
    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode(screenSize)
    return clock, screen

def get_origin():
    origin = [screenSize[0]/2, screenSize[1]/2]
    return origin
        
def paint_rect(screen, wantedPosition):
    origin = get_origin()
    rectangleXPosition = float(wantedPosition[0] + origin[0])
    rectangleYPosition = float(wantedPosition[1] + origin[1])
    pointerRectangle = pygame.Rect(rectangleXPosition, rectangleYPosition, 5, 5)
    pygame.draw.rect(screen, arm1color, pointerRectangle) 

def paint_arm(screen, armVectors: jnp.array):
    origin = jnp.array(get_origin())

    for armNum in range(len(armVectors)):
        if armNum == 0:
            startingPosition = origin
        else:
          startingPosition = origin + jnp.sum(armVectors[0:armNum], 0)
        endingPosition = origin + jnp.sum(armVectors[0:armNum + 1], 0)
        pygame.draw.line(screen, arm2color, startingPosition.tolist(), endingPosition.tolist(), 5)