import pygame
import jax.numpy as jnp
from config import armColor, screenSize

# TODO:
def game_setup():
    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode(screenSize)
    return clock, screen

def textSetup():
    font = pygame.font.Font('freesansbold.ttf', 16)
    areaString = f"Area: inital whatevers"
    text = font.render(areaString, True, (0,0,0), (255,255,255))
    textRect = text.get_rect()
    locationRatio = .7
    textRect.center = (screenSize[0] * locationRatio, screenSize[1] * (locationRatio + .2) )
    return font, textRect

def get_origin():
    origin = jnp.array([screenSize[0]/2, screenSize[1]/2])
    return origin
        
def paint_rect(screen, wantedPosition):
    origin = get_origin()
    rectanglePosition = wantedPosition + origin
    pointerRectangle = pygame.Rect(float(rectanglePosition[0]), float(rectanglePosition[1]), 5, 5)
    pygame.draw.rect(screen, armColor, pointerRectangle) 

def paint_arm(screen, armVectors: jnp.array):
    origin = jnp.array(get_origin())

    for armNum in range(len(armVectors)):
        if armNum == 0:
            startingPosition = origin
        else:
          startingPosition = origin + jnp.sum(armVectors[0:armNum], 0)
        endingPosition = origin + jnp.sum(armVectors[0:armNum + 1], 0)
        pygame.draw.line(screen, armColor, startingPosition.tolist(), endingPosition.tolist(), 5)

def createEllipseRectangle(centerPosition, majorLength, minorLength) -> pygame.Rect:
    origin = get_origin()
    leftPosition = origin[0] + centerPosition[0] - majorLength
    topPosition = origin[1] + centerPosition[1] - minorLength
    rectangleWidth = 2 * majorLength
    rectangleHeight = 2 * minorLength

    referenceRectangle = pygame.Rect(float(leftPosition), float(topPosition),
                                     float(rectangleWidth), float(rectangleHeight))
    return referenceRectangle

def paintEllipseAngle(surface, color, rect, angle, width = 2):
    ellipseSurface = pygame.Surface(rect.size, pygame.SRCALPHA)
    pygame.draw.ellipse(ellipseSurface, color, ellipseSurface.get_rect(), width)

    rotatedSurface = pygame.transform.rotate(ellipseSurface, -jnp.degrees(angle))
    surface.blit(rotatedSurface, rotatedSurface.get_rect(center = rect.center))

def paintText(surface, area, font, textRect):
    areaString = f"Area: {area:7.0f} whatevers"
    text = font.render(areaString, True, (0,0,0), (255,255,255))
    surface.blit(text, textRect)
    return