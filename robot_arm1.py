# import and initialize pygame library
import pygame
import numpy as np
pygame.init()

# Import pygame.locals for easier access to key coordinates
# Updated to conform to flake8 and black standards
from pygame.locals import (
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    K_ESCAPE,
    KEYDOWN,
    QUIT,
)

# Setup the clock for a decent framerate
clock = pygame.time.Clock()

# Move the sprite based on user keypresses
def update_wanted_q(pressed_keys, wntd_x_pos, wntd_y_pos, max_pos_delt):
    # get user input
    if pressed_keys[K_UP]:
        wntd_y_pos -= max_pos_delt
    if pressed_keys[K_DOWN]:
        wntd_y_pos += max_pos_delt
    if pressed_keys[K_LEFT]:
        wntd_x_pos -= max_pos_delt
    if pressed_keys[K_RIGHT]:
        wntd_x_pos += max_pos_delt

    return wntd_x_pos, wntd_y_pos

def update(pressed_keys, len1, len2, alpha, beta, wntd_x_pos, wntd_y_pos):
    
    # caculate error
    # get actual end effector position
    x_pos = len1*np.cos(alpha) + len2*np.cos(alpha + beta)
    y_pos = len1*np.sin(alpha) + len2* np.sin(alpha + beta)
    e = np.array([wntd_x_pos - x_pos, wntd_y_pos - y_pos])

    # get delta alpha and delta beta 
    J = np.array([[-len1*np.sin(alpha)-len2*np.sin(alpha+beta), -len2*np.sin(alpha+beta)],
                  [len1*np.cos(alpha) + len2*np.cos(alpha+beta), len2*np.cos(alpha+beta)]])

    # calculate delta q
    # damped least-squares method
    damp_const = 50
    damping_matrix = np.linalg.inv(J.T @ J + damp_const**2 * np.eye(2))
    delta_q = damping_matrix @ J.T @ e
    d_alpha = delta_q[0]
    d_beta = delta_q[1]

    # rotate arms
    # update angles
    alpha += d_alpha
    beta += d_beta

    # recalculate vectors from angles
    vector1 = np.array([len1 * np.cos(alpha), len1 * np.sin(alpha)])
    vector2 = np.array([len2 * np.cos(alpha + beta), len2 * np.sin(alpha + beta)])

    return vector1, vector2, alpha, beta
   
# set up drawing window
screen = pygame.display.set_mode([500, 500])
line_color1 = (255, 0, 0)
line_color2 = (0, 255, 0)

# set up math
len1 = 100
len2 = 100
origin = np.array([250, 250])
# starting positions
alpha = 0.1
beta = 0.1
vector_1 = np.array([len1 * np.cos(alpha), len1 * np.sin(alpha)])
vector_2 = np.array([len2 * np.cos(alpha + beta), len2 * np.sin(alpha + beta)]) # update angles

origin_l = origin.tolist()

wntd_x_pos = len1*np.cos(alpha) + len2*np.cos(alpha + beta)
wntd_y_pos = len1*np.sin(alpha) + len2*np.sin(alpha + beta)

max_pos_delt = 3

# run until user asks to quit
running = True;
while running:

    # did the user click the window close button?
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # fill screen background with white
    screen.fill((255, 255, 255))

    pressed_keys = pygame.key.get_pressed()

    wntd_x_pos, wntd_y_pos = update_wanted_q(pressed_keys, wntd_x_pos, wntd_y_pos, max_pos_delt)
    vector_1, vector_2, alpha, beta = update(pressed_keys, len1, len2, alpha, beta, wntd_x_pos, wntd_y_pos)
    pos_1 = origin + vector_1
    pos_2 = pos_1 + vector_2

    # draw a solid blue circle in the center
    pos_1l = pos_1.tolist()
    pos_2l = pos_2.tolist()
    
    rect1 = pygame.Rect(wntd_x_pos+origin[0], wntd_y_pos+origin[1], 2, 2)
    pygame.draw.line(screen, line_color1, origin, pos_1, 5)
    pygame.draw.line(screen, line_color2, pos_1, pos_2, 5)
    pygame.draw.rect(screen, line_color1, rect1)   

    # flip display
    pygame.display.flip()

    # Ensure program maintains a rate of 30 frames per second
    clock.tick(30)

# done! time to quit
pygame.quit()