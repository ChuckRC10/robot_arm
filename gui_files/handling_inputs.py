# Import pygame.locals for easier access to key coordinates
# Updated to conform to flake8 and black standards
from pygame.locals import (
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
)

def get_movement(pressed_key, max_pos_delta):
    '''
    returns how far the pointer will move on the screen based on key inputs
    '''
    x_mvmnt = 0
    y_mvmnt = 0
    # get user input
    if pressed_key[K_UP]:
        y_mvmnt = -max_pos_delta
    if pressed_key[K_DOWN]:
        y_mvmnt = max_pos_delta
    if pressed_key[K_LEFT]:
        x_mvmnt = -max_pos_delta
    if pressed_key[K_RIGHT]:
        x_mvmnt = max_pos_delta

    return x_mvmnt, y_mvmnt