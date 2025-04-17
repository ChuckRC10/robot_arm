import pygame
from gui_files.game_view import game_setup, did_quit, paint_arm, paint_rect
from gui_files.handling_inputs import get_movement
from config import background_color, max_pos_delta, len1, len2, damping_const
from arm.arm_model import RobotArm
from arm import kinematics

clock, screen = game_setup() # initialize window
arm = RobotArm(len1, len2) # initialize arm class

wntd_pos = arm.get_end_effector()

running = True
while running:

    # 1st screen update
    #running = did_quit()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False  
        else:
            running = True
            
    screen.fill(background_color)

    # get info to move arm
    pressed_key = pygame.key.get_pressed()
    x_mvmnt, y_mvmnt = get_movement(pressed_key, max_pos_delta)

    # update wanted position
    wntd_pos[0] += x_mvmnt
    wntd_pos[1] += y_mvmnt

    # get change in arm angles
    jacobian = arm.get_jacobian()
    error = arm.get_error(wntd_pos)
    delta_q = kinematics.inv_kinematics_least_sqr(jacobian, error, damping_const)

    # update arm angles
    arm.set_angles(arm.alpha + delta_q[0], arm.beta + delta_q[1])

    # set new arm vectors
    vec1 = arm.get_vec1()
    vec2 = arm.get_vec2()

    # 2nd screen update
    paint_rect(screen, wntd_pos)
    paint_arm(screen, vec1, vec2)

    # refresh screen
    pygame.display.flip()
    clock.tick(30) # ensure program maintains 30fps

pygame.quit()

