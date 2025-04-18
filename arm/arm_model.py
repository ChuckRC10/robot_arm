import jax.numpy as jnp
from jax import sin, cos

class RobotArm:
    def __init__(self, length1:float, length2: float):
        
        self.l1 = length1
        self.l2 = length2
        self.alpha = 0.0
        self.beta = 0.0
    
    def set_angles(self, a1: float, a2: float):
        self.alpha = a1
        self.beta = a2

    def get_vec1(self):
        x = self.l1 * jnp.cos(self.alpha)
        y = self.l1 * jnp.sin(self.alpha)
        return [x, y]

    def get_vec2(self):
        x = self.l2 * jnp.cos(self.alpha + self.beta)
        y = self.l2 * jnp.sin(self.alpha + self.beta)
        return [x, y]
    
    def get_end_effector(self):
        vec1 = self.get_vec1()
        vec2 = self.get_vec2()
        end_vec = [vec1[0] + vec2[0], vec1[1] + vec2[1]]
        return end_vec
    
    def get_jacobian(self) -> jnp.array:
        # shorten variables
        l1 = self.l1
        l2 = self.l2
        alpha = self.alpha
        beta = self.beta
        # calculate Jacobian
        J = jnp.array([[-l1*sin(alpha)-l2*sin(alpha+beta), -l2*sin(alpha+beta)],
            [l1*cos(alpha) + l2*cos(alpha+beta), l2*cos(alpha+beta)]])
        return J
    
    def get_error(self, wntd_pos:list) -> jnp.array:
        arm_pos = self.get_end_effector()
        e = jnp.array([wntd_pos[0] - arm_pos[0], wntd_pos[1] - arm_pos[1]])
        return e