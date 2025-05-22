import jax
import numpy as np
import jax.numpy as jnp
from numpy import sin, cos

class RobotArm:
    def __init__(self, armLengths:np.array):
        
        self.armLengths = armLengths
        self.armNumber = len(armLengths)
        self.armAngles = np.ones(self.armNumber)

    def set_angles(self, armAngles: np.array):
        self.armAngles = armAngles

    def getArmVectors(self, angles) -> jnp.array:
        lens = self.armLengths
        globalAngles = jnp.cumsum(angles)

        # calculate vector coordinates
        xArray = lens * jnp.cos(globalAngles)
        yArray = lens * jnp.sin(globalAngles)
        
        armVectorArray = jnp.array([xArray, yArray]).T
        return armVectorArray

    def get_end_effector(self, angles) -> jnp.array:
        armVectorArray = self.getArmVectors(angles)
        armEndVector = jnp.sum(armVectorArray, axis=0)

        return armEndVector
    
    def get_jacobian(self) -> np.array:
        J = jax.jacrev(lambda angles: self.get_end_effector(angles))
        return  np.array(J(self.armAngles))
    
    def get_error(self, wntd_pos: np.array) -> np.array:
        arm_pos = self.get_end_effector(self.armAngles)
        error = wntd_pos - arm_pos
        return error