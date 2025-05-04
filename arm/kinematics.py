import jax.numpy as jnp

def inv_kinematics_least_sqr(Jacobian: jnp.array, error: jnp.array, damping_c: float) -> list:
    """Calculate delta q, damped least-squares"""
    eye_size = Jacobian.shape[1]
    damping_matrix = jnp.linalg.inv(Jacobian.T @ Jacobian + damping_c**2 * jnp.eye(eye_size))
    delta_q = damping_matrix @ Jacobian.T @ error
    return delta_q


def getVelocityEllipseAngle(ellipseEquationMatrix):
    _, eigenvectors = jnp.linalg.eig(ellipseEquationMatrix)
    angle = jnp.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
    if jnp.isnan(angle):
        angle = 0

    return float(jnp.real(angle))

def get_ellipse_size(ellipseEquationMatrix):
    #TODO: Test if this even works right
    
    eigenvalues = jnp.real(jnp.linalg.eigvals(ellipseEquationMatrix))

    axisSize1 = 1/jnp.sqrt(eigenvalues[0])
    axisSize2 = 1/jnp.sqrt(eigenvalues[1])

    maximum_size = 200
    minimum_size = .1
    if jnp.isinf(axisSize1):
        axisSize1 = maximum_size
    if jnp.isinf(axisSize2):
        axisSize2 = maximum_size

    if jnp.isnan(axisSize1) or axisSize1 < minimum_size:
        axisSize1 = minimum_size
    if jnp.isnan(axisSize2) or axisSize2 < minimum_size:
        axisSize2 = minimum_size

    return jnp.array([axisSize1, axisSize2])

def getEllipseArea(axisSize1, axisSize2) -> float:
    area = jnp.pi * axisSize1 * axisSize2
    return float(area)