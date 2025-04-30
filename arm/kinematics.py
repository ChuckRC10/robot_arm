import jax.numpy as jnp

def inv_kinematics_least_sqr(Jacobian: jnp.array, error: jnp.array, damping_c: float) -> list:
    """Calculate delta q, damped least-squares"""
    eye_size = Jacobian.shape[1]
    damping_matrix = jnp.linalg.inv(Jacobian.T @ Jacobian + damping_c**2 * jnp.eye(eye_size))
    delta_q = damping_matrix @ Jacobian.T @ error
    return delta_q

def get_velocity_ellipse_coefficients(Jacobian: jnp.array) -> jnp.array:
    """Calculate coefficients A, B, C for an ellipse equation the form Ax^2 + Bxy + Cy^2 = D
    
    might want to move to statics folder - its statics
    """
    ellipsoid_equation_matrix = jnp.linalg.inv(Jacobian @ Jacobian.T)

    A_coefficient = ellipsoid_equation_matrix[0, 0] + ellipsoid_equation_matrix[1, 0]
    B_coefficient = jnp.sum(ellipsoid_equation_matrix)
    C_coefficient = ellipsoid_equation_matrix[0, 1] + ellipsoid_equation_matrix[1, 1]

    return jnp.array([A_coefficient, B_coefficient, C_coefficient])

def get_ellipse_size(ellipse_coefficients):
    #TODO: Test if this even works right

    A_coefficient = ellipse_coefficients[0]
    B_coefficient = ellipse_coefficients[1]
    C_coefficient = ellipse_coefficients[2]

    conic_quadratic_matrix = jnp.array([A_coefficient, B_coefficient/2],
                                       [B_coefficient/2, C_coefficient])
    
    eigenvalue_array = jnp.linalg.eig(conic_quadratic_matrix)

    new_A_coefficient = eigenvalue_array[0]
    new_B_coefficient = eigenvalue_array[1]

    major_axis_size = 1/jnp.sqrt(new_A_coefficient)
    minor_axis_size = 1/jnp.sqrt(new_B_coefficient)

    return jnp.array([major_axis_size, minor_axis_size])
