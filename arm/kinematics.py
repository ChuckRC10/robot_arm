import numpy as np

def inv_kinematics_least_sqr(J: np.array, e: np.array, damping_c: float) -> list:
    """Calculate delta q, damped least-squares"""
    damping_matrix = np.linalg.inv(J.T @ J + damping_c**2 * np.eye(2))
    delta_q = damping_matrix @ J.T @ e
    return delta_q.tolist()