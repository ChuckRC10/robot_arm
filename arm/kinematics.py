import numpy as np

def inv_kinematics_least_sqr(Jacobian: np.array, error: np.array, damping_c: float) -> list:
    """Calculate delta q, damped least-squares"""
    eye_size = Jacobian.shape[1]
    damping_matrix = np.linalg.inv(Jacobian.T @ Jacobian + damping_c**2 * np.eye(eye_size))
    delta_q = damping_matrix @ Jacobian.T @ error
    return delta_q

def R_z(theta):
    """Rotation matrix about z-axis in 3D."""
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])

def forward_kinematics(joint_angles, link_lengths):
    """compute forward kinematics on robot arm with arbitrary number of links
       to get end effector position. Uses homogeneous transform matrix."""

    assert len(joint_angles) == len(link_lengths), "Each joint angle must have a corresponding link length."
    
    T = np.eye(4)

    for theta, length in zip(joint_angles, link_lengths):
        R = R_z(theta)
        translation_vector = np.array([length, 0, 0])
        T_step = make_T_matrix(R, translation_vector)
        T = T @ T_step

    ee_pos = T @ np.array([0, 0, 0, 1])
    return ee_pos[:3]

def make_T_matrix(R, translation_vector):
    """Construct a 4x4 homogeneous transformation matrix."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = translation_vector
    return T

def getVelocityEllipseAngle(ellipseEquationMatrix):
    _, eigenvectors = np.linalg.eig(ellipseEquationMatrix)
    angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
    if np.isnan(angle):
        angle = 0

    return float(np.real(angle))

def get_ellipse_size(ellipseEquationMatrix):
    #TODO: Test if this even works right
    
    eigenvalues = np.real(np.linalg.eigvals(ellipseEquationMatrix))

    axisSize1 = 1/np.sqrt(eigenvalues[0])
    axisSize2 = 1/np.sqrt(eigenvalues[1])

    maximum_size = 200
    minimum_size = .1
    if np.isinf(axisSize1):
        axisSize1 = maximum_size
    if np.isinf(axisSize2):
        axisSize2 = maximum_size

    if np.isnan(axisSize1) or axisSize1 < minimum_size:
        axisSize1 = minimum_size
    if np.isnan(axisSize2) or axisSize2 < minimum_size:
        axisSize2 = minimum_size

    return np.array([axisSize1, axisSize2])

def getEllipseArea(axisSize1, axisSize2) -> float:
    area = np.pi * axisSize1 * axisSize2
    return float(area)