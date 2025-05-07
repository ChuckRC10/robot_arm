import numpy as np

def rnea(
        q: list[np.ndarray],
        qd: list[np.ndarray],
        qdd: list[np.ndarray],
        f_ext: list[np.ndarray]
) -> list[np.ndarray]:

    # TODO: write  compute_joint function
    # - write spatial_cross function
    # - write spatial_cross_dual function
    
    n = len(qd) - 1 # number of links == number of joints - 1
    lambda_ = create_lambda_from_kinematic_structure("serial", n)
    v = [np.empty((6,)) for i in range(n + 1)]
    a = [np.empty((6,)) for i in range(n + 1)]
    f = [np.empty((6,)) for i in range(n + 1)]
    tau = [np.empty(qd[i].shape) for i in range(n + 1)]
    v[0] = np.zeros((6,))
    a[0] = -np.array([0.0, 0.0, -9.81])

    # forward pass
    for i in range(1, n + 1):
        p = lambda_[1]
        X_p_to_i[i], S[i], I[i] = compute_joint(joint_type[i], q[i])
        v[i] = X_p_to_i[i] * v[p] + S[i] * qd[i]
        a[i] = X_p_to_i[i] * a[p] + S[i] * qdd[i] + spatial_cross(v[i], S[i] * qd[i])
        f[i] = I[i] * a[i] + spatial_cross_dual(v[i], I[i] * v[i]) - f_ext[i]

    for i in range(n, 0, -1)
    p = lambda_[i]
    tau[i] = S[i].T * f[i]
    f[p] += X_p_to_i[i].T * f[i]

    return tau

def create_lambda_from_kinematic_structure(structure: str, n: int):
    if structure == "serial":
        return [None] + list(range(n))
    elif structure == "tree":
        # TODO: implement tree logic
        pass 