import numpy as np
from scipy.linalg import expm

def discretize_system(A_c, Q_c, dt):
    """Discretize continuous-time system (A_c, Q_c) with timestep dt."""
    # Build block matrix for Van Loan method
    n = A_c.shape[0]
    M = np.zeros((2*n, 2*n))
    M[0:n, 0:n] = -A_c
    M[0:n, n:2*n] = Q_c
    M[n:2*n, n:2*n] = A_c.T

    # Matrix exponential
    expM = expm(M * dt)

    # Extract blocks
    Phi12 = expM[0:n, n:2*n]
    Phi22 = expM[n:2*n, n:2*n]

    # Discretized matrices
    A_d = expm(A_c * dt)
    Q_d = Phi22.T @ Phi12

    return A_d, Q_d

# Example usage
A_c = np.array([[0, 1, 0],
                [0, 0, 1],
                [0, 0, 0]])
Q_c = np.array([[10, 0, 0],
                [0, 10, 0],
                [0, 0, 10]])
dt = 0.005

A_d, Q_d = discretize_system(A_c, Q_c, dt)
print("A_d:\n", A_d)
print("Q_d:\n", Q_d)
