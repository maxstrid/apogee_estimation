import jax.numpy as jnp
import jax.scipy as jscp

from jax_dynamics import Dynamics, rk4_tx

class Ekf:
    def __init__(self, X_initial, P_initial, Q, dynamics):
        self.X = X_initial
        self.P = P_initial

        self.Q = Q

        self.dynamics = dynamics

        self.t = 0.0

    def predict(self, dt):
        A = self.dynamics.A(self.t, self.X)

        self.t += dt

        A_d, Q_d = self.__discretize_AQ(A, self.Q, dt)

        self.X = rk4_tx(self.dynamics.f, self.X, self.t, dt)

        self.P = A_d @ self.P @ A_d.transpose() + Q_d

    def update(self, Z, R):
        H = self.dynamics.H(self.t, self.X)

        S = H @ self.P @ H.transpose() + R

        K = self.P @ H.transpose() @ jnp.linalg.inv(S)

        self.X = self.X + K @ (Z - self.dynamics.h(self.t, self.X))

        self.P = (jnp.eye(self.P.shape[0]) - K @ H) @ self.P

    def __discretize_AQ(self, A, Q, dt):
        n = A.shape[0]
        M = jnp.block([
            [A, Q],
            [jnp.zeros((n, n)), -A.T]
        ])

        M_exp = jscp.linalg.expm(M * dt)

        A_d = jscp.linalg.expm(A * dt)
        Q_d = A_d @ M_exp[:n, n:]

        return (A_d, Q_d)
