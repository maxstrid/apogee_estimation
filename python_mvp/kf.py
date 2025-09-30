from scipy.linalg import expm
import numpy as np

# Im implementing this myself so I can do it later on on real hw

# I'm not adding control input here because it doesn't seem useful
class Kf:
    def __init__(self, A, Q, X_initial, P_initial, dt):
        self.X = X_initial
        self.P = P_initial
        #TODO: Replace this with another algorithm because this will kinda suck to do on an mcu
        self.A = expm(A * dt)
        self.Q = self.__discretize_Q(A, Q, dt)


    # This is taken from Ai and is not what we want to do on an mcu, there are methods that do a quicker discretization of both Q and A
    def __discretize_Q(self, A_c, Q_c, dt):
        """Van Loan method for Q discretization"""
        n = A_c.shape[0]
        M = np.block([[-A_c, Q_c], 
                      [np.zeros((n,n)), A_c.T]]) * dt
        Phi = expm(M)
        Q_d = Phi[:n, n:] @ Phi[n:, :n]
        return Q_d

    def update(self, z, H, R):
        self.__predict() 

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.X = self.X + K @ (z - H @ self.X)
        self.P = (np.eye(K.shape[0]) - K @ H) @ self.P

    def __predict(self):
        self.X = self.A @ self.X
        self.P = self.A @ self.P @ self.A.T + self.Q
