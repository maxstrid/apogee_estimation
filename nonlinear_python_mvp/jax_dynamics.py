from pandas.core.generic import DtypeBackend
from data_ingest import MassThrustInterpolator
import jax
import jax.numpy as jnp

# x = [x, v, F_e]
# u = []
# z = [p, a]

g = 9.81
# RK4 Integration for a function which takes in t and x
def rk4_tx(f, x, t, dt):
    h = dt

    k1 = f(t, x)
    k2 = f(t + (h/2.0), x + h * (k1/2.0))
    k3 = f(t + (h/2.0), x + h * (k2/2.0))
    k4 = f(t + h, x + h * k3)

    return x + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)

class Dynamics:
    def __init__(self, cD, A_ref, rho, mt_interpolator: MassThrustInterpolator) -> None:
        self.cD = cD
        self.A_ref = A_ref
        self.rho = rho

        self.mt_interpolator = mt_interpolator

    def a(self, x, t) -> float:
        F_thrust = self.mt_interpolator.thrust(t)

        mass = self.mt_interpolator.mass(t)

        F_weight = mass * g

        F_external = x[2]

        F_drag = 0.5 * self.cD * self.rho * self.A_ref * x[1]**2 * jax.nn.soft_sign(x[1])

        return (F_thrust - F_drag - F_weight + F_external)/mass

    def integrate(self, x0, t0, t1, dt=0.005):
        n_steps = int(jnp.ceil((t1 - t0) / dt))
        x = x0
        trajectory = []

        t = t0
        for _ in range(n_steps):
            x = rk4_tx(self.f, x, t, dt)
            trajectory.append(x)
            t += dt

        return jnp.stack(trajectory)

    def compute_jacobians(self):
        self.__A_jit = jax.jit(jax.jacrev(self.f, argnums=1))
        self.__H_jit = jax.jit(jax.jacrev(self.h, argnums=1))

    def f(self, t, x) -> float:
        return jnp.array([x[1], self.a(x, t), 0])

    def h(self, t, x) -> float:
        return jnp.array([x[0], self.a(x, t)])

    def A(self, t, x):
        return self.__A_jit(t, x)

    def H(self, t, x):
        return self.__H_jit(t, x)

