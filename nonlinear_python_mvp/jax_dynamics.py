from pandas.core.generic import DtypeBackend
from data_ingest import MassThrustInterpolator
import jax
import jax.numpy as jnp
from scipy.constants import g

# x = [x, v, F_e]
# u = [thrust, mass]
# z = [p, a]


class Dynamics:
    def __init__(self, cD, A_ref, rho, mt_interpolator: MassThrustInterpolator) -> None:
        self.cD = cD
        self.A_ref = A_ref
        self.rho = rho

        self.mt_interpolator = mt_interpolator

    # RK4 Integration for a function which takes in t and x
    def __rk4_tx(self, f, x, t, dt):
        h = dt
        U = lambda t: jnp.array([self.mt_interpolator.thrust(t), self.mt_interpolator.mass(t)])

        k1 = f(x, U(t))
        k2 = f(x + h * (k1/2.0), U(t + (h/2.0)))
        k3 = f(x + h * (k2/2.0), U(t + (h/2.0)))
        k4 = f(x + h * k3, U(t+h))

        return x + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def a(self, x, U) -> float:
        F_thrust = U[0]

        mass = U[1]

        F_weight = mass * g

        F_external = x[2]

        F_drag = 0.5 * self.cD * self.rho * self.A_ref * x[1]**2 * jax.nn.soft_sign(x[1])

        return (F_thrust - F_drag - F_weight + F_external)/mass

    def integrate(self, x0, t0, t1, dt=0.005):
        n_steps = jnp.ceil((t1 - t0) / dt).astype(int)

        def loop(i, carry):
            t, x0, x_max = carry
            x = self.__rk4_tx(self.f, x0, t, dt)
            t += dt
            x_max = jnp.maximum(x[0], x_max)

            return (t, x, x_max)

        _,_,x_max = jax.lax.fori_loop(lower=0, upper=n_steps, body_fun=loop, init_val=(t0, x0, x0[0]))

        return x_max 

    def f(self, x, U):
        return jnp.array([x[1], self.a(x, U), 0])

    def h(self, x, U):
        return jnp.array([x[0], self.a(x, U)])


