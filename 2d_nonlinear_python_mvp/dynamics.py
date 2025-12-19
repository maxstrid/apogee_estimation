import jax
import jax.numpy as jnp

import math

from dataclasses import dataclass

g = 9.81

def rk4_xu(f, x, u, dt):
    h = dt

    k1 = f(x, u)
    k2 = f(x + h * (k1/2.0), u)
    k3 = f(x + h * (k2/2.0), u)
    k4 = f(x + h * k3, u)

    return x + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)

@jax.tree_util.register_pytree_node_class
@dataclass
class DynamicsConstants:
    # Drag coefficients
    cD: float
    A_ref: float
    rho: float
    
    mass: float
    moi: float
    
    # Radius from the CoG to the CoP
    r_cp: float

    def tree_flatten(self):
        children = (self.cD, self.A_ref, self.rho, self.mass, self.moi, self.r_cp)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

# This is a 2d dynamics function which assume we already know the theta and omega about which the rocket is rotated, 
# which requires an estimation of our 3d state to find that single theta and omega.

# x = [p_x p_y v_x v_y f_e_n f_e_a theta omega]^T
# u = [cd_brake a_brake]^T
# z = [p a_x a_y omega]^T

# The mpc dynamics have to work a little differently, because we can find a terminal apogee estimate and we know our goal apogee. So we can build a quadratic cost function:
# J = Q(apogee - goal_apogee)^2 + ∫_t^t_apogee U^TRU
# We don't integrate the apogee here because its only happening at the terminal state of our system.
# We do want to integrate the control input cost from t to t_apogee because it changes over that time frame.

@jax.tree_util.register_pytree_node_class
class Dynamics:
    def __init__(self, constants: DynamicsConstants) -> None:
        self.constants = constants

    def a_x(self, x, u):
        return self.__a(x, u)[0]

    def a_y(self, x, u):
        return self.__a(x, u)[1]

    def alpha(self, x, u):
        return self.__a(x, u)[2]

    def tree_flatten(self):
        children = (self.constants,)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
    
    # f defined such that x_dot = f(x, u)
    def f(self, x, u):
        v_x = x[2]
        v_y = x[3]

        omega = x[7]

        a = self.__a(x, u)

        return jnp.array([v_x, v_y, a[0], a[1], 0, 0, omega, a[2]])
    
    # h defined such that z = h(x, u)
    def h(self, x, u):
        a = self.__a(x, u)

        omega = x[7]

        p_y = x[1]

        return jnp.array([p_y, a[0], a[1], omega])
    
    # Integrates our state from x0 for t
    def integrate(self, x0, u, t, dt=0.005):
        n_steps = math.ceil(t / dt)
        
        def scan_fn(x, _):
            x_next = self._rk4_step(x, u, dt)
            return x_next, x_next
        
        _, trajectory = jax.lax.scan(scan_fn, x0, None, length=n_steps)
        time = jnp.arange(n_steps) * dt
        return time, trajectory

    # Integrates our state from x0 using a sequence of control inputs
    def integrate_over_u_seq(self, x0, u_sequence, dt=0.005):
        def scan_fn(x, u):
            x_next = self._rk4_step(x, u, dt)
            return x_next, x_next
        
        _, trajectory = jax.lax.scan(scan_fn, x0, u_sequence)
        
        n_steps = u_sequence.shape[0]
        time = jnp.arange(n_steps) * dt
        
        return time, trajectory
    
    # Given some initial state and control conditions predict what the terminal apogee of our rocket will be.
    @jax.jit(static_argnames=['dt'])
    def estimate_apogee(self, x0, u, dt=0.005):
        def condition_fn(x):
            velocity_y = x[3]
            return velocity_y >= 0

        def while_fn(x):
            x_next = self._rk4_step(x, u, dt)
            return x_next

        x_f = jax.lax.while_loop(condition_fn, while_fn, x0)
        return x_f[1]

    # A matrix as the linearized jacobian of f evaluated at x and u
    def A(self, x, u):
        return jax.jacrev(self.f, argnums=0)(x, u)

    # H matrix as the linearized jacobian of h evaluated at x and u
    def H(self, x, u):
        return jax.jacrev(self.h, argnums=0)(x, u)

    # Returns (a_x, a_y, alpha)
    def __a(self, x, u):
        v_x = x[2]
        v_y = x[3]

        f_e_n = x[4]
        f_e_a = x[5]

        theta = x[6]

        cd_brake = u[0]
        a_brake = u[1]

        cos_theta = jnp.cos(theta)
        sin_theta = jnp.sin(theta)

        R = jnp.array([[cos_theta, -sin_theta],
                    [sin_theta, cos_theta]])

        v_body = R.T @ jnp.array([v_x, v_y]) 

        v = v_body[1]

        # Dynamics in the rocket body frame

        # Force along the rocket axis
        F_a = -self.constants.rho * 0.5 * (a_brake + self.constants.A_ref) * (cd_brake + self.constants.cD) * v**2 * jnp.sign(v) + f_e_a

        # Force normal to the rocket axis
        F_n = f_e_n

        # Torque about CoG
        T = -F_n * self.constants.r_cp

        # Dynamics in the world frame

        F_net = R @ jnp.array([F_n, F_a])

        F_net = F_net.at[1].add(-self.constants.mass * g)

        a_x = F_net[0]/self.constants.mass
        a_y = F_net[1]/self.constants.mass

        alpha = T/self.constants.moi

        return jnp.array([a_x, a_y, alpha])
    
    # Defines the mpc cost function J = Q(apogee - goal_apogee)^2 + ∫_t^t_apogee(U^TRU)
    # This makes a few assumptions:
    #   1. The apogee is always inside of the horizon
    #   2. U is not applied after apogee
    def mpc_cost_fn(self, Q, R, x0, u_sequence, goal_apogee, dt=0.005):
        
        def scan_fn(x, u):
            x_next = self._rk4_step(x, u, dt)

            control_cost = u @ R @ u

            return x_next, (x_next, control_cost)

        _, (trajectory, cost_trajectory) = jax.lax.scan(scan_fn, x0, u_sequence)

        position_y = trajectory[:, 1]
        velocity_y = trajectory[:, 3]

        # mask control effort to before apogee
        is_rising = jnp.where(velocity_y > 0, 1.0, 0.0)
        mask = jnp.cumprod(is_rising)

        # This makes cost only effective before we hit apogee
        integral_cost = jnp.sum(cost_trajectory * mask) * dt

        apogee = jnp.max(position_y)

        terminal_cost = Q * (apogee - goal_apogee)**2

        return terminal_cost + integral_cost
    
    def _rk4_step(self, x, u, dt):
        return rk4_xu(self.f, x, u, dt)
