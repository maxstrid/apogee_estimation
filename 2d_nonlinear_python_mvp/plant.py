import jax
import jax.numpy as jnp

from dataclasses import dataclass

from dynamics import Dynamics, DynamicsConstants, g

@dataclass
class RocketPlantConstants:
    dynamics_constants: DynamicsConstants
    
    # We are gonna assume we start with 0 tangential velocity
    initial_velocity: float
    # We are gonna assume we aren't off at all in x position.
    initial_position: float

    initial_theta: float

    dt: float


# This class handles the main simulation of the rocket dynamics using the Dynamics class
# All we are simulating is the time after coast
class RocketPlant:
    def __init__(self, constants: RocketPlantConstants) -> None:
        self.constants = constants

        self.dynamics = Dynamics(constants.dynamics_constants)

        self.x = jnp.array([0.0, constants.initial_position, 0.0, constants.initial_velocity, 0.0, 0.0, constants.initial_theta, 0.0])

        self.dt = self.constants.dt

    @jax.jit(static_argnames=['self', 't']) 
    def integrate(self, u, t):
        return self.dynamics.integrate(self.x, u, t, self.dt)

    @jax.jit(static_argnames=['self']) 
    def integrate_over_u_seq(self, u_sequence):
        return self.dynamics.integrate_over_u_seq(self.x, u_sequence, self.dt)

    def update(self, u):
        self.x = self.dynamics._rk4_step(self.x, u, self.dt)

    def reset(self):
        self.x = jnp.array([0.0, self.constants.initial_position, 0.0, self.constants.initial_velocity, 0.0, 0.0, self.constants.initial_theta, 0.0])
