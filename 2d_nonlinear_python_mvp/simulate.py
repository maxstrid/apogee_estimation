import argparse

from dynamics import DynamicsConstants
from plant import RocketPlant, RocketPlantConstants
from mpc_solver import MPCSolver
import jax.numpy as jnp

import matplotlib.pyplot as plt



dt = 0.005

dynamics_constants=DynamicsConstants(
    cD=0.6,
    A_ref=0.0049325,
    rho=1.0,

    mass=4.713304,
    moi=0.007,

    r_cp=0.112,
)

constants = RocketPlantConstants(
    dynamics_constants=dynamics_constants,

    initial_velocity=136,
    initial_position=150,

    initial_theta=0.1,
    dt=dt,
)

def base_sim():
    def base_plot(times, y_positions, y_drag_positions, y_max, y_drag_max):
        fig, axes = plt.subplots(1, 1, constrained_layout=True)

        axes.plot(times, y_positions, color='green', label="Altitude")
        axes.plot(times, y_drag_positions, color='green', linestyle="--", label="Full Drag Altitude")
        plt.axhline(y=y_max, color='red', linestyle='--', linewidth=1.5, label=f'Normal Peak = {y_max:.2f}')
        plt.axhline(y=y_drag_max, color='blue', linestyle='--', linewidth=1.5, label=f'Drag Peak = {y_drag_max:.2f}')
        axes.set_xlabel('Time (s)')
        axes.set_ylabel('Altitude (m)')
        axes.set_title('Altitude vs Time')

        plt.legend()
        plt.show()

    plant = RocketPlant(constants)
    u_seq = jnp.full((int(13.0 / dt), 2), jnp.array([0.445, 0.00948]))

    times, trajectory = plant.integrate(jnp.array([0.0, 0.0]), 13.0)
    y_positions = trajectory[:, 1]

    _, drag_trajectory = plant.integrate_over_u_seq(u_seq)
    y_drag_positions = drag_trajectory[:, 1]

    y_max = jnp.max(y_positions)
    y_drag_max = jnp.max(y_drag_positions)

    base_plot(times, y_positions, y_drag_positions, y_max, y_drag_max)

def nmpc_sim():
    def nmpc_plot(times, y_positions, y_mpc_positions):
        fig, axes = plt.subplots(2, 1, constrained_layout=True)

        axes[0].plot(times, y_positions, color='green', label="Altitude")
        axes[0].plot(times, y_mpc_positions, color='green', linestyle="--", label="MPC Policy Controlled Altitude")
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Altitude (m)')
        axes[0].set_title('Altitude vs Time')

        plt.legend()
        plt.show()

    plant = RocketPlant(constants)
    solver = MPCSolver(plant.dynamics)

    initial_u_guess = jnp.zeros((int(13.0 / dt), 2))
    u_seq = solver.solve(
            plant.x, # Initial X
            initial_u_guess,
            1000.0, # Q
            jnp.diag(jnp.array([1.0, 1.0])), # R
            850.0 # Goal apogee
    )

    times, trajectory = plant.integrate(jnp.array([0.0, 0.0]), 13.0)
    _, mpc_trajectory = plant.integrate_over_u_seq(u_seq)

    y_positions = trajectory[:, 1]
    y_mpc_positions = mpc_trajectory[:, 1]

    nmpc_plot(times, y_positions, y_mpc_positions)

def main():
    parser = argparse.ArgumentParser(description="Run simulations.")
    parser.add_argument("sim", choices=["nmpc", "base"], help="Which simulation to run")
    args = parser.parse_args()

    sims = {
        "nmpc": nmpc_sim,
        "base": base_sim,
    }

    sims[args.sim]()
    pass

if __name__ == "__main__":
    main()
