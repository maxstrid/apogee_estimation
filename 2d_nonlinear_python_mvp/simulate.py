import argparse

from dynamics import DynamicsConstants
from plant import RocketPlant, RocketPlantConstants
from mpc_solver import MPCSolver
import jax.numpy as jnp

import matplotlib.pyplot as plt

def plot(times, y_positions, y_drag_positions, cds, arefs):
    fig, axes = plt.subplots(2, 1, constrained_layout=True)

    axes[0].plot(times, y_positions, color='green', label="Altitude")
    # axes[0].plot(times, y_drag_positions, color='green', linestyle="--", label="Full Drag Altitude")
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Altitude (m)')
    axes[0].set_title('Altitude vs Time')

    #axes[1].plot(times, cds, color='green', label="Altitude")
    #axes[1].plot(times, arefs, color='green', linestyle="--", label="Full Drag Altitude")
    #axes[1].set_xlabel('Time (s)')
    #axes[1].set_ylabel('Altitude (m)')
    #axes[1].set_title('Altitude vs Time')

    plt.show()

dt = 0.005

dynamics_constants=DynamicsConstants(
    cD=0.6,
    A_ref=0.0049325,
    rho=1.0,

    mass=4.713304,
    moi=0.007,

    r_cp=0.112,
)

def normal_sim():
    constants = RocketPlantConstants(
        dynamics_constants=dynamics_constants,

        initial_velocity=136,
        initial_position=150,

        initial_theta=0.1,
        dt=dt,
    )

    plant = RocketPlant(constants)

    solver = MPCSolver(plant.dynamics)

    x_positions = []
    y_positions = []
    y_drag_positions = []

    x_velocities = []
    y_velocities = []

    thetas = []

    times = []

    cds = []
    arefs = []

    t = 0.0

    #initial_apogee_estimate = plant.dynamics.estimate_apogee(plant.x, jnp.array([0.0, 0.0]))
    #print(initial_apogee_estimate)
    initial_u_guess = jnp.zeros((2500, 2))
    u_seq = solver.solve(plant.x, initial_u_guess, 1000.0, jnp.array([[1.0000, 0.0], [0.0, 1.0000]]), 850.0)


    times, trajectory = plant.integrate_over_u_seq(u_seq)

    y_positions = trajectory[:, 1]

    print(jnp.max(y_positions))

    plot(times, y_positions, y_drag_positions, cds, arefs)

def main():
    parser = argparse.ArgumentParser(description="Run simulations.")
    parser.add_argument("sim", choices=["normal"], help="Which simulation to run")
    args = parser.parse_args()

    sims = {
        "normal": normal_sim,
    }

    sims[args.sim]()
    pass

if __name__ == "__main__":
    main()
