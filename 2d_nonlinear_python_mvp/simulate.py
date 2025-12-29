import argparse

from dynamics import DynamicsConstants
from plant import RocketPlant2D, RocketPlantConstants
from mpc_solver import MPCSolver
import jax.numpy as jnp
from dataclasses import asdict
from ekf import Ekf
import math
import jax

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

def base_sim(args):
    def base_plot(times, y_positions, y_drag_positions, y_max, y_drag_max):
        fig, axes = plt.subplots(1, 1, constrained_layout=True)

        axes.plot(times, y_positions, color='green', label="Altitude")
        axes.plot(times, y_drag_positions, color='green', linestyle="--", label="Full Drag Altitude")
        plt.axhline(y=y_max, color='red', linestyle='--', linewidth=1.5, label=f'Normal Peak = {y_max:.6f}')
        plt.axhline(y=y_drag_max, color='blue', linestyle='--', linewidth=1.5, label=f'Drag Peak = {y_drag_max:.6f}')
        axes.set_xlabel('Time (s)')
        axes.set_ylabel('Altitude (m)')
        axes.set_title('Altitude vs Time')

        plt.legend()
        plt.show()

    plant = RocketPlant2D(constants)
    u_seq = jnp.full((int(13.0 / dt), 2), jnp.array([0.445, 0.00948]))

    times, trajectory = plant.integrate(jnp.array([0.0, 0.0]), 13.0)
    y_positions = trajectory[:, 1]

    _, drag_trajectory = plant.integrate_over_u_seq(u_seq)
    y_drag_positions = drag_trajectory[:, 1]

    y_max = jnp.max(y_positions)
    y_drag_max = jnp.max(y_drag_positions)

    base_plot(times, y_positions, y_drag_positions, y_max, y_drag_max)

def nmpc_sim(args):
    def nmpc_plot(times, y_positions, y_mpc_positions, y_max, y_mpc_max, y_goal):
        fig, axes = plt.subplots(1, 1, constrained_layout=True)

        axes.plot(times, y_positions, color='green', label="Altitude")
        axes.plot(times, y_mpc_positions, color='orange' , label="MPC Policy Controlled Altitude")
        axes.axhline(y=y_max, color='red', linestyle='--', linewidth=1.5, label=f'Normal Peak = {y_max:.2f}')
        axes.axhline(y=y_goal, color='purple', linestyle='--', linewidth=1.5, label=f'Goal Peak = {y_goal:.2f}')
        axes.axhline(y=y_mpc_max, color='blue', linestyle='--', linewidth=1.5, label=f'MPC Peak = {y_mpc_max:.2f}')
        axes.set_xlabel('Time (s)')
        axes.set_ylabel('Altitude (m)')
        axes.set_title('Altitude vs Time')

        plt.legend()
        plt.show()

    plant = RocketPlant2D(constants)
    solver = MPCSolver(plant.dynamics)

    initial_u_guess = jnp.zeros((int(13.0 / dt), 2))
    u_seq = solver.solve(
            plant.x, # Initial X
            initial_u_guess,
            1000.0, # Q
            jnp.diag(jnp.array([1.0, 1.0])), # R
            args.goal, # Goal apogee
            steps=args.iterations,
    )

    times, trajectory = plant.integrate(jnp.array([0.0, 0.0]), 13.0)
    _, mpc_trajectory = plant.integrate_over_u_seq(u_seq)

    y_positions = trajectory[:, 1]
    y_mpc_positions = mpc_trajectory[:, 1]

    y_max = jnp.max(y_positions)
    y_mpc_max = jnp.max(y_mpc_positions)

    nmpc_plot(times, y_positions, y_mpc_positions, y_max, y_mpc_max, args.goal)

def ekf_sim(args):
    def ekf_plot(times, y_positions, y_est_positions):
        fig, axes = plt.subplots(1, 1, constrained_layout=True)

        axes.plot(times, y_positions, color='green', label="Altitude")
        axes.plot(times, y_est_positions, color='orange', linestyle="--", label="Estimated Altitude")
        axes.set_xlabel('Time (s)')
        axes.set_ylabel('Altitude (m)')
        axes.set_title('Altitude vs Time')

        plt.legend()
        plt.show()
    plant = RocketPlant2D(constants)

    P_initial = jnp.eye(plant.x.shape[0]) * 1e-3
    Q = jnp.diag(jnp.array([
        1.0, # p_x
        0.01, # p_y
        0.01, # v_x
        1.0, # v_y
        0.5, # f_e_n
        5.0, # f_e_a
        0.01, # theta
        0.01, # omega
    ]))

    w_sigmas = jnp.array([
        0.05, # p_x
        0.0, # p_y
        0.05, # v_x
        0.0, # v_y
        0.5, # f_e_n
        5.0, # f_e_a
        0.01, # theta
        0.01, # omega
    ])

    R = jnp.diag(jnp.array([
        100.0, # p_y
        5.0, # a_x
        5.0, # a_y
        0.5 # omega
    ]))

    v_sigmas = jnp.array([
        10.0, # p_y
        10.0, # a_x
        10.0, # a_y
        0.5, # omega
    ])

    ekf = Ekf(plant.x, P_initial, Q, R, plant.dynamics)

    u = jnp.array([0.0, 0.0])

    def scan_fn(carry, _):
        (ekf, key), (x, x_hat) = carry

        key, proc_key, sensor_key = jax.random.split(key, 3)

        # Get current state
        x = ekf.dynamics._rk4_step(x, u, dt)

        w = jax.random.normal(proc_key, shape=x.shape) * w_sigmas

        x = x + w

        # create z
        z = ekf.dynamics.h(x, u)

        v = jax.random.normal(sensor_key, shape=z.shape) * v_sigmas

        z = z + v

        # run ekf

        ekf = ekf.predict(u, dt).update(z, u)

        return ((ekf, key), (x, ekf.X_hat)), (x, ekf.X_hat)


    n_steps = math.ceil(13.0 / dt)

    init_carry = ((ekf, jax.random.PRNGKey(0)), (plant.x, plant.x))

    _, (trajectory, trajectory_estimate) = jax.lax.scan(scan_fn, init_carry, None, length=n_steps)
    times = jnp.arange(n_steps) * dt

    y_positions = trajectory[:, 1]

    y_est_positions = trajectory_estimate[:, 1]

    ekf_plot(times, y_positions, y_est_positions)


def main():
    parser = argparse.ArgumentParser(description="Run simulations.")
    subparsers = parser.add_subparsers(dest="sim", required=True, help="Which simulation to run")

    parser_nmpc = subparsers.add_parser("nmpc", help="Run the NMPC simulation")
    parser_nmpc.add_argument("--iterations", type=int, default=300, help="Number of iterations")
    parser_nmpc.add_argument("--goal", type=int, default=850, help="Goal apogee to reach")
    parser_nmpc.set_defaults(func=nmpc_sim)

    parser_base = subparsers.add_parser("base", help="Run the Base simulation")
    parser_base.set_defaults(func=base_sim)

    parser_ekf = subparsers.add_parser("ekf", help="Run the EKF simulation")
    parser_ekf.set_defaults(func=ekf_sim)

    args = parser.parse_args()

    args.func(args)

if __name__ == "__main__":
    main()
