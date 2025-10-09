import pandas as pd
import matplotlib.pyplot as plt
from data_ingest import FlightDataInterpolator, MassThrustInterpolator
from ekf import Ekf
from jax_dynamics import Dynamics
import jax.numpy as jnp
import numpy as np

def plot(time, thrust, mass, altitude, velocity, acceleration, altitude_est, velocity_est, acceleration_est, external_force, apogee_estimates):
    fig1, axes1 = plt.subplots(2, 1, constrained_layout=True)

    # Plot altitude
    axes1[0].plot(time, thrust)
    axes1[0].set_xlabel('Time (s)')
    axes1[0].set_ylabel('Thrust (N)')
    axes1[0].set_title('Thrust vs Time')

    axes1[1].plot(time, mass)
    axes1[1].set_xlabel('Time (s)')
    axes1[1].set_ylabel('Mass (kg)')
    axes1[1].set_title('Mass vs Time')

    fig2, axes2 = plt.subplots(5, 1, constrained_layout=True)

    axes2[0].plot(time, altitude, color='green', label="Altitude")
    # axes2[0].plot(time, traj_altitude, color='red', linestyle="--", label="Estimated Altitude")
    axes2[0].plot(time, altitude_est, color='blue', linestyle="--", label="Estimated Altitude")
    axes2[0].set_xlabel('Time (s)')
    axes2[0].set_ylabel('Altitude (m)')
    axes2[0].set_title('Altitude vs Time')

    axes2[1].plot(time, velocity, label="Velocity")
    # axes2[1].plot(time, traj_velocity, color='red', linestyle="--", label="Estimated Velocity")
    axes2[1].plot(time, velocity_est, color='blue', linestyle="--", label="Estimated Velocity")
    axes2[1].set_xlabel('Time (s)')
    axes2[1].set_ylabel('Velocity (m/s)')
    axes2[1].set_title('Velocity vs Time')

    axes2[2].plot(time, acceleration, color='red', label="Acceleration")
    axes2[2].plot(time, acceleration_est, color='red', linestyle="--", label="Estimated Acceleration")
    axes2[2].set_xlabel('Time (s)')
    axes2[2].set_ylabel('Acceleration (m/s^2)')
    axes2[2].set_title('Acceleration vs Time')

    axes2[3].plot(time, external_force, color='red', linestyle="--")
    axes2[3].set_xlabel('Time (s)')
    axes2[3].set_ylabel('External Force (N)')
    axes2[3].set_title('External Force vs Time')

    axes2[4].plot(time, apogee_estimates, color='blue', linestyle='--')
    axes2[4].set_xlabel('Time (s)')
    axes2[4].set_ylabel('Apogee Estimate (m)')
    axes2[4].set_title('Apogee Estimate vs Time')

    plt.show()

def main():
    mt_interpolator = MassThrustInterpolator('mass_thrust_curves.csv')
    data_interpolator = FlightDataInterpolator('1st Launch.csv')

    thrust = [] 
    mass = []

    altitude = []
    velocity = []
    acceleration = []

    time = []
    dt = 0.005

    t = 0

    cD = 0.6
    A_ref = 0.0049325
    rho = 1.0

    dynamics = Dynamics(cD, A_ref, rho, mt_interpolator)
    dynamics.compute_jacobians()
    #x0 = jnp.array([data_interpolator.altitude(0.0), data_interpolator.velocity(0.0), 0.0])  # initial position, velocity, external force
    #num = int((1.0 / dt) * 30)
    #trajectory = dynamics.integrate(x0, 0, 30)


    #traj_altitude = trajectory[:, 0]
    #traj_velocity = trajectory[:, 1]

    X_initial = jnp.array([data_interpolator.altitude(0.0), data_interpolator.velocity(0.0), 0.0])
    P_initial = jnp.diag(jnp.array([1e-2, 1e-5, 1e-5]))

    Q = jnp.diag(jnp.array([1.0, 1.0, 0.001]))

    ekf = Ekf(X_initial, P_initial, Q, dynamics)

    altitude_est = []
    velocity_est = []
    acceleration_est = []

    external_force = []

    apogee_estimates = []


    for _ in range(int(13 * (1 / dt))):
        print(t)
        ekf.predict(dt)

        Z = jnp.array([data_interpolator.altitude(t), data_interpolator.acceleration(t)])

        R = jnp.diag(jnp.array([1.0, 10.0]))

        ekf.update(Z, R)

        altitude_est.append(ekf.X[0])
        velocity_est.append(ekf.X[1])
        external_force.append(ekf.X[2])
        acceleration_est.append(dynamics.a(ekf.X, t))

        thrust.append(mt_interpolator.thrust(t))
        mass.append(mt_interpolator.mass(t))

        altitude.append(data_interpolator.altitude(t))
        velocity.append(data_interpolator.velocity(t))
        acceleration.append(data_interpolator.acceleration(t))

        if t > 10.0:
            x0 = ekf.X
            trajectory = dynamics.integrate(x0, t, 13, dt=0.1)

            altitude_traj = trajectory[:, 0]

            apogee_est = jnp.max(altitude_traj)

            apogee_estimates.append(apogee_est)
        else:
            apogee_estimates.append(730)

        time.append(t)

        t += dt

    plot(time, thrust, mass, altitude, velocity, acceleration, altitude_est, velocity_est, acceleration_est, external_force, apogee_estimates)

    
if __name__ == "__main__":
    main()
