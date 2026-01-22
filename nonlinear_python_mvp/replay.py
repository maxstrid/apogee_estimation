import pandas as pd
import matplotlib.pyplot as plt
from pandas.core.series import series_sub_kwargs
from data_ingest import FlightDataInterpolator, MassThrustInterpolator
from jaxfilter import ekf, rts_smoother
from jax_dynamics import Dynamics
import jax.numpy as jnp
import jax
import numpy as np

def plot(time, thrust, mass, altitude, velocity, acceleration, altitude_est, velocity_est, acceleration_est, external_force, apogee_estimates, altitude_est_rts, velocity_est_rts, external_force_rts, apogee_estimates_rts):
    fig1, axes1 = plt.subplots(2, 1, constrained_layout=True)

    # Plot altitude
    axes1[0].plot(time, thrust)
    axes1[0].set_xlabel('Time (s)')
    axes1[0].set_ylabel('Thrust (N)')
    axes1[0].set_title('Thrust vs Time')
    axes1[0].legend()

    axes1[1].plot(time, mass)
    axes1[1].set_xlabel('Time (s)')
    axes1[1].set_ylabel('Mass (kg)')
    axes1[1].set_title('Mass vs Time')
    axes1[1].legend()

    fig2, axes2 = plt.subplots(5, 1, constrained_layout=True)

    axes2[0].plot(time, altitude, color='green', label="Altitude")
    axes2[0].plot(time, altitude_est, color='blue', linestyle="--", label="Estimated Altitude")
    axes2[0].plot(time[:-1], altitude_est_rts, color='green', label="Estimated Altitude")
    axes2[0].set_xlabel('Time (s)')
    axes2[0].set_ylabel('Altitude (m)')
    axes2[0].set_title('Altitude vs Time')
    axes2[0].legend()

    axes2[1].plot(time, velocity, label="Velocity")
    axes2[1].plot(time, velocity_est, color='blue', linestyle="--", label="Estimated Velocity")
    axes2[1].plot(time[:-1], velocity_est_rts, color='green', label="Estimated Velocity")
    axes2[1].set_xlabel('Time (s)')
    axes2[1].set_ylabel('Velocity (m/s)')
    axes2[1].set_title('Velocity vs Time')
    axes2[1].legend()

    axes2[2].plot(time, acceleration, color='red', label="Acceleration")
    axes2[2].plot(time, acceleration_est, color='red', linestyle="--", label="Estimated Acceleration")
    axes2[2].set_xlabel('Time (s)')
    axes2[2].set_ylabel('Acceleration (m/s^2)')
    axes2[2].set_title('Acceleration vs Time')
    axes2[2].legend()

    axes2[3].plot(time, external_force, color='red', linestyle="--")
    axes2[3].plot(time[:-1], external_force_rts, color='green')
    axes2[3].set_xlabel('Time (s)')
    axes2[3].set_ylabel('External Force (N)')
    axes2[3].set_title('External Force vs Time')
    axes2[3].legend()

    axes2[4].plot(time, apogee_estimates, color='blue', linestyle='--')
    axes2[4].plot(time[:-1], apogee_estimates_rts, color='green')
    axes2[4].set_xlabel('Time (s)')
    axes2[4].set_ylabel('Apogee Estimate (m)')
    axes2[4].set_title('Apogee Estimate vs Time')
    axes2[4].legend()

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


    cD = 0.6
    A_ref = 0.0049325
    rho = 1.0

    dynamics = Dynamics(cD, A_ref, rho, mt_interpolator)

    X_initial = jnp.array([data_interpolator.altitude(0.0), data_interpolator.velocity(0.0), 0.0])
    P_initial = jnp.diag(jnp.array([1e-2, 1e-5, 1e-5]))

    Q = jnp.diag(jnp.array([1.0, 1.0, 0.001]))
    R = jnp.diag(jnp.array([1.0, 10.0]))

    U_initial = jnp.array([mt_interpolator.thrust(0.0), mt_interpolator.mass(0.0)])

    filter = ekf.Ekf(dynamics.f, Q, X_initial, U_initial, P_initial, 3, 2)

    def calculate_apogee(X_predict, t):
        x0 = X_predict
        x_max = dynamics.integrate(x0, t, 13, dt=0.1)
        return x_max

    def apogee_default(X_predict, t):
        return 700.0

    def loop(carry, x):
        previous_t, filter = carry
        t, Z, vel = x
        dt = t - previous_t
        U = jnp.array([mt_interpolator.thrust(t), mt_interpolator.mass(t)])

        filter = ekf.update_nonlinear(ekf=filter, z=Z, h=dynamics.h, R=R * jnp.where(dt > 0.0, dt, 0.01), U=U, dt=dt, stable=True)
        X_predict = filter.X

        apogee_estimate = jax.lax.cond(jnp.logical_and(t > 10.0, t < 13.0), calculate_apogee, apogee_default, X_predict, t)

        return (t, filter), (t, X_predict, Z, U, dynamics.a(X_predict, U), apogee_estimate, vel, filter.X_predicted, filter.P, filter.P_predicted, U)

    (_, filter), series_values = jax.lax.scan(f=loop, init=(0.0, filter), xs=(data_interpolator.times, jnp.stack([data_interpolator.altitude_curve, data_interpolator.acceleration_curve], axis=1), data_interpolator.velocity_curve))
    smoother = rts_smoother.Rts(filter.F_c, filter.num_states)
    X_rts,_ = smoother.run(data_interpolator.times, series_values[1], series_values[7], series_values[8], series_values[9], series_values[10], filter.Q_c, stable=True)
    def loop_predict(t, X_predict):
        apogee_estimate = jax.lax.cond(jnp.logical_and(t > 10.0, t < 13.0), calculate_apogee, apogee_default, X_predict, t)

        return apogee_estimate

    apogee_estimates_rts = jax.vmap(loop_predict, (0, 0), 0)(data_interpolator.times[:-1], X_rts)

    time = series_values[0]
    X = series_values[1]
    Z = series_values[2]
    thrust = series_values[3][:, 0]
    mass = series_values[3][:, 1]

    altitude = Z[:, 0]
    velocity = series_values[6] # todo: fix this
    acceleration = Z[:, 1]
    altitude_est = X[:, 0]
    altitude_est_rts = X_rts[:, 0]
    velocity_est = X[:, 1]
    velocity_est_rts = X_rts[:, 1]
    acceleration_est = series_values[4]
    external_force = X[:, 2]
    external_force_rts = X_rts[:, 2]
    apogee_estimates = series_values[5]

    plot(time, thrust, mass, altitude, velocity, acceleration, altitude_est, velocity_est, acceleration_est, external_force, apogee_estimates, altitude_est_rts, velocity_est_rts, external_force_rts, apogee_estimates_rts)
    
if __name__ == "__main__":
    main()
