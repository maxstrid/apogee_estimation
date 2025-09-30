import kf
import numpy as np
import matplotlib.pyplot as plt
from plant import RocketPlant
from scipy.integrate import solve_ivp

# x = [x, v]

def basic_kf_test():
    X_initial = np.matrix('0 ; 3')

    A = np.matrix('0 1; 0 0')

    Q = np.matrix('1 0; 0 1000')

    R = np.matrix('10')

    P_initial = np.matrix(f'{1e-5} 0; 0 1')

    # 5ms
    dt = 0.005

    filter = kf.Kf(A, Q, X_initial, P_initial, dt)

    noise = np.random.normal(0,1, int(10 * (1 / dt)))

    positions = []
    velocities = []
    times = []
    measured_velocities = []

    for i in range(0, int(10 * (1 / dt))):
        velocity = 5.0
        velocity += noise[i]

        z = np.matrix(f'{velocity}')

        H = np.matrix('0 1')

        filter.update(z, H, R) 

        positions.append(float(filter.X[0, 0]))
        velocities.append(float(filter.X[1, 0]))
        times.append(i * dt)
        measured_velocities.append(velocity)
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Position plot
    ax1.plot(times, positions, 'b-', linewidth=2, label='Estimated Position')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Position (m)')
    ax1.set_title('Kalman Filter: Position Estimate')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Velocity plot
    ax2.plot(times, velocities, 'r-', linewidth=2, label='Estimated Velocity')
    ax2.plot(times, measured_velocities, 'g.', alpha=0.3, markersize=1, label='Measured Velocity (noisy)')
    ax2.axhline(y=5.0, color='k', linestyle='--', alpha=0.7, label='True Velocity (5 m/s)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_title('Kalman Filter: Velocity Estimate vs Measurements')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()


def rocket_model_test():
    rocket_plant = RocketPlant(25.0, 0.45, 0.0113, 1.225, 5.0, 0, 0.0)
    # x = [y, v, a, y_apogee]
    dt = 0.005
    A = np.matrix('0.02 1 0 0 ; 0 0 1 0 ; 0 0 0.0 0 ; 0 0 0 0')
    Q = np.diag([1e-6, 1e-4, 1e-1, 1e-4])
    P_initial = np.matrix(np.diag([1e-2, 1.0, 10.0, 10.0]))
    X_initial = np.matrix('0 ; 0 ; 0 ; 100')
    filter = kf.Kf(A, Q, X_initial, P_initial, dt)
    
    # Rocket dynamics function for apogee estimation
    
    def rocket_dynamics_ivp(tau, y, plant, current_abs_time):
        pos, vel = y
        
        # tau is time forward from current moment
        # current_abs_time is the absolute time when prediction starts
        absolute_time = current_abs_time + tau
        
        temp = RocketPlant(plant.initial_mass, plant.cD, plant.A,
                           plant.rho, plant.noise_std_dev, plant.iteration_latency,
                           0)  # Always use 0 for initial velocity
        temp.v = vel  # Set current velocity directly
        temp.y = pos
        temp.t = absolute_time
        
        F_thrust = temp._RocketPlant__thrust()
        
        drag_direction = -1 if vel >= 0 else 1
        F_drag = drag_direction * 0.5 * temp.cD * temp.A * temp.rho * vel**2
        
        F_g = temp._RocketPlant__mass() * 9.81
        
        F_net = F_thrust - F_drag - F_g
        acc = F_net / temp._RocketPlant__mass()
        
        return [vel, acc]
    
    def estimate_apogee(current_time, current_position, current_velocity):
        """
        Estimate apogee using scipy solve_ivp
        """
        # Initial conditions: [position, velocity]
        y0 = [current_position, current_velocity]
        
        # Create event function to detect apogee (when velocity = 0)
        def apogee_event(t, y):
            return y[1]  # velocity component
        apogee_event.terminal = True
        apogee_event.direction = -1
        
        # Time span - integrate up to 100 seconds max
        t_span = (current_time, current_time + 100)
        
        try:
            # Solve the ODE
            sol = solve_ivp(
                fun=lambda t, y: rocket_dynamics_ivp(t, y, rocket_plant, current_time),
                t_span=t_span,
                y0=y0,
                events=apogee_event,
                dense_output=True,
                rtol=1e-6,
                method='RK45'
            )
            
            if sol.t_events[0].size > 0:
                # Apogee detected
                apogee_altitude = sol.y_events[0][0][0]  # position at apogee
                return apogee_altitude
            else:
                # No apogee detected, return final position
                return sol.y[0][-1]
        except:
            # Integration failed, return current position as fallback
            return current_position
    
    # Data storage for plotting
    time_data = []
    actual_accel = []
    actual_velocity = []
    actual_position = []
    
    estimated_accel = []
    estimated_velocity = []
    estimated_position = []
    estimated_apogees = []  # New array for apogee estimates
    
    # Separate arrays for measurements with their timestamps
    accel_measurements = []
    accel_times = []
    position_measurements = []
    position_times = []
    
    for i in range(0, int(50 * (1 / dt))):
        rocket_plant.update(dt)
        current_time = i * dt
        time_data.append(current_time)
        actual_accel.append(rocket_plant.a)
        actual_velocity.append(rocket_plant.v)
        actual_position.append(rocket_plant.y)
        
        if (i % 2 == 0):
            accel_measurement = rocket_plant.accelerometer_plant.get_acceleration()
            z = np.matrix(f'{rocket_plant.accelerometer_plant.get_acceleration()}')
            R = np.matrix('0.001')
            H = np.matrix('0 0 1 0')
            filter.update(z, H, R)
            accel_measurements.append(accel_measurement)
            accel_times.append(current_time)
        else:
            alt_measurement = rocket_plant.barometer_plant.get_altitude()
            z = np.matrix(f'{rocket_plant.barometer_plant.get_altitude()}')
            R = np.matrix('0.25')
            H = np.matrix('1 0 0 0')
            filter.update(z, H, R)
            position_measurements.append(alt_measurement)
            position_times.append(current_time)
            
        # Store estimated values
        state = filter.X
        estimated_position.append(float(state[0, 0]))
        estimated_velocity.append(float(state[1, 0]))
        estimated_accel.append(float(state[2, 0]))
        
        # Estimate apogee from current KF state
        est_pos = float(state[0, 0])
        est_vel = float(state[1, 0])
        
        if est_vel > 0 and current_time < 40:  # Only estimate if ascending and reasonable time
            apogee_est = estimate_apogee(current_time, rocket_plant.y, rocket_plant.v)
            estimated_apogees.append(apogee_est)
        else:
            estimated_apogees.append(None)  # No estimate when descending
        
    # Get actual apogee for comparison
    actual_apogee = max(actual_position)
    print(f"Actual apogee: {actual_apogee:.1f} m")
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 12))
    fig.suptitle('Rocket State Estimation with Apogee Prediction', fontsize=16)
    
    # Acceleration plot
    ax1.plot(time_data, actual_accel, 'b-', label='Actual', linewidth=2)
    ax1.scatter(accel_times, accel_measurements, c='red', s=10, alpha=0.6, label='Measured')
    ax1.plot(time_data, estimated_accel, 'g--', label='Estimated', linewidth=1.5)
    ax1.set_ylabel('Acceleration (m/s²)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Acceleration Comparison')
    
    # Velocity plot
    ax2.plot(time_data, actual_velocity, 'b-', label='Actual', linewidth=2)
    ax2.plot(time_data, estimated_velocity, 'g--', label='Estimated', linewidth=1.5)
    ax2.set_ylabel('Velocity (m/s)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Velocity Comparison')
    
    # Position plot
    ax3.plot(time_data, actual_position, 'b-', label='Actual', linewidth=2)
    ax3.scatter(position_times, position_measurements, c='red', s=10, alpha=0.6, label='Measured')
    ax3.plot(time_data, estimated_position, 'g--', label='Estimated', linewidth=1.5)
    ax3.axhline(y=actual_apogee, color='blue', linestyle=':', alpha=0.7, label=f'Actual Apogee ({actual_apogee:.0f}m)')
    ax3.set_ylabel('Position (m)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Position Comparison')
    
    # Apogee estimates plot
    valid_times = [time_data[i] for i in range(len(estimated_apogees)) if estimated_apogees[i] is not None]
    valid_apogees = [est for est in estimated_apogees if est is not None]
    
    ax4.plot(valid_times, valid_apogees, 'purple', linewidth=2, label='Apogee Estimates')
    ax4.axhline(y=actual_apogee, color='blue', linestyle=':', alpha=0.7, label=f'Actual Apogee ({actual_apogee:.0f}m)')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Estimated Apogee (m)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_title('Apogee Estimation Over Time')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    rocket_model_test()
