import numpy as np
import math

# TODO: You could also model accelerometer accuracy by rounding up/down to certain +-g numbers
# This is just 1d because i think its okay to assume that accel_y will always be in the direction of -g
class AccelerometerPlant:
    def __init__(self, noise_std_dev, iteration_latency):
        self.noise_std_dev = noise_std_dev
        self.iteration_latency = iteration_latency
        self.actual_acceleration = 0
    
    def apply_force(self, net_force, mass):
        self.actual_acceleration = net_force / mass
    
    def get_acceleration(self):
        return np.random.normal(0.0, self.noise_std_dev, size=1)[0] + self.actual_acceleration

# This is just going to give back altitude instead of pressure which needs to be converted because I'm assuming the sensor is doing this for us, if not its not a crazy thing to be adding
class BarometerPlant:
    def __init__(self, noise_std_dev, iteration_latency):
        self.noise_std_dev = noise_std_dev
        self.iteration_latency = iteration_latency
        self.actual_altitude = 0
    
    def apply_accel(self, accel, velocity, dt):
        self.actual_altitude += velocity * dt + 0.5 * accel * dt**2
    
    def get_altitude(self):
        return np.random.normal(0.0, self.noise_std_dev, size=1)[0] + self.actual_altitude

class RocketPlant:
    def __init__(self, initial_mass, cD, A, rho, noise_std_dev, iteration_latency, initial_velocity):
        self.barometer_plant = BarometerPlant(noise_std_dev, iteration_latency)
        self.accelerometer_plant = AccelerometerPlant(noise_std_dev, iteration_latency)
        
        self.initial_mass = initial_mass
        self.propellant_mass = 8.0
        self.cD = cD
        self.A = A
        self.rho = rho
        
        self.total_impulse = 5000
        self.burn_time = 8.0
        self.average_thrust = self.total_impulse / self.burn_time
        self.max_thrust = self.average_thrust * 1.5
        
        # State variables
        self.noise_std_dev = noise_std_dev
        self.iteration_latency = iteration_latency
        self.initial_velocity = initial_velocity
        self.a = 0
        self.v = initial_velocity
        self.y = 0
        self.t = 0
   
    # These mass and thrust functions + constants are taken from claude so I can see if my idea works at all, these will need to be changed
    def __mass(self):
        if self.t <= self.burn_time:
            consumed_propellant = (self.t / self.burn_time) * self.propellant_mass
            return self.initial_mass - consumed_propellant
        else:
            return self.initial_mass - self.propellant_mass
    
    def __thrust(self):
        if self.t > self.burn_time:
            return 0.0
        
        burn_fraction = self.t / self.burn_time
        
        if burn_fraction < 0.1:
            return self.average_thrust * 1.5
        elif burn_fraction < 0.8:
            return self.average_thrust
        else:
            tail_fraction = (1.0 - burn_fraction) / 0.2
            return self.average_thrust * tail_fraction
    
    def update(self, dt):
        self.t += dt
        
        F_thrust = self.__thrust()
        
        drag_direction = -1 if self.v >= 0 else 1
        F_drag = drag_direction * 0.5 * self.cD * self.A * self.rho * self.v**2 
        
        F_g = self.__mass() * 9.81
        F_net = F_thrust - F_drag - F_g
        
        self.a = F_net / self.__mass()
        self.v += self.a * dt
        self.y += self.v * dt + 0.5 * self.a * dt**2
        
        self.barometer_plant.apply_accel(self.a, self.v, dt)
        self.accelerometer_plant.apply_force(F_net, self.__mass())
