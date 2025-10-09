import pandas as pd
import jax.numpy as jnp

class MassThrustInterpolator:
    def __init__(self, filename) -> None:
        df = pd.read_csv(filename, skiprows=1, header=None, names=['time', 'mass', 'thrust'], comment='#')

        # Convert all columns to numeric, invalid values become NaN
        df = df.apply(pd.to_numeric, errors="coerce")

        self.mass_curve = jnp.array(df['mass'].values)
        self.thrust_curve = jnp.array(df['thrust'].values)
        self.times = jnp.array(df['time'].values)

    def mass(self, t: float):
        return jnp.interp((t + 0.1), self.times, self.mass_curve, left=jnp.nan, right=jnp.nan) / 1000.0

    def thrust(self, t: float):
        return jnp.interp((t + 0.1), self.times, self.thrust_curve, left=jnp.nan, right=jnp.nan)

class FlightDataInterpolator:
    def __init__(self, filename) -> None:
        df = pd.read_csv(filename)
        
        self.altitude_curve = jnp.array(df["altitude"].values)
        self.velocity_curve = jnp.array(df['speed'].values)
        self.acceleration_curve = jnp.array(df['acceleration'].values)

        self.times = jnp.array(df['time'].values)

    def altitude(self, t):
        return jnp.interp(t, self.times, self.altitude_curve, left=jnp.nan, right=jnp.nan)

    def velocity(self, t):
        return jnp.interp(t, self.times, self.velocity_curve, left=jnp.nan, right=jnp.nan)

    def acceleration(self, t):
        return jnp.interp(t, self.times, self.velocity_curve, left=jnp.nan, right=jnp.nan)

        
