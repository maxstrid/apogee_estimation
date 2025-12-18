import jax
import jax.numpy as jnp

import optax

class MPCSolver:
    def __init__(self, dynamics, learning_rate=0.001):
        self.dynamics = dynamics
        self.optimizer = optax.adam(learning_rate=learning_rate)

    @jax.jit(static_argnums=(0,))
    def __step(self, u_seq, opt_state, x0, Q, R, goal_apogee):
        def loss_fn(u):
            return self.dynamics.mpc_cost_fn(Q, R, x0, u, goal_apogee)
        
        cost, grads = jax.value_and_grad(loss_fn)(u_seq)

        updates, opt_state = self.optimizer.update(grads, opt_state)

        new_u_seq = optax.apply_updates(u_seq, updates)

        cd_min, cd_max = 0.0, 0.445,
        area_min, area_max = 0.0, 0.00948
        
        new_u_seq = new_u_seq.at[:, 0].set(jnp.clip(new_u_seq[:, 0], cd_min, cd_max))
        new_u_seq = new_u_seq.at[:, 1].set(jnp.clip(new_u_seq[:, 1], area_min, area_max))
        
        return new_u_seq, opt_state, cost

    def solve(self, x0, initial_u_guess, Q, R, goal_apogee, steps=300):
        opt_state = self.optimizer.init(initial_u_guess)
        u_seq = initial_u_guess
        
        step_fn = self.__step
        
        print(f"Starting MPC Optimization for Goal Apogee: {goal_apogee}m")
        
        for i in range(steps):
            u_seq, opt_state, cost = step_fn(u_seq, opt_state, x0, Q, R, goal_apogee)
            
            if i % 20 == 0:
                print(f"Iter {i}: Cost = {cost}")
                
        return u_seq
