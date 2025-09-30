#include "plant.h"
#include <Arduino.h>

RocketDynamics::RocketDynamics(DynamicsParams params) : params_(params) {}

const Matrix<3, 1> RocketDynamics::Dynamics(const Matrix<3, 1> &X, float32_t t,
                                            float32_t dt) const {
  AccelFunction accel = [&](float32_t t, float32_t v) -> float32_t {
    const float32_t rocket_mass = mass(t);
    const float32_t F_thrust = thrust(t);

    const int sign = v < 0 ? -1 : 1;
    const float32_t F_drag =
        -0.5 * params_.cD * params_.A * params_.rho * v * abs(v);
    const float32_t F_weight = rocket_mass * 9.81;
    const float32_t F_net = F_thrust - F_drag - F_weight;

    return F_net / rocket_mass;
  };

  const Matrix<2, 1> X_mini{X.at(States::Y, 0), X.at(States::V, 0)};

  const Matrix<2, 1> X_mini_out = rk4_step(t, dt, X_mini, accel);

  const Matrix<3, 1> X_t{X_mini_out.at(States::Y, 0),
                         X_mini_out.at(States::V, 0),
                         accel(t + dt, X_mini_out.at(States::V, 0))};

  return X_t;
}

const Matrix<2, 1> RocketDynamics::rk4_step(float32_t t, float32_t dt,
                                            const Matrix<2, 1> &X,
                                            AccelFunction accel) const {
  const float32_t v0 = X.at(States::V, 0);
  const float32_t y0 = X.at(States::Y, 0);

  const float32_t v1 = v0;
  const float32_t a1 = accel(t, v1);

  const float32_t v2 = v0 + (a1 * dt / 2.0);
  const float32_t a2 = accel(t + (dt / 2.0), v2);

  const float32_t v3 = v0 + (a2 * dt / 2.0);
  const float32_t a3 = accel(t + (dt / 2.0), v3);

  const float32_t v4 = v0 + (a3 * dt);
  const float32_t a4 = accel(t + dt, v4);

  const Matrix<2, 1> output({y0 + (dt / 6.0) * (v1 + 2 * v2 + 2 * v3 + v4),
                             v0 + (dt / 6.0) * (a1 + 2 * a2 + 2 * a3 + a4)});

  return output;
}

float32_t RocketDynamics::mass(float32_t t) const {
  if (t <= params_.burn_time) {
    float32_t propellant_consumed =
        (t / params_.burn_time) * params_.propellant_mass;

    return params_.initial_mass - propellant_consumed;
  } else {
    return params_.initial_mass - params_.propellant_mass;
  }
}

float32_t RocketDynamics::thrust(float32_t t) const {
  if (t > params_.burn_time) {
    return 0.0;
  }

  const float32_t burn_fraction = t / params_.burn_time;

  const float32_t average_thrust = params_.total_impulse / params_.burn_time;

  if (burn_fraction < 0.1) {
    return average_thrust * 1.5;
  } else if (burn_fraction < 0.8) {
    return average_thrust;
  } else {
    const float32_t tail_fraction = (1.0 - burn_fraction) / 0.2;
    return average_thrust * tail_fraction;
  }
}

RocketPlant::RocketPlant(RocketDynamics::DynamicsParams params)
    : dynamics_(params), a_(0), v_(0), y_(0), t_(0) {}

void RocketPlant::Update(float32_t dt) {
  const Matrix<3, 1> X{y_, v_, a_};

  const Matrix<3, 1> X_t = dynamics_.Dynamics(X, t_, dt);

  t_ += dt;

  a_ = X_t.at(RocketDynamics::States::A, 0);
  v_ = X_t.at(RocketDynamics::States::V, 0);
  y_ = X_t.at(RocketDynamics::States::Y, 0);
}

float32_t RocketPlant::accel() const { return a_ + random(-3, 3); }

float32_t RocketPlant::position() const { return y_ + random(-10, 10); }
