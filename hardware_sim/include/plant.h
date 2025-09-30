#ifndef PLANT_H_
#define PLANT_H_

#include "arm_math.h"
#include "matrix.h"

class RocketDynamics {
public:
  struct DynamicsParams {
    float32_t initial_mass;
    float32_t propellant_mass;
    float32_t burn_time;

    float32_t cD;
    float32_t A;
    float32_t rho;

    float32_t total_impulse;
    float32_t initial_velocity;
  };
  enum States {
    Y = 0,
    V = 1,
    A = 2,
  };

  RocketDynamics(DynamicsParams params);

  const Matrix<3, 1> Dynamics(const Matrix<3, 1> &X, float32_t t,
                              float32_t dt) const;

private:
  typedef std::function<float32_t(float32_t, float32_t)> AccelFunction;

  float32_t mass(float32_t t) const;
  float32_t thrust(float32_t t) const;

  const Matrix<2, 1> rk4_step(float32_t t, float32_t dt, const Matrix<2, 1> &X,
                              AccelFunction accel) const;

  DynamicsParams params_;
};

class RocketPlant {
public:
  RocketPlant(RocketDynamics::DynamicsParams params);

  void Update(float32_t dt);

  float32_t accel() const;

  float32_t position() const;

  const Matrix<3, 1> state() const {
    const Matrix<3, 1> X{y_, v_, a_};
    return X;
  }

  const RocketDynamics &dynamics() const { return dynamics_; }

private:
  RocketDynamics dynamics_;

  float32_t a_;
  float32_t v_;
  float32_t y_;

  float32_t t_;
};

#endif // PLANT_H_
