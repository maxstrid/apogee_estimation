#ifndef KF_H_
#define KF_H_

#include "arm_math.h"
#include "matrix.h"

// Right now im being super lazy by not discretizing Q and A properly. This
// **NEEDS** to be done but is fine to avoid at fast enough update times.
// Right now A and D should be given discrete.
template <int State> class KalmanFilter {
public:
  KalmanFilter(Matrix<State, 1> X_initial, Matrix<State, State> P_initial,
               Matrix<State, State> A, Matrix<State, State> Q, double dt)
      : X_hat_(X_initial), P_(P_initial), A_(A), Q_(Q), dt_(dt) {}

  void Predict() {
    X_hat_ = A_ * X_hat_;

    P_ = A_ * P_ * A_.transpose() + Q_;
  }

  template <int Observations>
  void Update(Matrix<Observations, 1> z, Matrix<Observations, State> H,
              Matrix<Observations, Observations> R) {
    Matrix<Observations, 1> expected = H * X_hat_;

    Matrix<State, Observations> K =
        P_ * H.transpose() * (H * P_ * H.transpose() + R).inverse();

    Matrix<State, 1> update = K * (z - expected);

    P_ = (Matrix<State, State>::Identity() - K * H) * P_;

    X_hat_ = X_hat_ + update;
  }

  const Matrix<State, 1> &X_hat() const { return X_hat_; }

private:
  float32_t dt_;

  Matrix<State, 1> X_hat_;
  Matrix<State, State> P_;
  Matrix<State, State> A_;
  Matrix<State, State> Q_;
};

#endif // KF_H_
