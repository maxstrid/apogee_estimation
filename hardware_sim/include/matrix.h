#ifndef MATRIX_H_
#define MATRIX_H_

#include "arm_math.h"
#include <Arduino.h>
template <int R, int C> class Matrix;
// Wrapper around a f32 cmsis dsp matrix object
// TODO(max): Chain optimization is required
template <int Rows, int Cols> class Matrix {
public:
  static constexpr size_t kDataSize = Rows * Cols;

  template <int R, int C> friend class Matrix;

  Matrix(float32_t data[kDataSize])
      : matrix_instance_{Rows, Cols, matrix_data_} {
    std::copy(data, data + kDataSize, matrix_data_);
  }

  Matrix(std::initializer_list<float32_t> values)
      : matrix_instance_{Rows, Cols, matrix_data_} {
    if (values.size() != kDataSize) {
      // You can assert, throw, or fill with zeros
      // For embedded, I'd suggest assert or just ignore extra
    }
    std::copy(values.begin(), values.end(), matrix_data_);
  }

  Matrix(const Matrix &other) : matrix_instance_{Rows, Cols, matrix_data_} {
    std::copy(other.matrix_data_, other.matrix_data_ + kDataSize, matrix_data_);
  }

  Matrix &operator=(const Matrix &other) {
    std::copy(other.matrix_data_, other.matrix_data_ + kDataSize, matrix_data_);

    matrix_instance_ = {Rows, Cols, matrix_data_};
    return *this;
  }

  static Matrix Zero() {
    Matrix matrix{};
    std::fill(matrix.matrix_data_, matrix.matrix_data_ + kDataSize, 0.0f);
    matrix.matrix_instance_ = {Rows, Cols, matrix.matrix_data_};
    return matrix;
  }

  static Matrix Identity() {
    static_assert(Rows == Cols, "Identity matrix must be square");

    Matrix matrix = Zero();

    for (int i = 0; i < Rows; i++) {
      matrix.set(i, i, 1.0f);
    }

    return matrix;
  }

  Matrix operator+(const Matrix<Rows, Cols> &rhs) const {
    Matrix result;

    arm_mat_add_f32(&matrix_instance_, &rhs.matrix_instance_,
                    &result.matrix_instance_);

    return result;
  }

  Matrix operator-(const Matrix<Rows, Cols> &rhs) const {
    Matrix result;

    const Matrix<Rows, Cols> rhs_fixed = rhs * -1.0f;

    arm_mat_add_f32(&matrix_instance_, &rhs_fixed.matrix_instance_,
                    &result.matrix_instance_);

    return result;
  }

  template <int RhsCols>
  Matrix<Rows, RhsCols> operator*(const Matrix<Cols, RhsCols> &rhs) const {
    Matrix<Rows, RhsCols> result;

    arm_mat_mult_f32(&matrix_instance_, &rhs.matrix_instance_,
                     &result.matrix_instance_);

    return result;
  }

  Matrix<Rows, Cols> operator*(const float32_t scalar) const {
    Matrix<Rows, Cols> out;

    for (int i = 0; i < Rows; i++) {
      for (int j = 0; j < Rows; j++) {
        out.set(i, j, at(i, j) * scalar);
      }
    }

    return out;
  }

  Matrix<Cols, Rows> transpose() const {
    Matrix<Cols, Rows> result;

    arm_mat_trans_f32(&matrix_instance_, &result.matrix_instance_);

    return result;
  }

  Matrix inverse() const {
    Matrix result;

    arm_mat_inverse_f32(&matrix_instance_, &result.matrix_instance_);

    return result;
  }

  void set(int i, int j, float32_t data) { matrix_data_[i * Cols + j] = data; }

  const float32_t at(int i, int j) const { return matrix_data_[i * Cols + j]; }

  void print() const {
    for (int i = 0; i < Rows; i++) {
      for (int j = 0; j < Cols; j++) {
        Serial.print(at(i, j));
        if (j != (Cols - 1)) {
          Serial.print(" ,");
        }
      }
      Serial.println();
    }
  }

protected:
  Matrix() : matrix_instance_{Rows, Cols, matrix_data_} {}
  arm_matrix_instance_f32 matrix_instance_;

  // We want to store the data here so it doesnt go out scope
  float32_t matrix_data_[kDataSize];
};

#endif // MATRIX_H_
