#include <Arduino.h>

#include "arm_math.h"
#include "kf.h"
#include "matrix.h"
#include "plant.h"

#include <memory>

// clang-format off
#include <FreeRTOS.h>
#include "task.h"
#include "queue.h"
// clang-format on

#include <HardwareTimer.h>
constexpr int kStates = 3;

QueueHandle_t logQueue = NULL;
TaskHandle_t MainTaskHandle = NULL;
std::unique_ptr<RocketPlant> rocket_plant;
std::unique_ptr<KalmanFilter<kStates>> kalman_filter;

HardwareTimer Timer2(TIM2);

void MainTask(void *param);
void LogTask(void *param);

void onTimer2() {
  BaseType_t xHigherPriorityTaskWoken = pdFALSE;
  vTaskNotifyGiveFromISR(MainTaskHandle, &xHigherPriorityTaskWoken);
  portYIELD_FROM_ISR(xHigherPriorityTaskWoken);
}

// 200Hz
void setupMainTimer() {
  Timer2.setPrescaleFactor(8400);
  Timer2.setOverflow(50);
  Timer2.attachInterrupt(onTimer2);
  Timer2.resume();
}

const float32_t dt = 0.005;

void setup() {
  Serial.begin(921600);
  while (!Serial) {
    ; // wait until serial is ready
  }

  rocket_plant = std::make_unique<RocketPlant>(RocketDynamics::DynamicsParams{
      .initial_mass = 25.0,
      .propellant_mass = 8.0,
      .burn_time = 8.0,

      .cD = 0.45,
      .A = 0.0113,
      .rho = 1.225,

      .total_impulse = 5000.0,
      .initial_velocity = 0.0,
  });
  Matrix<kStates, kStates> A({
      // clang-format off
      1.00e+00, 5.00e-03, 1.25e-05,
      0.00e+00, 1.00e+00, 5.00e-03,
      0.00e+00, 0.00e+00, 1.00e+00
      // clang-format on
  });

  Matrix<kStates, kStates> Q({
      // clang-format off
      5.00004167e-02, 1.25000781e-04, 2.08333333e-07,
      1.25000781e-04, 5.00004167e-02, 1.25000000e-04,
      2.08333333e-07, 1.25000000e-04, 5.00000000e-02
      // clang-format on
  });

  Matrix<kStates, 1> X_initial({0.0, 0.0, 0.0});

  Matrix<kStates, kStates> P_initial({
      // clang-format off
      1e-2, 0, 0,
      0, 1.0, 0,
      0, 0, 10.0
      // clang-format on
  });

  kalman_filter =
      std::make_unique<KalmanFilter<kStates>>(X_initial, P_initial, A, Q, dt);

  Serial.println("Serial is ready!");

  logQueue = xQueueCreate(16, sizeof(char[256]));
  xTaskCreate(LogTask, "Logger", 256, NULL, 1, NULL);
  xTaskCreate(MainTask, "MainTask", 256, NULL, 100, &MainTaskHandle);
  setupMainTimer();
  vTaskStartScheduler();
}

void logPrint(const char *msg) { xQueueSendToBack(logQueue, msg, 0); }

void loop() {}

int iterations = 0;
float32_t t = 0.0;
void MainTask(void *param) {
  while (1) {
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    unsigned long start = micros();

    if (iterations >= (int)(50 * (1 / dt))) {
      return;
    }

    t += dt;

    rocket_plant->Update(dt);

    kalman_filter->Predict();

    const float32_t altitude = rocket_plant->position();
    const float32_t accel = rocket_plant->accel();
    {

      const Matrix<1, 1> z{altitude};
      const Matrix<1, 3> H{1, 0, 0};
      const Matrix<1, 1> R{1};

      kalman_filter->Update<1>(z, H, R);
    }

    {

      const Matrix<1, 1> z{accel};
      const Matrix<1, 3> H{0, 0, 1};
      const Matrix<1, 1> R{1};

      kalman_filter->Update<1>(z, H, R);
    }

    const Matrix<3, 1> X = rocket_plant->state();

    float32_t apogee_estimate = 0.0;

    const Matrix<3, 1> X_estimated = kalman_filter->X_hat();
    {
      if (t > 15.0) {

        Matrix<3, 1> X_est = X_estimated;
        float32_t t_elapsed = t;
        float32_t dt_predict =
            0.20; // small timestep for prediction, e.g., 200ms
        while (X_est.at(RocketDynamics::States::V, 0) > 0.0f) {
          X_est =
              rocket_plant->dynamics().Dynamics(X_est, t_elapsed, dt_predict);
          t_elapsed += dt_predict;
        }

        if (t_elapsed != t) {
          apogee_estimate = X_est.at(RocketDynamics::States::Y, 0);
        }
      }
    }

    char y_buf[16];
    dtostrf(X.at(RocketDynamics::States::Y, 0), 6, 3, y_buf);
    char v_buf[16];
    dtostrf(X.at(RocketDynamics::States::V, 0), 6, 3, v_buf);
    char a_buf[16];
    dtostrf(X.at(RocketDynamics::States::A, 0), 6, 3, a_buf);
    char t_buf[16];
    dtostrf(t, 6, 3, t_buf);

    char y_measured_buf[16];
    dtostrf(altitude, 6, 3, y_measured_buf);
    char a_measured_buf[16];
    dtostrf(accel, 6, 3, a_measured_buf);

    char y_est_buf[16];
    dtostrf(X_estimated.at(RocketDynamics::States::Y, 0), 6, 3, y_est_buf);

    char v_est_buf[16];
    dtostrf(X_estimated.at(RocketDynamics::States::V, 0), 6, 3, v_est_buf);

    char a_est_buf[16];
    dtostrf(X_estimated.at(RocketDynamics::States::A, 0), 6, 3, a_est_buf);

    char apogee_estimate_buf[16];
    dtostrf(apogee_estimate, 6, 3, apogee_estimate_buf);

    char buffer[256];

    unsigned long duration = micros() - start;
    sprintf(buffer, "%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %d, %ld", t_buf,
            y_buf, y_est_buf, y_measured_buf, v_buf, v_est_buf, a_buf,
            a_est_buf, a_measured_buf, apogee_estimate_buf, iterations,
            duration);

    logPrint(buffer);

    iterations++;
  }
}

void LogTask(void *param) {
  char msg[256];
  for (;;) {
    if (xQueueReceive(logQueue, msg, portMAX_DELAY)) {
      Serial.println(msg); // This can block, but task is low priority
    }
  }
}
