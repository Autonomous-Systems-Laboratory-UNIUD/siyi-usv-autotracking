import numpy as np


class KalmanFilter2D:
    def __init__(
        self,
        dt=1.0,
        process_noise=50.0,
        measurement_noise=25.0,
        initial_velocity_uncertainty=100000.0,
    ):
        """
        State: [x, y, vx, vy]
        Measurement: [x, y]
        """
        self.dt = float(dt)
        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)
        self.initial_velocity_uncertainty = float(initial_velocity_uncertainty)

        # State initial value
        self.x = np.zeros((4, 1))  # [x, y, vx, vy]^T

        # Covariance initial value
        self.P = self._initial_covariance()

        # State transition matrix
        self.F = np.eye(4)

        # Measurement matrix
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=float)

        # Measurement noise
        self.R = np.eye(2) * self.measurement_noise

        # Identity matrix
        self.I = np.eye(4)

        self.Q = np.eye(4)
        self._set_dt(self.dt)
        self.initialized = False

    def initialize(self, x, y, vx=0.0, vy=0.0) -> None:
        self.x = np.array([[x], [y], [vx], [vy]], dtype=float)
        self.P = self._initial_covariance()
        self.initialized = True

    def reset(self):
        self.x = np.zeros((4, 1))
        self.P = self._initial_covariance()
        self.initialized = False

    @property
    def position(self) -> tuple[float, float]:
        return tuple(float(value) for value in self.x[:2, 0])

    @property
    def velocity(self) -> tuple[float, float]:
        return tuple(float(value) for value in self.x[2:, 0])

    def _initial_covariance(self):
        return np.diag([
            self.measurement_noise,
            self.measurement_noise,
            self.initial_velocity_uncertainty,
            self.initial_velocity_uncertainty,
        ])

    def _set_dt(self, dt):
        self.dt = max(1e-3, float(dt))
        dt2 = self.dt * self.dt
        dt3 = dt2 * self.dt
        dt4 = dt2 * dt2

        self.F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=float)

        self.Q = self.process_noise * np.array([
            [dt4 / 4.0, 0, dt3 / 2.0, 0],
            [0, dt4 / 4.0, 0, dt3 / 2.0],
            [dt3 / 2.0, 0, dt2, 0],
            [0, dt3 / 2.0, 0, dt2],
        ], dtype=float)

    def predict(self, dt=None):
        if dt is not None:
            self._set_dt(dt)

        # Prediction of the state
        self.x = self.F @ self.x

        # Prediction of the covariance
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.x[:2].ravel()

    def update(self, z):
        """
        z: np.array([[x], [y]]) or tuple/list (x, y)
        """
        z = np.asarray(z, dtype=float).reshape(2, 1)

        # Innovation
        y = z - (self.H @ self.x)

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y

        # Covariance update
        self.P = (self.I - K @ self.H) @ self.P

        return self.x[:2].ravel()
