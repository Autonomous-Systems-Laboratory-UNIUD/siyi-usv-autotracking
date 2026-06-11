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

        self.x = np.zeros((4, 1), dtype=np.float64)
        self.P = self._initial_covariance()

        self.F = np.eye(4, dtype=np.float64)
        self.H = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]],
            dtype=np.float64
        )

        self.R = np.eye(2, dtype=np.float64) * self.measurement_noise
        self.I = np.eye(4, dtype=np.float64)

        self.Q = np.eye(4, dtype=np.float64)

        self._set_dt(self.dt)

        self.initialized = False

    def initialize(self, x, y, vx=0.0, vy=0.0):
        self.x = np.array([[x], [y], [vx], [vy]], dtype=np.float64)
        self.P = self._initial_covariance()
        self.initialized = True

    def reset(self):
        self.x = np.zeros((4, 1), dtype=np.float64)
        self.P = self._initial_covariance()
        self.initialized = False

    @property
    def position(self):
        return tuple(self.x[:2, 0])

    @property
    def velocity(self):
        return tuple(self.x[2:, 0])

    def _initial_covariance(self):
        return np.diag([
            self.measurement_noise,
            self.measurement_noise,
            self.initial_velocity_uncertainty,
            self.initial_velocity_uncertainty,
        ]).astype(np.float64)

    def _set_dt(self, dt):
        self.dt = max(1e-3, float(dt))

        dt2 = self.dt ** 2
        dt3 = dt2 * self.dt
        dt4 = dt2 ** 2

        self.F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)

        self.Q = self.process_noise * np.array([
            [dt4/4, 0, dt3/2, 0],
            [0, dt4/4, 0, dt3/2],
            [dt3/2, 0, dt2, 0],
            [0, dt3/2, 0, dt2],
        ], dtype=np.float64)

    def predict(self, dt=None):
        if dt is not None:
            self._set_dt(dt)

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.x.ravel()

    def update(self, z):
        z = np.asarray(z, dtype=np.float64).reshape(2, 1)

        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (self.I - K @ self.H) @ self.P

        return self.x.ravel()