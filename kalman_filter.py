import numpy as np


class KalmanFilter2D:
    '''
    2D Kalman Filter for estimating an object's position and velocity.

    The filter maintains a state vector:
        [x, y, vx, vy]

    where:
        x, y   = object position (pixels)
        vx, vy = object velocity (pixels / second)

    The input measurements contain only the object's position [x, y]. Velocity is
    estimated indirectly by observing how the position changes over time.

    The filter uses a constant-velocity motion model:
        x(t+dt)  = x(t)  + vx(t) * dt
        y(t+dt)  = y(t)  + vy(t) * dt
        vx(t+dt) = vx(t)
        vy(t+dt) = vy(t)

    At each iteration, two steps are performed:
        1. Predict:
        Predicts the new position and velocity based on the previous state and the
        elapsed time (dt).

        2. Update:
        Compares the prediction with the new measurement [x, y] and corrects the
        estimated state based on the uncertainty of the prediction and the
        measurement noise.

    Main parameters:
        dt:
            Time interval between two updates.

        process_noise:
            Variance of the unmodelled acceleration, in px^2/s^4. Higher values make
            the filter more responsive to sudden changes in motion. This is the
            parameter that sets the group delay of the velocity estimate: too low and
            the estimate trails the true motion by a second or more.

        measurement_noise:
            Variance of the detector's centre estimate, in px^2. Higher values make
            the filter rely more on the prediction than on the measurements.

        initial_velocity_uncertainty:
            Initial variance of the estimated velocity, in (px/s)^2. Keep this near
            the accuracy of whatever seed velocity is passed to initialize(); an
            inflated value produces large spurious velocity estimates in the first
            frames after initialization.

        max_dt:
            Upper clamp on the prediction interval. Protects the covariance from
            blowing up if the loop stalls or a frame takes unusually long.

        gate_threshold:
            Chi-square threshold on the normalized innovation squared (2 dof).
            13.82 corresponds to 99.9%. Measurements beyond this are rejected as
            outliers rather than being fused.
    '''

    def __init__(
        self,
        dt=1.0,
        process_noise=5000.0,
        measurement_noise=25.0,
        initial_velocity_uncertainty=2500.0,
        max_dt=0.2,
        gate_threshold=13.82,
    ):
        """
        State: [x, y, vx, vy]
        Measurement: [x, y]
        """

        self.dt = float(dt)
        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)
        self.initial_velocity_uncertainty = float(initial_velocity_uncertainty)
        self.max_dt = float(max_dt)
        self.gate_threshold = float(gate_threshold)

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
        self.last_accepted = True
        self.last_nis = 0.0
        self.consecutive_rejections = 0

    def initialize(self, x, y, vx=0.0, vy=0.0):
        self.x = np.array([[x], [y], [vx], [vy]], dtype=np.float64)
        self.P = self._initial_covariance()
        self.initialized = True
        self.last_accepted = True
        self.last_nis = 0.0
        self.consecutive_rejections = 0

    def reset(self):
        self.x = np.zeros((4, 1), dtype=np.float64)
        self.P = self._initial_covariance()
        self.initialized = False
        self.last_accepted = True
        self.last_nis = 0.0
        self.consecutive_rejections = 0

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
        self.dt = min(self.max_dt, max(1e-3, float(dt)))

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
        """
        Fuse a position measurement.

        Returns the state vector, as before. Whether the measurement was actually
        fused is reported through self.last_accepted, and the normalized innovation
        squared through self.last_nis. A measurement whose NIS exceeds
        gate_threshold is rejected and the state is left untouched.
        """
        z = np.asarray(z, dtype=np.float64).reshape(2, 1)

        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        S_inv = np.linalg.inv(S)

        self.last_nis = (y.T @ S_inv @ y).item()

        if self.last_nis > self.gate_threshold:
            self.last_accepted = False
            self.consecutive_rejections += 1
            return self.x.ravel()

        K = self.P @ self.H.T @ S_inv

        self.x = self.x + K @ y
        self.P = (self.I - K @ self.H) @ self.P

        self.last_accepted = True
        self.consecutive_rejections = 0
        return self.x.ravel()