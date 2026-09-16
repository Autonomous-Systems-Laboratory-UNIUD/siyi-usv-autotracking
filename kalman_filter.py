import numpy as np


class KalmanFilter2D:
    '''
    2D Kalman Filter for estimating an object's position and velocity.

    The filter maintains a state vector:
        [x, y, vx, vy]

    where:
        x, y   = object position
        vx, vy = object velocity

    The input measurements contain only the object's position [x, y]. Velocity is estimated indirectly by observing how the position changes over time.

    The filter uses a constant-velocity motion model:
        x(t+dt)  = x(t)  + vx(t) * dt
        y(t+dt)  = y(t)  + vy(t) * dt
        vx(t+dt) = vx(t)
        vy(t+dt) = vy(t)

    At each iteration, two steps are performed:
        1. Predict:
        Predicts the new position and velocity based on the previous state and the elapsed time (dt).

        2. Update:
        Compares the prediction with the new measurement [x, y] and corrects the estimated state based on the uncertainty of the prediction and the measurement noise.

    Main parameters:
        dt:
            Time interval between two updates.

        process_noise:
            Uncertainty of the motion model. Higher values make the filter more responsive to sudden changes in motion.

        measurement_noise:
            Uncertainty of the measurements. Higher values make the filter rely more on the prediction than on the measurements.

        initial_velocity_uncertainty:
            Initial uncertainty of the estimated velocity.

    The filter is suitable for tracking objects using cameras, sensors, or other systems that provide noisy 2D position measurements.
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
 
        Returns (accepted, nis). The measurement is rejected and the state left
        untouched when the normalized innovation squared exceeds gate_threshold.
        """
        z = np.asarray(z, dtype=np.float64).reshape(2, 1)
 
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        S_inv = np.linalg.inv(S)
 
        nis = float(y.T @ S_inv @ y)
        if nis > self.gate_threshold:
            return False, nis
 
        K = self.P @ self.H.T @ S_inv
 
        self.x = self.x + K @ y
        self.P = (self.I - K @ self.H) @ self.P
 
        return True, nis
