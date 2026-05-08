from __future__ import annotations

from time import monotonic

import cv2

from detector_adapter import Detection
from kalman_filter import KalmanFilter2D


class SiyiAutoTracker:
    def __init__(
        self,
        camera,
        *,
        yaw_position_gain: float = 18.0,
        pitch_position_gain: float = 18.0,
        yaw_velocity_gain: float = 75.0,
        pitch_velocity_gain: float = 75.0,
        position_deadzone_x: float = 0.06,
        position_deadzone_y: float = 0.06,
        velocity_deadzone_px_s: float = 20.0,
        min_speed: int = 5,
        max_speed: int = 20,
        command_interval: float = 0.08,
        lost_timeout: float = 0.30,
        process_noise: float = 50.0,
        measurement_noise: float = 25.0,
    ) -> None:
        self.camera = camera
        self.yaw_position_gain = yaw_position_gain
        self.pitch_position_gain = pitch_position_gain
        self.yaw_velocity_gain = yaw_velocity_gain
        self.pitch_velocity_gain = pitch_velocity_gain
        self.position_deadzone_x = position_deadzone_x
        self.position_deadzone_y = position_deadzone_y
        self.velocity_deadzone_px_s = velocity_deadzone_px_s
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.command_interval = command_interval
        self.lost_timeout = lost_timeout

        self.filter = KalmanFilter2D(
            process_noise=process_noise,
            measurement_noise=measurement_noise,
        )
        self.estimated_center: tuple[float, float] | None = None
        self.estimated_velocity: tuple[float, float] | None = None
        self.last_command = (0, 0)
        self._last_filter_at = 0.0
        self._last_command_at = 0.0
        self._last_seen_at = 0.0
        self._moving = False

    def update(
        self,
        frame_shape: tuple[int, ...] | None,
        detection: Detection | None,
    ) -> tuple[int, int]:
        now = monotonic()

        if frame_shape is None:
            if detection is not None:
                raise ValueError("frame_shape is required when detection is available")
            self._handle_lost_detection(now)
            return self.last_command

        if detection is None:
            if not self._handle_lost_detection(now):
                return self.last_command
        else:
            self._last_seen_at = now
            self._update_filter(now, detection.center)

        if self.estimated_center is None or self.estimated_velocity is None:
            return self.last_command

        if now - self._last_command_at < self.command_interval:
            return self.last_command

        frame_height, frame_width = frame_shape[:2]
        frame_cx = frame_width / 2.0
        frame_cy = frame_height / 2.0
        estimated_cx, estimated_cy = self.estimated_center
        vx, vy = self.estimated_velocity

        error_x = (estimated_cx - frame_cx) / frame_cx
        error_y = (estimated_cy - frame_cy) / frame_cy

        yaw_speed = self._combined_speed(
            self._position_component(
                error_x,
                self.position_deadzone_x,
                self.yaw_position_gain,
            ),
            self._velocity_component(vx, frame_width, self.yaw_velocity_gain),
        )
        pitch_speed = self._combined_speed(
            self._position_component(
                -error_y,
                self.position_deadzone_y,
                self.pitch_position_gain,
            ),
            self._velocity_component(-vy, frame_height, self.pitch_velocity_gain),
        )

        if yaw_speed == 0 and pitch_speed == 0:
            if self._moving:
                self.stop()
            return self.last_command

        self.camera.requestGimbalSpeed(yaw_speed, pitch_speed)
        self.last_command = (yaw_speed, pitch_speed)
        self._last_command_at = now
        self._moving = True
        return self.last_command

    def _update_filter(self, now: float, measured_center: tuple[float, float]) -> None:
        if not self.filter.initialized:
            self.filter.initialize(*measured_center)
            self._last_filter_at = now
            self.estimated_center = self.filter.position
            self.estimated_velocity = self.filter.velocity
            return

        dt = now - self._last_filter_at
        self.filter.predict(dt)
        self.filter.update(measured_center)
        self._last_filter_at = now
        self.estimated_center = self.filter.position
        self.estimated_velocity = self.filter.velocity

    def _handle_lost_detection(self, now: float) -> bool:
        if (
            self.filter.initialized
            and self._last_seen_at > 0.0
            and now - self._last_seen_at < self.lost_timeout
        ):
            dt = now - self._last_filter_at
            self.filter.predict(dt)
            self._last_filter_at = now
            self.estimated_center = self.filter.position
            self.estimated_velocity = self.filter.velocity
            return True

        self.filter.reset()
        self.estimated_center = None
        self.estimated_velocity = None
        if (
            self._moving
            and now - self._last_command_at >= self.command_interval
        ):
            self.stop()
        return False

    def stop(self) -> None:
        self.camera.requestGimbalSpeed(0, 0)
        self.last_command = (0, 0)
        self._last_command_at = monotonic()
        self._moving = False

    def _position_component(self, error: float, deadzone: float, gain: float) -> float:
        magnitude = abs(error)
        if magnitude <= deadzone:
            return 0.0

        usable_error = (magnitude - deadzone) / (1.0 - deadzone)
        speed = usable_error * gain
        return speed if error > 0 else -speed

    def _velocity_component(
        self,
        velocity_px_s: float,
        axis_size: int,
        gain: float,
    ) -> float:
        if abs(velocity_px_s) <= self.velocity_deadzone_px_s:
            return 0.0

        half_axis = max(1.0, axis_size / 2.0)
        normalized_velocity = abs(velocity_px_s) / half_axis
        speed = normalized_velocity * gain
        return speed if velocity_px_s > 0 else -speed

    def _combined_speed(self, *components: float) -> int:
        speed_value = sum(components)
        if speed_value == 0:
            return 0

        speed = int(abs(speed_value))
        speed = min(self.max_speed, max(self.min_speed, speed))
        return speed if speed_value > 0 else -speed


def draw_tracking_overlay(
    frame,
    detection: Detection | None,
    *,
    command: tuple[int, int] | None = None,
    estimated_center: tuple[float, float] | None = None,
    estimated_velocity: tuple[float, float] | None = None,
):
    display = frame.copy()

    if detection is not None:
        cv2.rectangle(
            display,
            (detection.x1, detection.y1),
            (detection.x2, detection.y2),
            (0, 0, 255),
            2,
        )
        label = f"{detection.label} {detection.confidence:.2f}"
        cv2.putText(
            display,
            label,
            (detection.x1, max(24, detection.y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    if estimated_center is not None:
        center = (int(round(estimated_center[0])), int(round(estimated_center[1])))
        cv2.circle(display, center, 5, (0, 255, 255), -1)

        if estimated_velocity is not None:
            velocity_scale = 0.15
            end = (
                int(round(estimated_center[0] + estimated_velocity[0] * velocity_scale)),
                int(round(estimated_center[1] + estimated_velocity[1] * velocity_scale)),
            )
            cv2.arrowedLine(
                display,
                center,
                end,
                (0, 255, 255),
                2,
                tipLength=0.25,
            )

    if command is not None:
        cv2.putText(
            display,
            f"kalman yaw_speed={command[0]} pitch_speed={command[1]}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return display
