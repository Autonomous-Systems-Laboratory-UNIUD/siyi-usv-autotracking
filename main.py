from argparse import ArgumentParser
from collections import deque
import os
import sys
import time
import cv2
import logging
import zenoh
import numpy as np

from zenoh.ext import z_serialize
from draw import draw_tracking_overlay
from detector_adapter import DetectionService
from gstreamer import GStreamerFrameStreamer
from keyboard_manager import TerminalKeyReader
from kalman_filter import KalmanFilter2D
from helpers import serialize_vector3

SDK_DIR = os.path.join(os.path.dirname(__file__), "siyi_sdk")
if SDK_DIR not in sys.path:
    sys.path.insert(0, SDK_DIR)

from siyi_sdk import SIYISDK


BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "usv_detection/models")

# Window of recent detections used to seed the velocity state at initialization.
SEED_WINDOW_S = 0.30


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="yolo26n", help="Model name without extension")
    parser.add_argument("--drone_name", type=str, required=True, choices=["X500B", "X500W"], help="Drone name to be displayed in the opencv window")
    parser.add_argument("--streaming_port", type=str, required=True, help="Port used to receive the streaming from the raspberry")
    parser.add_argument("--min_consecutive_detections", type=int, required=True, help="Minimun number of consecutive detections (frames) to be considered to apply the kalman filter")
    parser.add_argument("--confidence", type=float, required=True, help="Model confidence")
    parser.add_argument("--max_coast", type=float, default=0.25, help="Seconds the filter may coast through missed detections before the track is dropped")
    parser.add_argument("--process_noise", type=float, default=5000.0, help="Kalman process noise (px^2/s^4). Higher is more responsive")
    parser.add_argument("--measurement_noise", type=float, default=25.0, help="Kalman measurement noise (px^2)")
    parser.add_argument("--max_rejections", type=int, default=5, help="Consecutive gated-out measurements before the filter is considered locked out and reset")
    return parser.parse_args()


def seed_velocity(history):
    """
    Least-squares velocity (px/s) over the buffered detections. Using a fit over
    ~0.3 s rather than a single frame difference keeps detector noise from being
    amplified by the short frame interval.
    """
    if len(history) < 2:
        return 0.0, 0.0

    t = np.array([h[0] for h in history], dtype=np.float64)
    t = t - t[0]
    span = t[-1]
    if span < 1e-3:
        return 0.0, 0.0

    x = np.array([h[1] for h in history], dtype=np.float64)
    y = np.array([h[2] for h in history], dtype=np.float64)

    A = np.vstack([t, np.ones_like(t)]).T
    vx = float(np.linalg.lstsq(A, x, rcond=None)[0][0])
    vy = float(np.linalg.lstsq(A, y, rcond=None)[0][0])
    return vx, vy


def main():
    args = parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("app")

    detector = DetectionService()
    model_path = os.path.join(MODELS_DIR, args.model)

    kalman = KalmanFilter2D(
        process_noise=args.process_noise,
        measurement_noise=args.measurement_noise,
    )
    last_time = time.time()

    consecutive_detections = 0
    history = deque()
    last_detection_time = None
    publishing = False

    streamer = GStreamerFrameStreamer(port=args.streaming_port, logger=logger)
    streamer.start()

    logger.info("Waiting for stream on UDP %s", args.streaming_port)

    with TerminalKeyReader() as key_reader:
        conf = zenoh.Config()
        conf.insert_json5("connect/endpoints", '["tcp/192.168.0.125:7447"]')
        with zenoh.open(conf) as session:
            cop = "cop1" if args.drone_name == "X500B" else "cop2"
            key = f"{cop}/movement/kalman"
            pub = session.declare_publisher(key)

            while True:
                frame = streamer.get_frame()

                if frame is None:
                    time.sleep(0.005)
                    continue

                detection = detector.detect(
                    model_path=model_path,
                    frame=frame,
                    confidence=args.confidence,
                )

                now = time.time()
                dt = now - last_time
                last_time = now

                if detection is not None:
                    cx, cy = detection.center

                    consecutive_detections += 1
                    last_detection_time = now

                    history.append((now, cx, cy))
                    while history and now - history[0][0] > SEED_WINDOW_S:
                        history.popleft()

                    if not kalman.initialized:
                        # After n consecutive detections we initialize kalman, seeding
                        # the velocity from the buffered detections so the estimate does
                        # not have to converge from zero.
                        if consecutive_detections >= args.min_consecutive_detections:
                            svx, svy = seed_velocity(history)
                            kalman.initialize(cx, cy, svx, svy)

                            logger.info(
                                "Kalman initialized after %d consecutive detections (seed vx=%.1f vy=%.1f px/s)",
                                consecutive_detections, svx, svy,
                            )
                    else:
                        kalman.predict(dt)
                        kalman.update((cx, cy))
                        if not kalman.last_accepted:
                            logger.warning(
                                "Rejected outlier detection (NIS=%.1f, %d in a row)",
                                kalman.last_nis, kalman.consecutive_rejections,
                            )

                        # Without this the filter can lock out: once the prediction
                        # drifts off the target every measurement fails the gate, nothing
                        # is fused, and the state coasts at a constant velocity forever
                        # while still publishing.
                        if kalman.consecutive_rejections >= args.max_rejections:
                            logger.warning(
                                "Filter locked out after %d rejections, resetting",
                                kalman.consecutive_rejections,
                            )
                            kalman.reset()
                            consecutive_detections = 0
                            history.clear()
                            last_detection_time = None

                else:
                    # No detection this frame. Coast on the motion model for a short
                    # window, then drop the track entirely rather than letting the state
                    # free-run through a multi-second blackout.
                    coasting = (
                        kalman.initialized
                        and last_detection_time is not None
                        and (now - last_detection_time) <= args.max_coast
                    )

                    if coasting:
                        kalman.predict(dt)
                    else:
                        if kalman.initialized:
                            logger.info("Track lost, resetting filter")
                        kalman.reset()
                        consecutive_detections = 0
                        history.clear()
                        last_detection_time = None

                # Publishing if target is stable
                coasting_now = (
                    detection is None
                    and kalman.initialized
                    and last_detection_time is not None
                    and (now - last_detection_time) <= args.max_coast
                )

                if (
                    kalman.initialized
                    and consecutive_detections >= args.min_consecutive_detections
                    and (detection is not None or coasting_now)
                ):
                    vx, vy = kalman.velocity

                    msg = serialize_vector3(vx, vy, 0.0)
                    pub.put(msg)
                    publishing = True

                    logger.info(
                        "Published vx=%.3f vy=%.3f",
                        vx,
                        vy,
                    )

                elif publishing:
                    # Emit a single zero so the consumer stops applying a stale
                    # correction for the duration of the dropout.
                    pub.put(serialize_vector3(0.0, 0.0, 0.0))
                    publishing = False

                    logger.info("Published zero (no valid track)")

                display = draw_tracking_overlay(frame, detection)

                cv2.imshow(f"{args.drone_name}", display)

                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

                if key_reader.poll() == "q":
                    break

    logger.info("Shutting down")

    streamer.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()