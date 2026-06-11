from argparse import ArgumentParser
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


CAMERA_IP = "192.168.144.25"
CAMERA_PORT = 5600

BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "usv_detection/models")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="yolo26x")
    parser.add_argument("--min_consecutive_detections", type=int, required=True)
    parser.add_argument("--confidence", type=float, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("app")

    camera = SIYISDK(server_ip=CAMERA_IP, port=CAMERA_PORT)

    detector = DetectionService()
    model_path = os.path.join(MODELS_DIR, args.model)

    kalman = KalmanFilter2D()
    last_time = time.time()

    consecutive_detections = 0

    streamer = GStreamerFrameStreamer(port=CAMERA_PORT, logger=logger)
    streamer.start()

    logger.info("Waiting for stream on UDP %s", CAMERA_PORT)

    with TerminalKeyReader() as key_reader:
        with zenoh.open(zenoh.Config()) as session:
            key = "cop1/movement/kalman"
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

                # Prediction step
                kalman.predict(dt)

                if detection is not None:
                    consecutive_detections += 1

                    cx, cy = detection.center

                    if not kalman.initialized:
                        # After n consecutive detections we initialize kalman, otherwise we just update it
                        if consecutive_detections >= args.min_consecutive_detections:
                            kalman.initialize(cx, cy)

                            logger.info("Kalman initialized after %d consecutive detections", consecutive_detections)
                    else:
                        kalman.update((cx, cy))

                else:
                    consecutive_detections = 0

                # Publishing if target is stable
                if kalman.initialized and consecutive_detections >= args.min_consecutive_detections:
                    vx, vy = kalman.velocity

                    msg = serialize_vector3(vx, vy, 0.0)
                    pub.put(msg)

                    logger.info(
                        "Published vx=%.3f vy=%.3f",
                        vx,
                        vy,
                    )

                display = draw_tracking_overlay(frame, detection)

                cv2.imshow("Siyi USVs detection", display)

                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

                if key_reader.poll() == "q":
                    break

    logger.info("Shutting down")

    streamer.stop()
    camera.disconnect()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()