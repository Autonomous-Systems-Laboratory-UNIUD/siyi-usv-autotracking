from argparse import ArgumentParser
import os
import sys
import time
import cv2
import logging

from auto_tracking import SiyiAutoTracker, draw_tracking_overlay
from detector_adapter import DetectorAdapter
from gstreamer import GStreamerFrameStreamer
from keyboard_manager import TerminalKeyReader

SDK_DIR = os.path.join(os.path.dirname(__file__), "siyi_sdk")
if SDK_DIR not in sys.path:
    sys.path.insert(0, SDK_DIR)

from siyi_sdk import SIYISDK

RTSP_URL = "rtsp://192.168.144.25:8554/main.264"
CAMERA_IP = "192.168.144.25"
CAMERA_PORT = 37260
WINDOW_NAME = "Latest frame from appsink"
MODELS_DIR_PATH = os.path.join(os.path.dirname(__file__), "usv_detection/models")

parser = ArgumentParser(prog="SIYI Auto Tracker", description="A tool to track ASVs using a SIYI camera and a YOLO models.")
parser.add_argument("--rtsp-url", type=str, default=RTSP_URL, help="RTSP URL of the camera stream")
parser.add_argument("--rtsp-transport", choices=("udp", "tcp"), default="udp", help="RTSP transport to use (default: udp)")
parser.add_argument("--rtsp-latency", type=int, default=10, help="RTSP jitter buffer latency in milliseconds (default: 10)")
parser.add_argument("--camera-ip", type=str, default=CAMERA_IP, help="IP address of the SIYI camera for control")
parser.add_argument("--camera-port", type=int, default=CAMERA_PORT, help="Port number for SIYI camera control")
parser.add_argument("--model", type=str, default="yolo26x", help="Object detection model to use (default: yolo26x)")
parser.add_argument("--confidence", type=float, required=True, help="Confidence threshold for object detection")
args = parser.parse_args()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main():
    camera = None
    tracker = None
    detector = DetectorAdapter(f"{MODELS_DIR_PATH}/{args.model}", args.confidence)
    streamer = GStreamerFrameStreamer(
        args.rtsp_url,
        rtsp_transport=args.rtsp_transport,
        rtsp_latency=args.rtsp_latency,
        logger=logger,
    )

    try:
        camera = SIYISDK(server_ip=args.camera_ip, port=args.camera_port, debug=False)
        if camera.connect():
            tracker = SiyiAutoTracker(camera)
            logger.info("Camera control connected.")
        else:
            # Ensure we don't call camera.disconnect() at the end in this case
            camera = None
            logger.warning("Camera control unavailable, streaming only.")
    except Exception as exc:
        logger.error(f"Camera control initialization failed: {exc}")

    logger.info("Starting RTSP stream %s over %s.", args.rtsp_url, args.rtsp_transport.upper())
    streamer.start()

    if camera is not None:
        camera.requestCenterGimbal()
    logger.info("Pipeline started, camera is centered and available. Press 'q' to quit.")

    with TerminalKeyReader() as key_reader:
        try:
            while True:
                frame = streamer.get_frame()

                if frame is None:
                    if tracker is not None:
                        tracker.update(None, None)
                else:
                    detection = detector.detect(frame)
                    command = None

                    if tracker is not None:
                        command = tracker.update(frame.shape, detection)
                    display = draw_tracking_overlay(
                        frame,
                        detection,
                        command=command,
                        estimated_center=(None if tracker is None else tracker.estimated_center),
                        estimated_velocity=(None if tracker is None else tracker.estimated_velocity),
                    )
                    cv2.imshow(WINDOW_NAME, display)

                # Listening for 'q' key press in the OpenCV window
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

                # Listening for 'q' key press in the terminal
                terminal_key = key_reader.poll()
                if terminal_key == "q":
                    break

                # Small sleep to prevent high CPU usage
                time.sleep(0.001)  
        finally:
            streamer.stop()
            if tracker is not None:
                tracker.stop()
            if camera is not None:
                camera.disconnect()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
