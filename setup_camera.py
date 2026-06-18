import sys
import logging
import os
import cv2
from time import sleep
from argparse import ArgumentParser
from keyboard_manager import TerminalKeyReader

SDK_DIR = os.path.join(os.path.dirname(__file__), "siyi_sdk")
if SDK_DIR not in sys.path:
    sys.path.insert(0, SDK_DIR)

from siyi_sdk import SIYISDK


parser = ArgumentParser()
parser.add_argument('--yaw', type=float, default=0.0)
parser.add_argument('--pitch', type=float, default=0.0)
parser.add_argument('--rec', type=bool, default=False)

args = parser.parse_args()

def test(yaw, pitch, rec):
    cam = SIYISDK(server_ip="192.168.144.25", port=37260)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if not cam.connect():
        logger.info("No connection, exiting.")
        sys.exit(1)

    try:
        logger.info("Camera type: %s", cam._hw_msg.cam_type_str)

        cam.requestCenterGimbal()
        sleep(1)

        cam.requestFPVMode()
        sleep(1)

        logger.info("Locking camera orientation.")

        while True:
            cam.requestSetAngles(yaw, pitch)
            sleep(0.1)

    except KeyboardInterrupt:
        logger.info("Stopping.")

    finally:
        cam.disconnect()

if __name__ == "__main__":
    test(args.yaw, args.pitch, args.rec)
