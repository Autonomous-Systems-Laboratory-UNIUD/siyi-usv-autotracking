"""
Live SIYI camera control example.

Features:
- Connects to the camera over UDP
- Displays the RTSP stream in an OpenCV window
- Optionally republishes frames to an RTMP endpoint
- Lets you control the gimbal and zoom via keyboard

Keyboard:
- w / s: pitch up / down
- a / d: yaw left / right
- + / -: zoom in / out
- c: center gimbal
- 1 / 2 / 3: follow / lock / fpv mode
- p: photo
- r: start/stop recording
- h: print help in terminal
- q: quit
"""

import argparse
import os
import sys
from time import monotonic, sleep

import cv2

current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)

sys.path.append(parent_directory)

from siyi_sdk import SIYISDK
from stream import RTMPSender, SIYIRTSP


WINDOW_NAME = "SIYI Live Control"


def print_help():
    print("Keyboard controls:")
    print("  w / s : pitch up / down")
    print("  a / d : yaw left / right")
    print("  + / - : zoom in / out")
    print("  c     : center gimbal")
    print("  1 / 2 / 3 : follow / lock / fpv mode")
    print("  p     : take photo")
    print("  r     : toggle recording")
    print("  h     : print this help")
    print("  q     : quit")


def overlay_status(frame, lines):
    display = frame.copy()
    y = 24
    for line in lines:
        cv2.putText(
            display,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        y += 24
    return display


def parse_args():
    parser = argparse.ArgumentParser(description="Live SIYI stream, keyboard control and optional RTMP publish")
    parser.add_argument("--server-ip", default="192.168.144.25", help="SIYI camera IP address")
    parser.add_argument("--port", type=int, default=37260, help="SIYI UDP control port")
    parser.add_argument("--rtsp-url", default="", help="RTSP URL; defaults to rtsp://<server-ip>:8554/main.264")
    parser.add_argument("--rtmp-url", default="", help="Optional RTMP publish URL")
    parser.add_argument("--yaw-speed", type=int, default=25, help="Yaw speed for each keyboard pulse")
    parser.add_argument("--pitch-speed", type=int, default=25, help="Pitch speed for each keyboard pulse")
    parser.add_argument("--move-duration", type=float, default=0.12, help="Duration in seconds of each movement pulse")
    parser.add_argument("--zoom-duration", type=float, default=0.35, help="Duration in seconds of each zoom pulse")
    parser.add_argument("--stream-start-timeout", type=float, default=3.0, help="Seconds to wait for the first RTSP frame")
    parser.add_argument("--disable-rtsp-low-latency", action="store_true", help="Disable low-latency RTSP capture options")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def get_mode_label(mode):
    mode_map = {
        0: "lock",
        1: "follow",
        2: "fpv",
    }
    return mode_map.get(mode, str(mode))


def main():
    args = parse_args()
    print_help()
    rtsp_url = args.rtsp_url or f"rtsp://{args.server_ip}:8554/main.264"

    cam = SIYISDK(server_ip=args.server_ip, port=args.port, debug=args.debug)
    if not cam.connect():
        print("Failed to connect to camera")
        return 1

    rtsp = None
    rtmp = None
    rtmp_attempted = False
    rtmp_started = False
    last_action = "connected"
    gimbal_deadline = 0.0
    zoom_deadline = 0.0
    gimbal_active = False
    zoom_active = False
    window_ready = False

    try:
        cam_name = cam.getCameraTypeString() or "SIYI"
        rtsp = SIYIRTSP(
            rtsp_url=rtsp_url,
            debug=args.debug,
            cam_name=cam_name,
            use_udp=True,
            allow_tcp_fallback=False,
            low_latency=not args.disable_rtsp_low_latency,
        )
        if not rtsp.isOpened():
            print(f"Failed to open RTSP stream over UDP: {rtsp.getLastError()}")
            print(f"Tried URL: {rtsp_url}")
            print("For A8 mini the official RTSP address is usually rtsp://192.168.144.25:8554/main.264.")
            print("A 404 here usually means the camera is not currently outputting video on Ethernet/RTSP.")
            return 2
        print(f"RTSP backend: {rtsp.getBackend()}  low-latency={'off' if args.disable_rtsp_low_latency else 'on'}")

        startup_deadline = monotonic() + args.stream_start_timeout
        frame = None
        while monotonic() < startup_deadline:
            frame = rtsp.getFrame()
            if frame is not None:
                break
            sleep(0.05)

        if frame is None:
            print(f"Connected to RTSP but received no frames within {args.stream_start_timeout:.1f}s")
            print("This usually means the stream is disabled upstream or the camera is not outputting Ethernet video.")
            return 3

        while True:
            now = monotonic()

            if gimbal_active and now >= gimbal_deadline:
                cam.requestGimbalSpeed(0, 0)
                gimbal_active = False

            if zoom_active and now >= zoom_deadline:
                cam.requestZoomHold()
                cam.requestCurrentZoomLevel()
                zoom_active = False

            frame = rtsp.getFrame()
            if frame is None:
                sleep(0.01)
                continue

            if not window_ready:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(WINDOW_NAME, 960, 540)
                window_ready = True

            if args.rtmp_url and not rtmp_attempted:
                rtmp_attempted = True
                rtmp = RTMPSender(rtmp_url=args.rtmp_url, debug=args.debug)
                rtmp.setImageSize(frame.shape[1], frame.shape[0])
                rtmp.setFPS(15)
                rtmp_started = rtmp.start()
                if rtmp_started:
                    last_action = f"publishing to {args.rtmp_url}"
                else:
                    last_action = "failed to start RTMP publish"

            if rtmp_started and rtmp is not None:
                rtmp.setFrame(frame)

            yaw, pitch, roll = cam.getAttitude()
            zoom_level = cam.getCurrentZoomLevel()
            mode = get_mode_label(cam.getMotionMode())
            status_lines = [
                f"{cam_name}  mode={mode}  zoom={zoom_level}",
                f"yaw={yaw:.1f} pitch={pitch:.1f} roll={roll:.1f}",
                f"last action: {last_action}",
                "w/s pitch  a/d yaw  +/- zoom  c center  1/2/3 mode  q quit",
            ]
            display = overlay_status(frame, status_lines)
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key == 255:
                continue

            if key == ord("q"):
                break
            if key == ord("h"):
                print_help()
                continue
            if key == ord("w"):
                cam.requestGimbalSpeed(0, args.pitch_speed)
                gimbal_deadline = monotonic() + args.move_duration
                gimbal_active = True
                last_action = f"pitch up @{args.pitch_speed}"
                continue
            if key == ord("s"):
                cam.requestGimbalSpeed(0, -args.pitch_speed)
                gimbal_deadline = monotonic() + args.move_duration
                gimbal_active = True
                last_action = f"pitch down @{-args.pitch_speed}"
                continue
            if key == ord("a"):
                cam.requestGimbalSpeed(args.yaw_speed, 0)
                gimbal_deadline = monotonic() + args.move_duration
                gimbal_active = True
                last_action = f"yaw left @{args.yaw_speed}"
                continue
            if key == ord("d"):
                cam.requestGimbalSpeed(-args.yaw_speed, 0)
                gimbal_deadline = monotonic() + args.move_duration
                gimbal_active = True
                last_action = f"yaw right @{-args.yaw_speed}"
                continue
            if key in (ord("+"), ord("=")):
                cam.requestZoomIn()
                zoom_deadline = monotonic() + args.zoom_duration
                zoom_active = True
                last_action = "zoom in"
                continue
            if key in (ord("-"), ord("_")):
                cam.requestZoomOut()
                zoom_deadline = monotonic() + args.zoom_duration
                zoom_active = True
                last_action = "zoom out"
                continue
            if key == ord("c"):
                cam.requestCenterGimbal()
                last_action = "center gimbal"
                continue
            if key == ord("1"):
                cam.requestFollowMode()
                last_action = "follow mode"
                continue
            if key == ord("2"):
                cam.requestLockMode()
                last_action = "lock mode"
                continue
            if key == ord("3"):
                cam.requestFPVMode()
                last_action = "fpv mode"
                continue
            if key == ord("p"):
                cam.requestPhoto()
                last_action = "take photo"
                continue
            if key == ord("r"):
                cam.requestRecording()
                last_action = "toggle recording"
                continue

        return 0
    except KeyboardInterrupt:
        return 130
    finally:
        if rtmp is not None:
            rtmp.stop()
        if rtsp is not None:
            rtsp.close()
        cam.disconnect()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
