"""
@file test_get_encoding_param.py
@Description: This is a test script for using the SIYI SDK Python implementation to get camera encoding parameters
@Author: Mohamed Abdelkader
@Contact: mohamedashraf123@gmail.com
All rights reserved 2022
"""
import sys
import os
from time import sleep

current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)
  
sys.path.append(parent_directory)

from siyi_sdk import SIYISDK

def print_encoding_param(encoding_param: tuple | None):
    if encoding_param is None:
        return
    stream_type, encoding_type, resolution_width, resolution_height, video_kbps, video_frame_rate = encoding_param
    stream_type_str = None
    encoding_type_str = None
    if stream_type == 0:
        stream_type_str = "Recording stream"
    elif stream_type == 1:
        stream_type_str = "Main stream"
    elif stream_type == 2:
        stream_type_str = "Sub-stream"

    if encoding_type == 1:
        encoding_type_str = "H264"
    elif encoding_type == 2:
        encoding_type_str = "H265"
    print(
        f"stream_type: {stream_type_str}\n"
        f"encoding_type: {encoding_type_str}\n"
        f"resolution: {resolution_width} x {resolution_height}\n"
        f"video_bitrate: {video_kbps} kbps\n"
        f"video_frame_rate: {video_frame_rate} fps\n"
    )

def test():
    cam = SIYISDK(server_ip="192.168.144.25", port=37260)
    if not cam.connect():
        print("No connection ")
        exit(1)

    for i in range(0,3):
        cam.requestCameraEncodingParameters(i)
        sleep(1)
        print_encoding_param(cam.getCameraEncodingParameters(i))
    cam.disconnect()

if __name__ == "__main__":
    test()