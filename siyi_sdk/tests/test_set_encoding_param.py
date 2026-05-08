"""
@file test_set_encoding_param.py
@Description: This is a test script for using the SIYI SDK Python implementation to set camera encoding parameters
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

def test():
    cam = SIYISDK(server_ip="192.168.144.25", port=37260)
    if not cam.connect():
        print("No connection ")
        exit(1)
    stream_type = 0
    cam.setCameraEncodingParameters(stream_type, 1, 2560, 1440, 30000)
    sleep(1)
    set_success = cam.getCameraEncodingParametersFeedback(stream_type)
    if set_success:
        print(f"Successfully updated camera encoding parameters for stream {stream_type}")
    else:
        print(f"Failed to update camera encoding parameters for stream {stream_type} ")
    cam.disconnect()

if __name__ == "__main__":
    test()