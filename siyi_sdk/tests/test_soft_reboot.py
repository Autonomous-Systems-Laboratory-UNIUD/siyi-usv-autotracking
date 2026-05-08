"""
@file test_soft_reboot.py
@Description: This is a test script for using the SIYI SDK Python implementation to reboot the gimbal, camera or both
@Author: Mohamed Abdelkader
@Contact: mohamedashraf123@gmail.com
All rights reserved 2022
"""

import sys
import os
import socket
from time import sleep
import threading
import gc
current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)
  
sys.path.append(parent_directory)

from siyi_sdk import SIYISDK

def test():
    cam = SIYISDK(server_ip="192.168.144.25", port=37260)

    if not cam.connect():
        print("No connection ")
        exit(1)

    # reboot gimbal
    print("Requesting to reboot the gimbal")
    cam.requestGimbalCameraSoftReboot(reboot_gimbal=True)

    if cam.getGimbalRebooted():
        print("Successfully rebooted the gimbal")
    else:
        print("Failed to reboot the gimbal")

    # reboot camera
    print("Request to reboot the camera")
    cam.requestGimbalCameraSoftReboot(reboot_camera=True)

    if not cam.getCameraRebooted():
        print("Failed to reboot the camera")
        exit(1)
    print("Successfully rebooted the camera")
    if not cam.connect():
        print("No connection")
        exit(1)

    # reboot gimbal and camera    
    print("Request to reboot the gimbal and camera")
    cam.requestGimbalCameraSoftReboot(reboot_camera=True, reboot_gimbal=True)
    if not cam.getCameraRebooted():
        print("Failed to reboot the camera")
        exit(1)

    if not cam.getGimbalRebooted():
        print("Failed to reboot the gimbal")
        exit(1)

    print("Successfully rebooted the camera and gimbal")
    if not cam.connect():
        print("No connection")
        exit(1)

    cam.requestCenterGimbal()
    sleep(1)
    cam.requestSetAngles(20, 20)
    sleep(1)
    cam.disconnect()

if __name__ == "__main__":
    test()