"""
@file test_format_sd_card.py
@Description: This is a test script for using the SIYI SDK Python implementation to format the microSD-card
@Author: Mohamed Abdelkader
@Contact: mohamedashraf123@gmail.com
All rights reserved 2022
"""
import sys
import os
import json
from urllib.request import urlopen
from time import sleep
  
current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)
  
sys.path.append(parent_directory)

from siyi_sdk import SIYISDK

def list_pictures(images: list):
    print("Listing images...")
    i = 0
    for img in images:
        i += 1
        print(img['name'])
    print(f"Found {i} images on microSD-card")

def get_list_pictures(ip: str):
    with urlopen(f"http://{ip}:82/cgi-bin/media.cgi/api/v1/getmedialist?media_type=0&path=101SIYI_IMG&start=0&count=9999") as url:
        data = json.load(url)
        if data['code'] != 200:
            raise RuntimeError("Failed to list images")
        return data['data']['list']

def test():
    ip = "192.168.144.25"
    cam = SIYISDK(server_ip=ip, port=37260, debug=False)
    if not cam.connect():
        print("No connection ")
        exit(1)
    
    print("Taking a picture...")
    cam.requestPhoto()
    sleep(1)

    print("\nBefore formatting the microSD-card")
    picture_list_before = get_list_pictures(ip) 
    picture_count_before = len(picture_list_before)
    list_pictures(picture_list_before)
    print()

    cam.requestFormatSdCard()
    print("Formatting the microSD-card...")
    sleep(2)
    print("\nAfter formatting the microSD-card")
    picture_list_after = get_list_pictures(ip)
    picture_count_after = len(picture_list_after)
    list_pictures(picture_list_after)
    print()
    if cam.getCameraTypeString() in ['ZT30', 'ZR30', 'A8 mini']:
        if picture_count_before is picture_count_after:
            print("Failed to format the microSD-card")
        else:
            print("Successfully formatted the microSD-card")
    else:
        if cam.getFormatSdCardFeedback():
            print("Successfully formatted the microSD-card")
        else:
            print("Failed to format the microSD-card")
    cam.disconnect()

if __name__ == "__main__":
    test()