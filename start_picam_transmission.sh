#!/bin/bash

PORT=$1
GROUNDSTATION_IP="192.168.0.125"

rpicam-vid \
    --timeout 0 \
    --nopreview \
    --width 1280 \
    --height 720 \
    --framerate 30 \
    --codec h264 \
    --bitrate 4000000 \
    --inline \
    --output - \
    | gst-launch-1.0 -q \
        fdsrc fd=0 \
        ! h264parse \
        ! rtph264pay pt=96 config-interval=1 \
        ! udpsink host=$GROUNDSTATION_IP port=$PORT sync=false async=false