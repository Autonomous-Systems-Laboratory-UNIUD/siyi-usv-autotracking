#!/bin/bash

PORT=$1
GROUNDSTATION_IP="192.168.0.125"

trap 'kill -- -$$' INT TERM EXIT

uv run setup_camera.py --yaw 0 --pitch -90 &

sleep 7

gst-launch-1.0 \
    rtspsrc location=rtsp://192.168.144.25:8554/main.264 latency=0 ! \
    rtph264depay ! \
    h264parse ! \
    rtph264pay pt=96 config-interval=1 ! \
    udpsink host=$GROUNDSTATION_IP port=$PORT