#!/bin/bash
# IP AI v3 - Run with HDMI Display
# Convenience script to run the pipeline with display output
#
# Usage: ./run_with_display.sh [options]
#
# Note: Requires HDMI monitor connected and X display available

set -e
cd "$(dirname "$0")"

# Set display for X11
export DISPLAY=:1
export XAUTHORITY=/run/user/1000/gdm/Xauthority

echo "Starting IP AI v3 with HDMI display..."
echo "Make sure HDMI monitor is connected."
echo ""

# Default to 1080p with headpose enabled
WIDTH="${WIDTH:-1920}"
HEIGHT="${HEIGHT:-1080}"

# Run with headpose model and strict attention thresholds (±5°)
exec python3 src/main.py \
    --camera "${CAMERA:-/dev/video0}" \
    --width "$WIDTH" \
    --height "$HEIGHT" \
    --fps "${FPS:-30}" \
    --headpose-model models/headpose/whenet_prepost.onnx \
    --yaw-threshold 10.0 \
    --pitch-threshold 10.0 \
    "$@"
