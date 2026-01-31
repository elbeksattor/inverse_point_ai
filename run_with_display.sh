#!/bin/bash
# IP AI v3 - Run with HDMI Display
# Convenience script to run the pipeline with display output
#
# Usage: ./run_with_display.sh
#
# Note: Requires HDMI monitor connected and X display available

set -e
cd "$(dirname "$0")"

# Set display for X11
export DISPLAY=:1

echo "Starting IP AI v3 with HDMI display..."
echo "Make sure HDMI monitor is connected."
echo ""

# Run main script with display enabled
exec ./run_camera.sh "$@"
