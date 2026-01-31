#!/bin/bash
# IP AI v3 - Run Headless (no display)
# Convenience script to run the pipeline without display output
#
# Usage: ./run_headless.sh
#
# This is ideal for SSH sessions or when no monitor is connected.
# All output will be shown in the console.

set -e
cd "$(dirname "$0")"

echo "Starting IP AI v3 in headless mode..."
echo ""

# Run main script without display
exec ./run_camera.sh --no-display "$@"
