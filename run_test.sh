#!/bin/bash
################################################################################
# Test Script for IP AI Analytics DeepStream Pipeline
# Runs person detection + tracking + RE-ID on test video
################################################################################

# Exit on error
set -e

# Project paths
PROJECT_ROOT="/home/nvidia/projects/inverse_point/ip_ai_analytics"
VIDEO_PATH="/home/nvidia/projects/inverse_point/ip_ai_assist_old/test_for_ai/video_2025-11-08_15-41-41.mp4"
OUTPUT_PATH="$PROJECT_ROOT/output/test_output.mp4"
DETECTOR_CONFIG="$PROJECT_ROOT/configs/peoplenet_detector_config.txt"
TRACKER_CONFIG="$PROJECT_ROOT/configs/nvdcf_tracker_config.yml"
DB_PATH="$PROJECT_ROOT/output/database/person_database.db"

# Create output directory
mkdir -p "$PROJECT_ROOT/output"

# Change to project directory
cd "$PROJECT_ROOT"

# Clean previous output
if [ -f "$OUTPUT_PATH" ]; then
    echo "Removing previous output video..."
    rm "$OUTPUT_PATH"
fi

if [ -f "$DB_PATH" ]; then
    echo "Removing previous database..."
    rm -rf "$(dirname "$DB_PATH")"
fi

# Print configuration
echo "================================"
echo "IP AI Analytics - Test Run"
echo "================================"
echo "Video:    $VIDEO_PATH"
echo "Output:   $OUTPUT_PATH"
echo "Detector: $DETECTOR_CONFIG"
echo "Tracker:  $TRACKER_CONFIG"
echo "Database: $DB_PATH"
echo "================================"
echo ""

# Run pipeline
echo "Starting DeepStream pipeline..."
python3 src/pipeline/deepstream_pipeline.py \
    --video "$VIDEO_PATH" \
    --output "$OUTPUT_PATH" \
    --detector-config "$DETECTOR_CONFIG" \
    --tracker-config "$TRACKER_CONFIG" \
    --db-path "$DB_PATH"

# Check results
echo ""
echo "================================"
echo "Pipeline completed!"
echo "================================"

if [ -f "$OUTPUT_PATH" ]; then
    echo "Output video: $OUTPUT_PATH ($(du -h "$OUTPUT_PATH" | cut -f1))"
else
    echo "WARNING: Output video not created"
fi

if [ -f "$DB_PATH" ]; then
    echo "Database: $DB_PATH ($(du -h "$DB_PATH" | cut -f1))"

    # Print database statistics
    echo ""
    echo "Database Statistics:"
    sqlite3 "$DB_PATH" "SELECT COUNT(*) as total_persons FROM persons;" 2>/dev/null | head -1 | xargs echo "  Total unique persons:"
    sqlite3 "$DB_PATH" "SELECT COUNT(*) as total_appearances FROM appearances;" 2>/dev/null | head -1 | xargs echo "  Total appearances:"
else
    echo "WARNING: Database not created"
fi

echo "================================"
