#!/bin/bash
################################################################################
# FAST Test Script for IP AI Analytics DeepStream Pipeline
# Keeps database to avoid re-initialization overhead
# Use this for iterative testing during development
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
mkdir -p "$PROJECT_ROOT/output/database"

# Change to project directory
cd "$PROJECT_ROOT"

# Only remove output video (keep database for faster startup)
if [ -f "$OUTPUT_PATH" ]; then
    rm "$OUTPUT_PATH"
fi

# Print configuration
echo "================================"
echo "IP AI Analytics - FAST Test Run"
echo "================================"
echo "Video:    $VIDEO_PATH"
echo "Output:   $OUTPUT_PATH"
echo "Database: $DB_PATH (preserved)"
echo "================================"
echo ""

# Record start time
START_TIME=$(date +%s)

# Run pipeline
echo "Starting DeepStream pipeline..."
python3 src/pipeline/deepstream_pipeline.py \
    --video "$VIDEO_PATH" \
    --output "$OUTPUT_PATH" \
    --detector-config "$DETECTOR_CONFIG" \
    --tracker-config "$TRACKER_CONFIG" \
    --db-path "$DB_PATH"

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

# Check results
echo ""
echo "================================"
echo "Pipeline completed in ${ELAPSED}s"
echo "================================"

if [ -f "$OUTPUT_PATH" ]; then
    echo "Output video: $OUTPUT_PATH ($(du -h "$OUTPUT_PATH" | cut -f1))"
fi

if [ -f "$DB_PATH" ]; then
    echo "Database: $DB_PATH"
    sqlite3 "$DB_PATH" "SELECT COUNT(*) as total_persons FROM persons;" 2>/dev/null | head -1 | xargs echo "  Total unique persons:"
fi

echo "================================"
