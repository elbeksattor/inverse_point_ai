#!/bin/bash
################################################################################
# Full Test Script with Demographics Overlay
#
# This script:
# 1. Runs the main DeepStream pipeline (person detection + tracking + RE-ID)
# 2. Adds demographics overlay (face detection + age/gender)
# 3. Produces final video with full annotations
################################################################################

set -e

PROJECT_ROOT="/home/nvidia/projects/inverse_point/ip_ai_analytics"
VIDEO_PATH="/home/nvidia/projects/inverse_point/ip_ai_assist_old/test_for_ai/video_2025-11-08_15-41-41.mp4"

# Intermediate output (RE-ID only)
REID_OUTPUT="$PROJECT_ROOT/output/test_reid_output.mp4"

# Final output (RE-ID + Demographics)
FINAL_OUTPUT="$PROJECT_ROOT/output/test_with_demographics.mp4"

DETECTOR_CONFIG="$PROJECT_ROOT/configs/peoplenet_detector_config.txt"
TRACKER_CONFIG="$PROJECT_ROOT/configs/nvdcf_tracker_config.yml"
DB_PATH="$PROJECT_ROOT/output/database/person_database.db"

cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/output/database"

echo "================================"
echo "IP AI Analytics - Full Pipeline"
echo "================================"
echo ""

# Clean previous outputs
rm -f "$REID_OUTPUT" "$FINAL_OUTPUT"

START_TIME=$(date +%s)

echo "Step 1: Running DeepStream Pipeline (RE-ID)..."
echo "----------------------------------------------"
python3 src/pipeline/deepstream_pipeline.py \
    --video "$VIDEO_PATH" \
    --output "$REID_OUTPUT" \
    --detector-config "$DETECTOR_CONFIG" \
    --tracker-config "$TRACKER_CONFIG" \
    --db-path "$DB_PATH"

REID_TIME=$(date +%s)
echo ""
echo "RE-ID pipeline completed in $((REID_TIME - START_TIME))s"
echo ""

echo "Step 2: Adding Demographics Overlay..."
echo "--------------------------------------"
python3 tools/add_demographics_overlay.py \
    --input "$VIDEO_PATH" \
    --output "$FINAL_OUTPUT" \
    --face-threshold 0.5

END_TIME=$(date +%s)

echo ""
echo "================================"
echo "Full Pipeline Completed!"
echo "================================"
echo "Total time: $((END_TIME - START_TIME))s"
echo ""
echo "Output files:"
echo "  RE-ID only:    $REID_OUTPUT"
echo "  With Demographics: $FINAL_OUTPUT"
echo ""

if [ -f "$FINAL_OUTPUT" ]; then
    echo "Final video size: $(du -h "$FINAL_OUTPUT" | cut -f1)"
fi

echo "================================"
