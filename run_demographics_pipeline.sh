#!/bin/bash
################################################################################
# IP AI Analytics - DeepStream Pipeline with Demographics
# Production-ready pipeline with:
# - Person detection (PeopleNet)
# - Face detection (PeopleNet class 2)
# - Person tracking with ReID (NvDCF)
# - Age/Gender estimation (GenderAge SGIE)
################################################################################

set -e

PROJECT_ROOT="/home/nvidia/projects/inverse_point/ip_ai_analytics"
VIDEO_PATH="/home/nvidia/projects/inverse_point/ip_ai_assist_old/test_for_ai/video_2025-11-08_15-41-41.mp4"

# Output paths
OUTPUT_PATH="$PROJECT_ROOT/output/demographics_output.mp4"
DB_PATH="$PROJECT_ROOT/output/database/person_database.db"

# Config paths
DETECTOR_CONFIG="$PROJECT_ROOT/configs/peoplenet_with_face_config.txt"
TRACKER_CONFIG="$PROJECT_ROOT/configs/nvdcf_tracker_config.yml"
DEMOGRAPHICS_CONFIG="$PROJECT_ROOT/configs/demographics_sgie_config.txt"

cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/output/database"

echo "=============================================="
echo "IP AI Analytics - Demographics Pipeline"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Video:       $VIDEO_PATH"
echo "  Output:      $OUTPUT_PATH"
echo "  Detector:    $DETECTOR_CONFIG"
echo "  Tracker:     $TRACKER_CONFIG"
echo "  Demographics: $DEMOGRAPHICS_CONFIG"
echo "  Database:    $DB_PATH"
echo ""

# Clean previous outputs
rm -f "$OUTPUT_PATH"
rm -rf "$(dirname "$DB_PATH")"
mkdir -p "$(dirname "$DB_PATH")"

# Record start time
START_TIME=$(date +%s)

echo "Starting DeepStream Demographics Pipeline..."
echo "----------------------------------------------"

python3 src/pipeline/deepstream_demographics_pipeline.py \
    --video "$VIDEO_PATH" \
    --output "$OUTPUT_PATH" \
    --detector-config "$DETECTOR_CONFIG" \
    --tracker-config "$TRACKER_CONFIG" \
    --demographics-config "$DEMOGRAPHICS_CONFIG" \
    --db-path "$DB_PATH"

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=============================================="
echo "Pipeline Completed in ${ELAPSED}s"
echo "=============================================="

if [ -f "$OUTPUT_PATH" ]; then
    echo "Output video: $OUTPUT_PATH ($(du -h "$OUTPUT_PATH" | cut -f1))"
fi

if [ -f "$DB_PATH" ]; then
    echo "Database: $DB_PATH"
    echo ""
    echo "Database Statistics:"
    sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM persons;" 2>/dev/null | xargs echo "  Total unique persons:"
    sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM appearances;" 2>/dev/null | xargs echo "  Total appearances:"
fi

echo "=============================================="
