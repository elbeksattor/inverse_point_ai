#!/bin/bash
# IP AI v3 - Run Camera Pipeline
# Usage: ./run_camera.sh [options]
#
# This is the main run script with all options.
# For convenience, use:
#   ./run_with_display.sh  - Run with HDMI display
#   ./run_headless.sh      - Run without display (SSH)

set -e

# Change to project directory
cd "$(dirname "$0")"

# Default values
CAMERA="${CAMERA:-/dev/video0}"
WIDTH="${WIDTH:-1280}"
HEIGHT="${HEIGHT:-720}"
FPS="${FPS:-30}"
NO_DISPLAY="${NO_DISPLAY:-false}"
NO_DATABASE="${NO_DATABASE:-false}"
DB_PATH="${DB_PATH:-data/person_database}"
REID_THRESHOLD="${REID_THRESHOLD:-0.50}"
LOG_LEVEL="${LOG_LEVEL:-info}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--camera)
            CAMERA="$2"
            shift 2
            ;;
        -W|--width)
            WIDTH="$2"
            shift 2
            ;;
        -H|--height)
            HEIGHT="$2"
            shift 2
            ;;
        -f|--fps)
            FPS="$2"
            shift 2
            ;;
        --no-display)
            NO_DISPLAY="true"
            shift
            ;;
        --no-database)
            NO_DATABASE="true"
            shift
            ;;
        --db-path)
            DB_PATH="$2"
            shift 2
            ;;
        --reid-threshold)
            REID_THRESHOLD="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        -h|--help)
            echo "IP AI v3 - Camera Pipeline"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  -c, --camera DEVICE       Camera device (default: /dev/video0)"
            echo "  -W, --width WIDTH         Frame width (default: 1280)"
            echo "  -H, --height HEIGHT       Frame height (default: 720)"
            echo "  -f, --fps FPS             Target FPS (default: 30)"
            echo "  --no-display              Run without display (headless mode)"
            echo "  --no-database             Disable RE-ID database"
            echo "  --db-path PATH            Database path (default: data/person_database)"
            echo "  --reid-threshold THRESH   RE-ID similarity threshold (default: 0.50)"
            echo "  --log-level LEVEL         Logging level: debug|info|warn|error (default: info)"
            echo "  -h, --help                Show this help"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Run with defaults"
            echo "  $0 --no-display                       # Headless mode"
            echo "  $0 --camera /dev/video2               # Different camera"
            echo "  $0 --reid-threshold 0.60              # Higher RE-ID threshold"
            echo "  $0 --log-level debug                  # Verbose logging"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if camera exists
if [ ! -e "$CAMERA" ]; then
    echo "Error: Camera device not found: $CAMERA"
    echo ""
    echo "Available video devices:"
    ls -la /dev/video* 2>/dev/null || echo "  No video devices found"
    exit 1
fi

# Build command
CMD="python3 src/main.py"
CMD="$CMD --camera $CAMERA"
CMD="$CMD --width $WIDTH"
CMD="$CMD --height $HEIGHT"
CMD="$CMD --fps $FPS"
CMD="$CMD --db-path $DB_PATH"
CMD="$CMD --reid-threshold $REID_THRESHOLD"
CMD="$CMD --log-level $LOG_LEVEL"

if [ "$NO_DISPLAY" = "true" ]; then
    CMD="$CMD --no-display"
fi

if [ "$NO_DATABASE" = "true" ]; then
    CMD="$CMD --no-database"
fi

# Print info
echo "=============================================="
echo "IP AI v3 - Camera Pipeline (Phase 4)"
echo "=============================================="
echo "Camera:       $CAMERA"
echo "Resolution:   ${WIDTH}x${HEIGHT} @ ${FPS}fps"
echo "Display:      $([ "$NO_DISPLAY" = "true" ] && echo "Disabled" || echo "Enabled")"
echo "RE-ID DB:     $([ "$NO_DATABASE" = "true" ] && echo "Disabled" || echo "Enabled ($DB_PATH)")"
echo "RE-ID Thresh: $REID_THRESHOLD"
echo "Log Level:    $LOG_LEVEL"
echo "=============================================="
echo ""
echo "Features enabled:"
echo "  - Person detection (PeopleNet)"
echo "  - Multi-object tracking (NvDeepSORT)"
echo "  - RE-ID database (FAISS + SQLite)"
echo "  - Demographics (InsightFace GenderAge)"
echo ""
echo "Press Ctrl+C to stop"
echo "=============================================="
echo ""

# Run
exec $CMD
