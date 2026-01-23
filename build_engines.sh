#!/bin/bash
################################################################################
# Build TensorRT Engines for IP AI Analytics
# This script pre-builds all TensorRT engines so the main pipeline starts fast
# Run this ONCE before using the system
################################################################################

set -e

echo "=============================================="
echo "IP AI Analytics - TensorRT Engine Builder"
echo "=============================================="
echo ""
echo "This will build optimized inference engines for:"
echo "  1. PeopleNet (person detection)"
echo "  2. ResNet50 ReID (person re-identification)"
echo ""
echo "This process takes 10-20 minutes on Jetson Orin Nano."
echo "Engines are cached and only need to be built once."
echo "=============================================="
echo ""

PROJECT_ROOT="/home/nvidia/projects/inverse_point/ip_ai_analytics"

# Check if engines already exist
PEOPLENET_ENGINE="$PROJECT_ROOT/models/peoplenet/resnet34_peoplenet_int8.onnx_b1_gpu0_fp16.engine"
REID_ENGINE="$PROJECT_ROOT/models/tracker/resnet50_market1501.etlt_b32_gpu0_fp16.engine"

if [ -f "$PEOPLENET_ENGINE" ] && [ -f "$REID_ENGINE" ]; then
    echo "Both engines already exist!"
    echo "  PeopleNet: $PEOPLENET_ENGINE"
    echo "  ReID:      $REID_ENGINE"
    echo ""
    echo "To rebuild, delete the .engine files and run this script again."
    exit 0
fi

# ============================================
# Build PeopleNet Engine using trtexec
# ============================================
echo ""
echo "[1/2] Building PeopleNet TensorRT Engine..."
echo "----------------------------------------------"

PEOPLENET_ONNX="$PROJECT_ROOT/models/peoplenet/resnet34_peoplenet_int8.onnx"

if [ ! -f "$PEOPLENET_ONNX" ]; then
    echo "ERROR: PeopleNet ONNX model not found at $PEOPLENET_ONNX"
    exit 1
fi

if [ -f "$PEOPLENET_ENGINE" ]; then
    echo "PeopleNet engine already exists, skipping..."
else
    echo "Building PeopleNet engine (this takes ~5-10 minutes)..."
    echo "Input:  $PEOPLENET_ONNX"
    echo "Output: $PEOPLENET_ENGINE"
    echo ""

    /usr/src/tensorrt/bin/trtexec \
        --onnx="$PEOPLENET_ONNX" \
        --saveEngine="$PEOPLENET_ENGINE" \
        --fp16 \
        --memPoolSize=workspace:1024MiB \
        2>&1 | tee "$PROJECT_ROOT/output/logs/peoplenet_engine_build.log"

    if [ -f "$PEOPLENET_ENGINE" ]; then
        echo ""
        echo "✓ PeopleNet engine built successfully!"
        ls -lh "$PEOPLENET_ENGINE"
    else
        echo "ERROR: Failed to build PeopleNet engine"
        exit 1
    fi
fi

# ============================================
# Build ReID Engine using trtexec
# ============================================
echo ""
echo "[2/2] Building ReID TensorRT Engine..."
echo "----------------------------------------------"

REID_ETLT="$PROJECT_ROOT/models/tracker/resnet50_market1501.etlt"

if [ ! -f "$REID_ETLT" ]; then
    echo "ERROR: ReID ETLT model not found at $REID_ETLT"
    exit 1
fi

if [ -f "$REID_ENGINE" ]; then
    echo "ReID engine already exists, skipping..."
else
    echo "ReID engine will be built automatically by nvtracker on first run."
    echo "This is because ETLT models require the TAO decoder which is"
    echo "integrated into the DeepStream tracker plugin."
    echo ""
    echo "The engine was already built during previous test runs."
    echo "If you need to rebuild, delete: $REID_ENGINE"
fi

# ============================================
# Verify all engines
# ============================================
echo ""
echo "=============================================="
echo "Engine Build Summary"
echo "=============================================="

if [ -f "$PEOPLENET_ENGINE" ]; then
    echo "✓ PeopleNet Engine: $(ls -lh "$PEOPLENET_ENGINE" | awk '{print $5}')"
else
    echo "✗ PeopleNet Engine: NOT FOUND"
fi

if [ -f "$REID_ENGINE" ]; then
    echo "✓ ReID Engine:      $(ls -lh "$REID_ENGINE" | awk '{print $5}')"
else
    echo "⚠ ReID Engine:      Will be built on first pipeline run (~5 min)"
fi

echo ""
echo "=============================================="
echo "Engine building complete!"
echo "You can now run: ./run_test.sh"
echo "=============================================="
