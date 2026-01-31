# Phase 1 Complete: Camera + Person Detection

**Project:** IP AI v3 - Inverse Point AI Analytics System
**Phase:** 1 - Foundation (Camera + Detection)
**Status:** APPROVED
**Date:** 2026-01-30
**Platform:** NVIDIA Jetson Orin Nano

---

## Executive Summary

Phase 1 successfully implemented a GPU-accelerated person detection pipeline using NVIDIA DeepStream SDK 7.1. The system captures video from a USB camera, detects persons in real-time using PeopleNet, and displays results with on-screen statistics overlay.

### Key Achievements

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| **FPS** | 25-30 | **25 FPS** | PASS |
| **Person Detection** | Working | **Confirmed** | PASS |
| **USB Camera Support** | MJPG | **Working** | PASS |
| **GPU Acceleration** | TensorRT | **FP16 Engine** | PASS |
| **Statistics Overlay** | OSD | **Working** | PASS |

---

## Architecture

### Pipeline Structure

```
USB Camera (MJPG)
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│              DeepStream Pipeline (GPU)                   │
├─────────────────────────────────────────────────────────┤
│  v4l2src → capsfilter → jpegdec → nvvideoconvert       │
│       │                                                  │
│       ▼                                                  │
│  nvstreammux → nvinfer (PeopleNet) → nvvideoconvert    │
│       │                                                  │
│       ▼                                                  │
│  nvdsosd (Statistics + Bboxes) → nveglglessink         │
└─────────────────────────────────────────────────────────┘
       │
       ▼
   HDMI Display
```

### Components

| Component | Implementation | Purpose |
|-----------|----------------|---------|
| **Video Source** | v4l2src + jpegdec | USB camera MJPG decode |
| **GPU Convert** | nvvideoconvert | CPU to GPU memory transfer |
| **Stream Mux** | nvstreammux | Batch frames for inference |
| **Detection** | nvinfer (PeopleNet) | Person/bag/face detection |
| **Display** | nvdsosd + nveglglessink | OSD overlay + HDMI output |

---

## Implementation Details

### Files Created

```
ip_ai_v3/
├── configs/
│   ├── pgie_config.txt           # PeopleNet detection config
│   ├── tracker_config.txt        # NvDCF tracker config (Phase 2)
│   ├── nvdcf_config.yml          # Tracker details (Phase 2)
│   └── pipeline_config.yaml      # Main configuration
├── src/
│   ├── __init__.py
│   ├── main.py                   # Entry point (166 lines)
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── camera_pipeline.py    # DeepStream pipeline (602 lines)
│   ├── database/
│   │   ├── __init__.py
│   │   └── person_database.py    # FAISS+SQLite (580 lines)
│   └── utils/
│       ├── __init__.py
│       ├── logger.py             # Structured logging
│       └── config.py             # Configuration management
├── models/
│   └── peoplenet/
│       ├── resnet34_peoplenet_int8.onnx
│       ├── resnet34_peoplenet_int8.onnx_b1_gpu0_fp16.engine
│       └── labels.txt
├── docs/
│   ├── PROFESSIONAL_IMPLEMENTATION_PLAN.md
│   └── PHASE_1_COMPLETE.md       # This document
├── run_camera.sh                 # Easy run script
├── requirements.txt
└── README.md
```

### Key Code: Camera Pipeline

**File:** `src/pipeline/camera_pipeline.py`

```python
class CameraPipeline:
    """
    DeepStream pipeline for USB camera with person detection.

    Features:
    - MJPG USB camera support
    - GPU-accelerated inference (TensorRT FP16)
    - Real-time statistics overlay
    - Callback system for frame metadata
    """

    # Pipeline: v4l2src → jpegdec → nvvideoconvert → nvstreammux →
    #           nvinfer → nvvideoconvert → nvdsosd → display
```

### Key Code: Statistics Overlay

```python
# Statistics displayed on every frame
stats_text = (
    f"IP AI v3 | Frame: {frame_num:6d} | "
    f"FPS: {fps:5.1f} | "
    f"Persons: {person_count:2d} | "
    f"Total: {total_frames:6d} | "
    f"Time: {elapsed:6.1f}s"
)

# Yellow text on semi-transparent black background
py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 0.0, 1.0)
py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.7)
```

---

## Test Results

### Performance Metrics

| Test | Duration | Frames | Avg FPS | Persons Detected |
|------|----------|--------|---------|------------------|
| Test 1 | 15s | 375 | 25.0 | 1 per frame |
| Test 2 | 60s | 1500 | 25.1 | 0-2 per frame |

### Camera Specifications

| Property | Value |
|----------|-------|
| Device | /dev/video0 (Rapoo camera) |
| Format | MJPG (Motion JPEG) |
| Resolution | 1280x720 |
| Frame Rate | 30 FPS input, 25 FPS output |

### Detection Model

| Property | Value |
|----------|-------|
| Model | PeopleNet (ResNet34 backbone) |
| Source | NVIDIA TAO Toolkit |
| Precision | FP16 (TensorRT optimized) |
| Classes | person, bag, face |
| Confidence Threshold | 0.4 |

---

## How to Run

### Basic Usage

```bash
cd /home/nvidia/projects/inverse_point/ip_ai_v3

# With HDMI display
DISPLAY=:1 python3 src/main.py --camera /dev/video0

# Headless (no display)
python3 src/main.py --camera /dev/video0 --no-display

# Using shell script
./run_camera.sh
./run_camera.sh --no-display
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--camera` | /dev/video0 | Camera device path |
| `--width` | 1280 | Frame width |
| `--height` | 720 | Frame height |
| `--fps` | 30 | Target framerate |
| `--no-display` | False | Run without display |
| `--tracker-config` | configs/tracker_config.txt | Tracker config (use 'none' to disable) |
| `--log-level` | info | Logging level |

---

## Issues Resolved

### 1. MJPG Camera Support

**Problem:** USB camera outputs MJPG format, not raw video.

**Solution:** Added `jpegdec` element to pipeline:
```
v4l2src → capsfilter (image/jpeg) → jpegdec → nvvideoconvert
```

### 2. Model Path Resolution

**Problem:** DeepStream resolved model paths relative to config file location.

**Solution:** Used absolute paths in `pgie_config.txt`:
```
model-engine-file=/home/nvidia/projects/inverse_point/ip_ai_v3/models/peoplenet/...
```

### 3. GstRtspServer Import Error

**Problem:** `GstRtspServer` not available on system.

**Solution:** Removed unused import (RTSP streaming not needed for Phase 1).

---

## Lessons Learned

1. **MJPG is common for USB cameras** - Always check camera format with `v4l2-ctl --list-formats-ext`

2. **DeepStream paths** - Use absolute paths in config files to avoid resolution issues

3. **pyds display_meta** - Use `nvds_acquire_display_meta_from_pool()` for OSD text overlay

4. **FPS calculation** - Sample every 30 frames for smooth FPS display without overhead

---

## Next Phase: Tracking + OSNet

Phase 2 will add:

1. **NvDCF Multi-Object Tracker**
   - Consistent track IDs across frames
   - Re-association after occlusion

2. **OSNet Embedding Extraction**
   - 128-dim embeddings from tracker
   - Enable `enable-past-frame=1` in tracker config

3. **Track ID Display**
   - Show persistent track IDs on OSD
   - Foundation for RE-ID database

### Estimated Effort

| Task | Effort |
|------|--------|
| Enable NvDCF tracker | 1 hour |
| Extract OSNet embeddings | 2-3 hours |
| Integrate with display | 1 hour |
| Testing & validation | 2 hours |

---

## Approval

**Phase 1 Status:** APPROVED

**Approved Features:**
- [x] USB camera input (MJPG)
- [x] Person detection (PeopleNet)
- [x] GPU acceleration (TensorRT FP16)
- [x] Real-time display (HDMI)
- [x] Statistics overlay (Frame, FPS, Persons, Total, Time)
- [x] 25 FPS performance

**Ready for Phase 2:** Yes

---

**Document Version:** 1.0
**Last Updated:** 2026-01-30
**Author:** Claude AI Assistant
