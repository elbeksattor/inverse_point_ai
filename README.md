# IP AI v3 - Inverse Point AI Analytics System

**Version:** 0.4.1 (Phase 4 Complete + RE-ID Improvements)
**Platform:** NVIDIA Jetson Orin Nano / Orin NX
**Framework:** NVIDIA DeepStream SDK 7.1

---

## Current Status

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 1** | Camera + Person Detection | **APPROVED** |
| **Phase 2** | Tracking + OSNet Embeddings | **APPROVED** |
| **Phase 3** | Persistent RE-ID Database | **APPROVED** |
| **Phase 4** | Demographics Analysis | **APPROVED** |
| Phase 5 | Attention Detection | Planned |
| Phase 6 | Production Integration | Planned |

---

## Overview

Production-grade, GPU-accelerated AI analytics system for indoor advertising displays. Provides:

- **Real-time person detection and tracking** at 25-30 FPS
- **Persistent person re-identification (RE-ID)** across time gaps
- **Demographics analysis** (age groups, gender)
- **Attention detection** (head pose, engagement tracking)
- **Dwell time measurement** per person
- **Privacy-compliant** (no image storage, vector-only)
- **Multi-camera support** (up to 4 USB cameras)
- **JSON output** for system controller integration

---

## Quick Start

### Run with HDMI Display

```bash
cd /home/nvidia/projects/inverse_point/ip_ai_v3

# With display (requires HDMI monitor)
DISPLAY=:1 python3 src/main.py --camera /dev/video0

# Or use the shell script
./run_with_display.sh
```

### Run Headless (SSH)

```bash
python3 src/main.py --camera /dev/video0 --no-display

# Or use the shell script
./run_headless.sh
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--camera` | /dev/video0 | Camera device path |
| `--width` | 1280 | Frame width |
| `--height` | 720 | Frame height |
| `--fps` | 30 | Target framerate |
| `--no-display` | False | Run without display |
| `--tracker-config` | configs/tracker_config.txt | Tracker config |
| `--sgie-config` | configs/sgie_demographics_config.txt | Demographics SGIE config |
| `--no-database` | False | Disable RE-ID database |
| `--db-path` | data/person_database | Database path |
| `--reid-threshold` | 0.50 | RE-ID similarity threshold |
| `--log-level` | info | Logging level |

---

## Features (Phase 1-4)

### Person Detection (Phase 1)
- **Model:** PeopleNet (ResNet34 backbone)
- **Precision:** TensorRT FP16
- **Classes:** person, bag, face
- **Performance:** 25 FPS at 1280x720

### Multi-Object Tracking (Phase 2)
- **Tracker:** NvDeepSORT
- **RE-ID Model:** OSNet ResNet50 Market1501
- **Embedding Size:** 256 dimensions
- **Track persistence** across temporary occlusions

### Persistent RE-ID Database (Phase 3)
- **Vector Database:** FAISS for fast similarity search
- **Metadata Storage:** SQLite
- **Cross-session recognition** - Person IDs persist across restarts
- **Multi-shot gallery** - Up to 10 embeddings per person
- **Temporal boosting** - Better matching for recently seen persons

### Demographics Analysis (Phase 4)
- **Model:** InsightFace GenderAge (96x96 input)
- **Gender Detection:** Male/Female classification
- **Age Estimation:** Integer age (known limitation: may show constant ~35)
- **Age Groups:** 0-14, 15-19, 20-29, 30-44, 45-59, 60+
- **Face-Person Association:** Automatic via bbox overlap

### On-Screen Display
- Real-time statistics overlay
- Bounding boxes with labels
- Person IDs with demographics (e.g., `P1 M35 0.85`)
- Frame count, FPS, person count, elapsed time

### Camera Support
- USB cameras (MJPG format)
- Tested with Rapoo camera
- Resolution: 1280x720 @ 30fps

---

## Project Structure

```
ip_ai_v3/
├── configs/                              # Configuration files
│   ├── pipeline_config.yaml              # Main config
│   ├── pgie_config.txt                   # Detection model config
│   ├── tracker_config.txt                # Tracker config
│   ├── nvdeepsort_config.yml             # NvDeepSORT config
│   └── sgie_demographics_config.txt      # Demographics SGIE config
├── src/                                  # Source code
│   ├── main.py                           # Entry point
│   ├── pipeline/
│   │   └── camera_pipeline.py            # DeepStream pipeline
│   ├── database/
│   │   └── person_database.py            # FAISS + SQLite RE-ID
│   └── utils/
│       ├── logger.py                     # Structured logging
│       └── config.py                     # Configuration
├── models/
│   ├── peoplenet/                        # Detection model
│   ├── tracker/                          # OSNet RE-ID model
│   └── demographics/                     # GenderAge model
├── data/                                 # Runtime data
│   └── person_database/                  # RE-ID database files
├── output/                               # Output files
├── docs/                                 # Documentation
│   ├── PROFESSIONAL_IMPLEMENTATION_PLAN.md
│   ├── PHASE_1_COMPLETE.md
│   ├── PHASE_2_COMPLETE.md
│   ├── PHASE_3_COMPLETE.md
│   ├── PHASE_4_COMPLETE.md
│   └── CHANGELOG.md
├── run_with_display.sh                   # Run with HDMI display
├── run_headless.sh                       # Run without display
├── requirements.txt
└── README.md
```

---

## Requirements

### Hardware
- NVIDIA Jetson Orin Nano (8GB) or Orin NX (16GB)
- USB camera (MJPG supported)
- HDMI monitor (optional, for display)

### Software
- JetPack 5.x
- DeepStream SDK 7.1
- CUDA 12.6
- TensorRT 10.3
- Python 3.10+
- pyds 1.2.0

### Python Dependencies
```
numpy>=1.21.0
opencv-python>=4.5.0
pyyaml>=5.4.0
structlog>=21.5.0
faiss-cpu>=1.7.0
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [Implementation Plan](docs/PROFESSIONAL_IMPLEMENTATION_PLAN.md) | Full 6-phase development plan |
| [Phase 1 Complete](docs/PHASE_1_COMPLETE.md) | Camera + Detection |
| [Phase 2 Complete](docs/PHASE_2_COMPLETE.md) | Tracking + OSNet |
| [Phase 3 Complete](docs/PHASE_3_COMPLETE.md) | RE-ID Database |
| [Phase 4 Complete](docs/PHASE_4_COMPLETE.md) | Demographics Analysis |
| [Changelog](docs/CHANGELOG.md) | Version history |

---

## Performance

| Configuration | FPS | GPU Memory |
|---------------|-----|------------|
| Detection only | 25 FPS | ~350MB |
| + Tracking | 20 FPS | ~400MB |
| + RE-ID Database | 20 FPS | ~420MB |
| + Demographics | 20 FPS | ~430MB |

**Resolution:** 1280x720
**Latency:** <40ms

---

## License

Proprietary - Inverse Point Project

---

**Last Updated:** 2026-01-31
