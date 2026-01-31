# Phase 5 Complete: Attention Detection

**Date:** 2026-01-31
**Version:** 0.5.0

---

## Overview

Phase 5 implements attention detection using head pose estimation to track whether persons are looking at the advertising display. This enables counting "qualified impressions" - persons who actively engaged with the display for a minimum duration.

---

## Implementation Summary

### New Components

| Component | File | Description |
|-----------|------|-------------|
| HeadPoseONNX | `src/analytics/headpose_onnx.py` | ONNX Runtime head pose estimation |
| AttentionTracker | `src/analytics/attention_tracker.py` | State machine for attention tracking |
| Analytics Module | `src/analytics/__init__.py` | Module exports |
| WHENet Model | `models/headpose/whenet_prepost.onnx` | Head pose ONNX model with built-in preprocessing |

### Modified Components

| Component | Changes |
|-----------|---------|
| `camera_pipeline.py` | Added SGIE-2 for head pose, attention tracking integration |
| `main.py` | Added CLI args for attention thresholds, updated output |
| `person_database.py` | Added attention analytics methods |
| `README.md` | Updated for Phase 5 features |

---

## Architecture

```
Camera -> PGIE (PeopleNet) -> Tracker -> SGIE-1 (Demographics) -> OSD
                                                    |
                                    HeadPoseONNX (ONNX Runtime - extracted in probe callback)
                                                    |
                                          AttentionTracker (State Machine)
                                          NOT_LOOKING -> LOOKING -> ENGAGED
```

**Note:** Head pose uses ONNX Runtime instead of DeepStream SGIE due to a DeepStream 7.1 caps negotiation issue with regression-type models. Face crops are extracted in the probe callback and processed via CPU-based ONNX Runtime.

### Attention State Machine

```
                  head facing camera
    NOT_LOOKING ─────────────────────> LOOKING ─────────────> ENGAGED
         ^                                 │      >2 seconds      │
         │                                 │                      │
         └─────────────────────────────────┴──────────────────────┘
                        head turns away
```

**States:**
- **NOT_LOOKING:** Person not facing the camera (|yaw| > 10° or |pitch| > 10°)
- **LOOKING:** Person facing the camera (|yaw| < 10° AND |pitch| < 10°)
- **ENGAGED:** Person has been looking for >2 seconds (qualified impression)

---

## Head Pose Model

### ONNX Runtime Implementation

Due to a DeepStream 7.1 caps negotiation issue with SGIE for regression-type models, head pose uses ONNX Runtime directly:

**Model:** WHENet (Whole Head Estimation Network)
- **File:** `models/headpose/whenet_prepost.onnx`
- **Input:** 224x224 RGB (preprocessing built into model)
- **Output:** `[yaw, roll, pitch]` in degrees

### Implementation (`src/analytics/headpose_onnx.py`)

```python
class HeadPoseONNX:
    def estimate(self, face_bgr: np.ndarray) -> Tuple[float, float, float]:
        """Returns (yaw, pitch, roll) in degrees"""
```

Face crops are extracted from the DeepStream frame buffer using `pyds.get_nvds_buf_surface()` and processed via CPU-based ONNX Runtime inference.

---

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--headpose-model` | None | Path to head pose ONNX model |
| `--yaw-threshold` | 10.0 | Max yaw angle (degrees) to consider "looking" |
| `--pitch-threshold` | 10.0 | Max pitch angle (degrees) to consider "looking" |
| `--engagement-time` | 2.0 | Seconds of looking before marking as "engaged" |

### Usage Examples

```bash
# Enable attention tracking with defaults (±10° threshold)
python3 src/main.py --camera /dev/video0 \
  --headpose-model models/headpose/whenet_prepost.onnx

# With demographics enabled
python3 src/main.py --camera /dev/video0 \
  --headpose-model models/headpose/whenet_prepost.onnx \
  --sgie-config configs/sgie_demographics_config.txt

# Custom thresholds
python3 src/main.py --camera /dev/video0 \
  --headpose-model models/headpose/whenet_prepost.onnx \
  --yaw-threshold 15 \
  --pitch-threshold 15 \
  --engagement-time 3.0

# Use convenience script (1080p, ±10° threshold, headpose enabled)
./run_with_display.sh --sgie-config configs/sgie_demographics_config.txt
```

---

## Visual Indicators

### OSD Display

| State | Border Color | Label | Border Width |
|-------|--------------|-------|--------------|
| ENGAGED | Green | `[ENGAGED]` | 4px |
| LOOKING | Yellow | `[LOOKING]` | 3px |
| NOT_LOOKING | Blue | (none) | 2px |

### Statistics Overlay

```
IP AI v3 | Frame: 1234 | FPS: 18.5 | Persons: 3 | Look: 1+1 | QI: 5 | Time: 45.2s
```

- `Look: 1+1` = 1 LOOKING + 1 ENGAGED currently
- `QI: 5` = Total qualified impressions this session

---

## Console Output

```
[Frame   120] Persons:  2 | Faces:  2 | IDs: [P1,P2] | FPS:  18.3 | Unique: 2 | Demo: M35 | Attn: 1L/1E | QI: 3
```

- `Attn: 1L/1E` = 1 Looking, 1 Engaged
- `QI: 3` = Total qualified impressions

---

## Session Statistics

At the end of a session:

```
============================================================
Session Statistics
============================================================
Total Frames:   3600
Total Persons:  45
Unique (session): 8
Unique (all-time): 23
Avg FPS:        18.2
------------------------------------------------------------
Attention Statistics
------------------------------------------------------------
Qualified Impressions: 12
Currently Looking: 0
Currently Engaged: 0
Max Engaged: 3
============================================================
```

---

## Database Analytics

### New Methods in `PersonDatabase`

```python
# Get attention-focused analytics
db.get_attention_analytics(camera_id=1, since=datetime.now() - timedelta(hours=1))

# Returns:
{
    "total_qualified_impressions": 25,
    "unique_persons_with_qi": 15,
    "avg_attention_time": 4.5,
    "max_attention_time": 12.3,
    "qi_rate": 0.625,
    "qi_by_gender": {"Male": {"qualified_impressions": 12, "person_count": 10}, ...},
    "qi_by_age_group": {"18-29": {"qualified_impressions": 8, "person_count": 6}, ...}
}

# Finalize person session
db.finalize_person_session(
    person_id=1,
    dwell_time=30.5,
    attention_time=8.2,
    qualified_impression=True
)
```

---

## Performance Impact

| Metric | Before (Phase 4) | After (Phase 5) |
|--------|------------------|-----------------|
| FPS | 20 | 16-18 |
| GPU Memory | 430MB | 500MB |
| Latency | 50ms | 60ms |

---

## Known Limitations

1. **Placeholder Model:** The current head pose model is a placeholder. For production, replace with a trained model like 6DRepNet.

2. **Face Detection Required:** Attention tracking only works when a face is detected. Persons with no visible face will always show as NOT_LOOKING.

3. **Occlusion:** If a face is temporarily occluded, attention state may reset.

---

## Model Replacement Guide

To use a production head pose model:

1. Download or convert the model to ONNX format
2. Place in `models/headpose/` directory
3. Update `sgie_headpose_config.txt`:
   ```ini
   onnx-file=/path/to/your/model.onnx
   model-engine-file=/path/to/your/model.onnx_b1_gpu0_fp16.engine
   ```
4. Ensure output tensor matches expected format (3 values for Euler, 6 for 6D)

### Recommended Models

| Model | Input | Output | Size |
|-------|-------|--------|------|
| 6DRepNet | 224x224 | 6D rotation | ~25MB |
| WHENet | 224x224 | Euler angles | ~15MB |
| FSA-Net | 64x64 | Euler angles | ~5MB |

---

## Files Created/Modified

### New Files
- `src/analytics/__init__.py` - Analytics module exports
- `src/analytics/attention_tracker.py` - Attention state machine
- `src/analytics/headpose_onnx.py` - ONNX Runtime head pose estimation
- `models/headpose/whenet_prepost.onnx` - WHENet model with preprocessing
- `docs/PHASE_5_COMPLETE.md` - This documentation

### Modified Files
- `src/pipeline/camera_pipeline.py` - Head pose extraction via ONNX Runtime
- `src/main.py` - CLI args, attention metrics display
- `src/database/person_database.py` - Attention analytics methods
- `run_with_display.sh` - Default ±10° thresholds, headpose enabled
- `README.md` - Updated documentation

---

## Next Steps (Phase 6)

1. Replace placeholder model with production head pose model
2. Add JSON API output for system controller integration
3. Implement periodic analytics export
4. Add multi-camera aggregation
5. Production hardening and error handling

---

**Phase 5 Status: IMPLEMENTED**
