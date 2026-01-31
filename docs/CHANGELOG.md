# IP AI v3 - Changelog

All notable changes to this project will be documented in this file.

---

## [0.1.0] - 2026-01-30

### Phase 1: Camera + Person Detection - APPROVED

#### Added
- DeepStream pipeline for USB camera input (`src/pipeline/camera_pipeline.py`)
- PeopleNet person detection with TensorRT FP16 acceleration
- MJPG camera support via jpegdec element
- Real-time statistics overlay (Frame, FPS, Persons, Total, Time)
- HDMI display output via nveglglessink
- Command-line interface (`src/main.py`)
- Configuration management (`src/utils/config.py`)
- Structured logging with structlog (`src/utils/logger.py`)
- Person database foundation (`src/database/person_database.py`)
- Shell script for easy execution (`run_camera.sh`)

#### Configuration Files
- `configs/pgie_config.txt` - PeopleNet detection config
- `configs/tracker_config.txt` - NvDCF tracker config (prepared for Phase 2)
- `configs/nvdcf_config.yml` - Tracker details (prepared for Phase 2)
- `configs/pipeline_config.yaml` - Main configuration

#### Models
- PeopleNet ResNet34 (ONNX + TensorRT FP16 engine)
- Labels: person, bag, face

#### Performance
- **FPS:** 25 FPS at 1280x720
- **Latency:** <40ms per frame
- **GPU Utilization:** ~50-60%

#### Documentation
- `docs/PROFESSIONAL_IMPLEMENTATION_PLAN.md` - Full implementation plan
- `docs/PHASE_1_COMPLETE.md` - Phase 1 completion report
- `docs/CHANGELOG.md` - This file

---

## [0.2.0] - 2026-01-30

### Phase 2: Tracking + OSNet RE-ID - APPROVED

#### Added
- NvDeepSORT multi-object tracker integration
- OSNet RE-ID model (ResNet50 Market1501) with 256-dim embeddings
- Track ID assignment and persistence across frames
- Track ID display on OSD (format: `#ID class confidence`)
- Console logging of active track IDs
- `configs/nvdeepsort_config.yml` - NvDeepSORT configuration

#### Modified
- `tracker_config.txt` - Updated to use NvDeepSORT with absolute paths
- `camera_pipeline.py` - Added tracker configuration, track display, embedding extraction
- `main.py` - Added track ID logging in frame callback

#### Models Added
- `models/tracker/resnet50_market1501.etlt` - RE-ID ETLT model
- `models/tracker/resnet50_market1501.etlt_b32_gpu0_fp16.engine` - TensorRT engine

#### Performance
- **FPS with Tracking:** 16.7 FPS at 1280x720
- **Embedding Size:** 256 dimensions
- **Track Probation:** 3 frames before ID assigned
- **Shadow Tracking:** 60 frames for lost tracks

#### Documentation
- `docs/PHASE_2_COMPLETE.md` - Phase 2 completion report

---

## [0.3.0] - 2026-01-30

### Phase 3: Persistent RE-ID Database - APPROVED

#### Added
- RE-ID embedding extraction from NvDeepSORT tracker (256-dim)
- FAISS vector database for fast cosine similarity search
- SQLite metadata storage for person information
- Cross-session person recognition (Person IDs persist across restarts)
- Moving average embedding updates (70% old + 30% new)
- Person ID display on OSD (`P{id} #{track} {conf}`)
- Tracker ID to Person ID caching for fast lookup

#### Modified
- `person_database.py` - Updated embedding dimension to 256
- `camera_pipeline.py` - Added RE-ID extraction, database integration
- `main.py` - Added database initialization, statistics, cleanup
- `nvdeepsort_config.yml` - Added `outputReidTensor: 1`

#### Configuration
- `--no-database` - Disable RE-ID database
- `--db-path` - Custom database path
- `--reid-threshold` - Cosine similarity threshold (default: 0.70)

#### Performance
- **FPS with RE-ID:** 19.9 FPS at 1280x720
- **Embedding Size:** 256 dimensions
- **Database Size:** ~50KB per 100 persons
- **Matching Latency:** <1ms (FAISS)

#### Documentation
- `docs/PHASE_3_COMPLETE.md` - Phase 3 completion report

---

## [0.4.0] - 2026-01-31

### Phase 4: Demographics Analysis - APPROVED

#### Added
- InsightFace GenderAge SGIE model integration
- Tensor metadata extraction using DeepStream 7.1 API (`pyds.NVDSINFER_TENSOR_OUTPUT_META`)
- Age/gender classification from face detections
- Face-to-person association via bounding box overlap calculation
- Demographics caching per Person ID (persists across frames)
- OSD display with demographics format: `P{id} {gender}{age} {confidence}`
- Console output with demographics info: `Demo: M35`
- Age group categorization (0-14, 15-19, 20-29, 30-44, 45-59, 60+)

#### Configuration Files
- `configs/sgie_demographics_config.txt` - SGIE configuration for GenderAge model
- `configs/demographics_labels.txt` - Label file

#### Models Added
- `models/demographics/genderage.onnx` - InsightFace GenderAge ONNX model
- `models/demographics/genderage.onnx_b1_gpu0_fp16.engine` - TensorRT FP16 engine

#### Modified
- `camera_pipeline.py` - Added SGIE config, tensor parsing, two-pass object processing, demographics cache
- `main.py` - Added demographics console output
- `pgie_config.txt` - Increased detection thresholds for better accuracy

#### Technical Details
- Model input: 96x96 RGB, normalized (x - 127.5) / 127.5
- fc1 output: [gender_logit, -gender_logit, age/100]
- Gender: positive logit = Male, negative = Female
- Face-person association: >50% bbox overlap required

#### Performance
- **FPS with Demographics:** 19.2 FPS at 1280x720
- **Memory:** ~430MB
- **GPU Utilization:** ~55%

#### Documentation
- `docs/PHASE_4_COMPLETE.md` - Phase 4 completion report

---

## [0.4.1] - 2026-01-31

### RE-ID Stability Improvements - IMPLEMENTED

#### Added
- **Multi-shot Gallery Storage** - Store up to 10 diverse embeddings per person instead of single averaged embedding
- **Gallery-based Matching** - Match against all gallery embeddings, use maximum similarity
- **Temporal Boost** - 0.15 threshold reduction for persons seen within last 10 seconds
- **Embedding Diversity Filter** - Only add embeddings >0.85 different from existing gallery
- **Gallery Statistics** - `get_gallery_stats()` method for monitoring

#### Modified
- `person_database.py` - Complete rewrite of matching algorithm for multi-shot support
  - `_match_against_gallery()` - Match against person's gallery embeddings
  - `_add_to_gallery()` - Add diverse embeddings to gallery
  - `match_or_create_person()` - Gallery-based matching with temporal boost
  - `save_index()` / `_load_index()` - Persist/load gallery data
  - `cleanup_old_persons()` - Clean up gallery for deleted persons

#### Configuration
```python
DEFAULT_REID_THRESHOLD = 0.50   # Lowered from 0.70 for ResNet50 model
MAX_EMBEDDINGS_PER_PERSON = 10   # Gallery size per person
EMBEDDING_DIVERSITY_THRESHOLD = 0.80  # Diversity filter (lowered for more gallery diversity)
RECENT_SEEN_BOOST_SECONDS = 15.0  # Temporal boost window (extended)
RECENT_SEEN_THRESHOLD_REDUCTION = 0.20  # Threshold reduction (increased)
```

#### Key Changes
- **Threshold lowered to 0.50** - ResNet50 Market1501 produces 0.50-0.65 similarity for same person in different poses
- **Temporal boost extended** - 15 seconds window with 0.20 reduction (effective threshold: 0.30)
- **Diversity threshold lowered** - Allows more pose variations in gallery

#### Expected Improvements
- Better matching when person's pose changes
- Reduced false positive new person creation
- Improved re-identification when person returns after leaving frame
- Gallery grows to capture different viewing angles

---

## [0.4.2] - 2026-01-31

### Demographics Age Estimation Investigation

#### Investigated
- **Age estimation limitation** - Analyzed InsightFace GenderAge model architecture
  - Age branch has bias ~0.32 with very small weights (std 0.0223)
  - Model requires aligned face images using 5-point facial landmarks
  - DeepStream face crops are not aligned, causing constant ~35 output

#### Attempted (Reverted)
- **Custom face alignment with YuNet**
  - Downloaded YuNet face detector for landmark detection
  - Implemented affine transformation for face alignment
  - Result: No improvement in age accuracy, significant FPS drop
  - Decision: Reverted to original SGIE-based approach

#### Final State
- Demographics SGIE enabled and working (~20 FPS)
- Gender detection works correctly
- Age shows ~35 for everyone (documented known limitation)
- System prioritizes performance over age accuracy

#### Documentation
- Updated README.md to reflect Phase 4 completion
- Updated PHASE_4_COMPLETE.md with detailed limitation notes
- Added run scripts for testing

---

## Upcoming

### [0.5.0] - Phase 5: Attention Detection (Planned)
- Head pose estimation (6DRepNet)
- Attention state machine
- Qualified impressions counting
- Dwell time calculation

### [1.0.0] - Phase 6: Production Release (Planned)
- JSON output for system controller
- Multi-camera support
- Systemd service integration
- Production optimization

---

## Version History

| Version | Phase | Status | Date |
|---------|-------|--------|------|
| 0.1.0 | Phase 1: Camera + Detection | APPROVED | 2026-01-30 |
| 0.2.0 | Phase 2: Tracking + OSNet | APPROVED | 2026-01-30 |
| 0.3.0 | Phase 3: RE-ID Database | APPROVED | 2026-01-30 |
| 0.4.0 | Phase 4: Demographics | APPROVED | 2026-01-31 |
| 0.4.1 | RE-ID Stability | IMPLEMENTED | 2026-01-31 |
| 0.4.2 | Age Investigation | DOCUMENTED | 2026-01-31 |
| 0.5.0 | Phase 5: Attention | Planned | - |
| 1.0.0 | Phase 6: Production | Planned | - |
