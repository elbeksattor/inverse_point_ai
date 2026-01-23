# IP AI Analytics - Implementation Plan
**Project:** Inverse Point AI Analytics Subsystem v2.0
**Platform:** NVIDIA Jetson Orin Nano
**Duration:** 8 weeks (target)
**Date:** 2026-01-19

---

## Executive Summary

This document outlines the **8-week implementation roadmap** for building the AI Analytics Subsystem from scratch. The plan is divided into 4 major phases, with incremental deliverables and testing at each stage.

### Timeline Overview

```
Week 1-2: Phase 1 - Core Pipeline
Week 3-4: Phase 2 - RE-ID Database
Week 5-6: Phase 3 - Analytics & Demographics
Week 7-8: Phase 4 - Integration & Optimization
```

---

## Phase 1: Core DeepStream Pipeline (Weeks 1-2)

**Goal:** Establish working DeepStream pipeline with person detection, tracking, and basic visualization.

### Week 1: Pipeline Foundation

#### Day 1-2: Environment Setup
**Tasks:**
- [x] Verify Jetson Orin Nano setup (JetPack, DeepStream, CUDA)
- [x] Install Python dependencies (see requirements.txt below)
- [x] Create project structure (directories, initial files)
- [x] Download PeopleNet model from NGC
- [x] Test basic DeepStream sample (deepstream-test1)

**Deliverables:**
- Development environment ready
- PeopleNet model downloaded and verified
- Project structure initialized

**Testing:**
- Run `deepstream-app --version` (should show 7.1.0)
- Run PeopleNet sample successfully

---

#### Day 3-4: Basic Detection Pipeline
**Tasks:**
- [ ] Implement `deepstream_pipeline.py` (basic pipeline class)
- [ ] Configure PeopleNet as Primary GIE (PGIE)
- [ ] Set up video file input (filesrc → decoder → PGIE)
- [ ] Implement basic frame metadata extraction
- [ ] Add simple OSD overlay (bounding boxes)

**Code Structure:**
```python
# src/pipeline/deepstream_pipeline.py
class DeepStreamPipeline:
    def __init__(self, config_path):
        self.pipeline = None
        self.config = load_config(config_path)

    def build_pipeline(self):
        # Create GStreamer pipeline
        # filesrc → h264parse → nvv4l2decoder → nvstreammux → nvinfer (PeopleNet) → nvvideoconvert → nvdsosd → nvvideoconvert → nvv4l2h264enc → filesink
        pass

    def add_probe(self, element, callback):
        # Add probe for metadata extraction
        pass

    def run(self):
        # Start pipeline
        pass
```

**Deliverables:**
- Working detection pipeline (video file → person detection → output video)
- Bounding boxes drawn on detected persons
- FPS counter displayed

**Testing:**
- Test with sample video (should detect persons, draw boxes)
- Verify FPS ≥ 30 on 1080p video

---

#### Day 5-7: Multi-Object Tracking
**Tasks:**
- [ ] Add NvDCF tracker to pipeline (after PGIE)
- [ ] Configure tracker with OSNet for RE-ID embeddings
- [ ] Extract tracker metadata (track IDs)
- [ ] Implement probe to access `NVDS_TRACKER_PAST_FRAME_META`
- [ ] Extract OSNet embeddings (128-dim float32 array)
- [ ] Display track IDs on OSD

**Tracker Configuration:**
```ini
# configs/tracker_config.txt
[tracker]
tracker-width=640
tracker-height=384
ll-lib-file=/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so
ll-config-file=configs/tracker_nvdcf_config.txt
enable-batch-process=1
enable-past-frame=1

[nvdcf]
gpu-id=0
feature-dim=128
search-scale=1.5
min-iou-diff=0.5
```

**Deliverables:**
- Tracking working (objects maintain consistent IDs across frames)
- OSNet embeddings successfully extracted
- Track IDs displayed on OSD

**Testing:**
- Verify track IDs are consistent (person keeps same ID across frames)
- Verify embeddings are 128-dim float32 arrays
- Test with occlusions and re-entries

---

### Week 2: Multi-Camera & Optimization

#### Day 8-10: Multi-Stream Support
**Tasks:**
- [ ] Extend pipeline to support multiple inputs (nvstreammux)
- [ ] Test with 2 video files simultaneously
- [ ] Implement per-stream metadata handling
- [ ] Add camera/stream ID to metadata
- [ ] Test FPS with 2 streams @ 1080p

**Pipeline Architecture:**
```
Video1 → Decoder → \
                    nvstreammux (batch=2) → PGIE → Tracker → OSD → Output
Video2 → Decoder → /
```

**Deliverables:**
- Multi-stream pipeline working (2+ cameras)
- Per-stream OSD rendering
- FPS ≥ 30 per stream (or adjust resolution)

**Testing:**
- Test with 2, 3, 4 video files
- Monitor GPU/memory usage
- Verify no frame drops

---

#### Day 11-12: Performance Optimization
**Tasks:**
- [ ] Profile pipeline (identify bottlenecks)
- [ ] Enable TensorRT FP16 optimization for PeopleNet
- [ ] Optimize streammux settings (batch size, buffer pool)
- [ ] Reduce inference resolution if needed
- [ ] Implement performance monitoring (FPS, GPU, memory)

**Deliverables:**
- Pipeline running at target FPS (30 FPS per stream)
- Performance metrics logged
- GPU utilization 70-80%

**Testing:**
- Benchmark with 1, 2, 4 streams
- Verify stable FPS over 10-minute run

---

#### Day 13-14: USB Camera Support
**Tasks:**
- [ ] Replace filesrc with v4l2src (USB cameras)
- [ ] Test with actual USB cameras on Jetson
- [ ] Handle camera initialization and errors
- [ ] Auto-detect available cameras (/dev/video*)
- [ ] Test camera hot-plug scenarios

**Pipeline Change:**
```python
# Before: filesrc → h264parse → nvv4l2decoder
# After:  v4l2src device=/dev/video0 → nvvideoconvert → nvstreammux
```

**Deliverables:**
- USB camera input working
- Multi-camera support (2-4 USB cameras)
- Error handling for missing cameras

**Testing:**
- Test with 1, 2, 4 USB cameras
- Verify FPS with real cameras
- Test camera disconnect/reconnect

---

**Phase 1 Milestone:**
✅ DeepStream pipeline with detection, tracking, multi-camera support
✅ OSNet embeddings extracted
✅ 30 FPS achieved on target hardware
✅ USB camera support verified

---

## Phase 2: Persistent RE-ID Database (Weeks 3-4)

**Goal:** Implement FAISS + SQLite database for cross-session person re-identification.

### Week 3: Database Foundation

#### Day 15-17: FAISS Vector Index
**Tasks:**
- [ ] Install FAISS with GPU support (faiss-gpu)
- [ ] Implement `faiss_index.py` (FAISS wrapper class)
- [ ] Test FAISS IndexFlatIP on GPU
- [ ] Implement embedding normalization (for cosine similarity)
- [ ] Implement top-K nearest neighbor search
- [ ] Test save/load index to disk

**Code Structure:**
```python
# src/database/faiss_index.py
class FAISSIndex:
    def __init__(self, dim=128, use_gpu=True):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # Inner product (cosine similarity)
        if use_gpu:
            self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)
        self.id_map = []  # Maps FAISS index → person_id

    def add(self, embeddings, person_ids):
        # Normalize embeddings
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.index.add(embeddings.astype('float32'))
        self.id_map.extend(person_ids)

    def search(self, embedding, k=5):
        embedding = embedding / np.linalg.norm(embedding)
        similarities, indices = self.index.search(embedding.reshape(1, -1), k)
        return [(self.id_map[idx], sim) for idx, sim in zip(indices[0], similarities[0])]

    def save(self, path):
        faiss.write_index(faiss.index_gpu_to_cpu(self.index), path)

    def load(self, path):
        self.index = faiss.read_index(path)
        if use_gpu:
            self.index = faiss.index_cpu_to_gpu(...)
```

**Deliverables:**
- FAISS index working on GPU
- Search latency <1ms
- Save/load persistence working

**Testing:**
- Add 100 random embeddings, search for nearest neighbors
- Verify cosine similarity scores
- Test persistence (save → reload → search)

---

#### Day 18-19: SQLite Metadata Store
**Tasks:**
- [ ] Implement `sqlite_store.py` (SQLite wrapper)
- [ ] Create database schema (persons, appearances tables)
- [ ] Implement CRUD operations for persons
- [ ] Implement appearance logging
- [ ] Add indexes for performance

**Deliverables:**
- SQLite database operational
- Person CRUD working
- Appearance logging working

**Testing:**
- Insert 100 persons, query by ID
- Log 1000 appearances, query by person_id
- Verify database integrity

---

#### Day 20-21: Integrated Person Database
**Tasks:**
- [ ] Implement `person_database.py` (combines FAISS + SQLite)
- [ ] Implement `match_or_create_person()` method
- [ ] Implement embedding moving average update
- [ ] Add 30-day retention policy (auto-delete old persons)
- [ ] Implement database statistics queries

**Algorithm:**
```python
def match_or_create_person(embedding, threshold=0.75):
    """
    1. Normalize embedding
    2. Query FAISS for top-5 matches
    3. If best_similarity > threshold:
        - Retrieve person_id from SQLite
        - Update embedding (moving avg: 70% old + 30% new)
        - Update last_seen, appearances++
        - Return (person_id, is_matched=True)
    4. Else:
        - Create new person in SQLite
        - Add embedding to FAISS
        - Return (new_person_id, is_matched=False)
    """
```

**Deliverables:**
- Integrated database working
- Match/create logic validated
- Retention policy implemented

**Testing:**
- Test with synthetic embeddings (10 persons, 100 appearances each)
- Verify same person gets matched across time
- Verify different persons get different IDs
- Test retention policy (expire old persons)

---

### Week 4: Pipeline Integration

#### Day 22-24: RE-ID Manager Module
**Tasks:**
- [ ] Implement `reid_manager.py` (analytics module)
- [ ] Integrate with pipeline probes
- [ ] Extract OSNet embeddings from tracker metadata
- [ ] Call `match_or_create_person()` for each tracked object
- [ ] Map tracker IDs → persistent person IDs
- [ ] Display persistent IDs on OSD

**Integration Flow:**
```python
def osd_sink_pad_buffer_probe(pad, info, u_data):
    frame_meta = pyds.gst_buffer_get_nvds_batch_meta(info.get_buffer())

    for frame in frame_meta.frame_meta_list:
        for obj in frame.obj_meta_list:
            # Extract tracker ID and embedding
            tracker_id = obj.object_id
            embedding = extract_osnet_embedding(obj)  # From NVDS_TRACKER_PAST_FRAME_META

            # Match against database
            person_id, is_new = reid_manager.match_or_create_person(
                embedding=embedding,
                threshold=0.75,
                confidence=obj.confidence,
                bbox=(obj.rect_params.left, obj.rect_params.top,
                      obj.rect_params.width, obj.rect_params.height),
                frame_number=frame.frame_num,
                camera_id=frame.source_id
            )

            # Display persistent ID on OSD
            obj.text_params.display_text = f"Person {person_id}"
```

**Deliverables:**
- RE-ID manager integrated with pipeline
- Persistent person IDs displayed on OSD
- Database updated in real-time

**Testing:**
- Test with video: person leaves and returns → same persistent ID
- Verify database grows correctly (new persons added)
- Verify same person matches across sessions

---

#### Day 25-28: Validation & Tuning
**Tasks:**
- [ ] Create ground truth dataset (annotated test videos)
- [ ] Measure RE-ID accuracy (precision, recall, F1)
- [ ] Tune FAISS similarity threshold (test 0.65, 0.70, 0.75, 0.80, 0.85)
- [ ] Optimize embedding update strategy (test different moving avg ratios)
- [ ] Test long-duration scenarios (30-minute videos)

**Metrics to Measure:**
- **True Positives:** Same person correctly matched
- **False Positives:** Different persons incorrectly matched
- **False Negatives:** Same person not matched (new ID assigned)
- **Target:** >80% F1 score

**Deliverables:**
- RE-ID accuracy report
- Optimal threshold determined
- Long-duration stability verified

**Testing:**
- Test with 10+ test videos
- Generate confusion matrix
- Document optimal parameters

---

**Phase 2 Milestone:**
✅ FAISS + SQLite database operational
✅ Persistent RE-ID working across time gaps
✅ RE-ID accuracy >80%
✅ Database integrated with pipeline

---

## Phase 3: Demographics & Attention Analytics (Weeks 5-6)

**Goal:** Add face detection, demographics classification, head pose estimation, and attention tracking.

### Week 5: Face & Demographics

#### Day 29-31: Face Detection (SGIE-1)
**Tasks:**
- [ ] Download RetinaFace ONNX model
- [ ] Convert to TensorRT engine
- [ ] Configure as SGIE-1 (operates on person crops)
- [ ] Extract face bounding boxes and landmarks
- [ ] Display face boxes on OSD

**DeepStream Config:**
```ini
[secondary-gie0]
enable=1
model-engine-file=models/retinaface/retinaface_mobilenet025_fp16.engine
operate-on-gie-id=1  # Operates on PGIE (person) outputs
batch-size=8
```

**Deliverables:**
- Face detection working
- Faces detected within person bounding boxes
- Face landmarks extracted

**Testing:**
- Test with videos containing multiple faces
- Verify face detection accuracy
- Test profile faces, occluded faces

---

#### Day 32-33: Demographics Classification (SGIE-2)
**Tasks:**
- [ ] Download MiVOLO ONNX model (age + gender)
- [ ] Convert to TensorRT engine
- [ ] Configure as SGIE-2 (operates on face crops)
- [ ] Extract age group and gender predictions
- [ ] Display demographics on OSD

**Output Format:**
```python
demographics = {
    "age_group": "20-29",      # 9 age groups
    "gender": "Male",          # Male, Female, Unknown
    "age_confidence": 0.87,
    "gender_confidence": 0.92
}
```

**Deliverables:**
- Demographics classification working
- Age and gender displayed on OSD
- Confidence scores available

**Testing:**
- Test with diverse age groups and genders
- Measure accuracy on labeled dataset
- Verify low-confidence cases

---

#### Day 34-35: Demographics Aggregation
**Tasks:**
- [ ] Implement `demographics_aggregator.py`
- [ ] Store demographics in person database
- [ ] Use majority vote or moving average per person
- [ ] Generate aggregate statistics (age/gender distribution)

**Logic:**
```python
class DemographicsAggregator:
    def update_person_demographics(self, person_id, age_group, gender, confidence):
        # Get person history from database
        # Apply majority vote or confidence-weighted average
        # Update person record
        pass

    def get_demographics_distribution(self, time_window):
        # Query database for persons in time window
        # Calculate percentages
        return {
            "age_distribution": {"20-29": 0.35, "30-39": 0.40, ...},
            "gender_distribution": {"Male": 0.60, "Female": 0.40}
        }
```

**Deliverables:**
- Demographics aggregation working
- Person profiles updated with demographics
- Distribution statistics available

**Testing:**
- Verify demographics stabilize over time for each person
- Test aggregate distribution calculation

---

### Week 6: Attention & Dwell Time

#### Day 36-38: Head Pose Estimation (SGIE-3)
**Tasks:**
- [ ] Download 6DRepNet ONNX model
- [ ] Convert to TensorRT engine
- [ ] Configure as SGIE-3 (operates on face crops)
- [ ] Extract yaw, pitch, roll angles
- [ ] Display head orientation on OSD (arrows or vectors)

**Output Format:**
```python
head_pose = {
    "yaw": -12.5,     # degrees (-90 to +90)
    "pitch": 5.2,     # degrees (-90 to +90)
    "roll": 2.1       # degrees (-180 to +180)
}
```

**Deliverables:**
- Head pose estimation working
- Yaw/pitch/roll angles extracted
- Visualization on OSD

**Testing:**
- Test with persons at different angles
- Verify accuracy of angle estimation
- Test extreme angles

---

#### Day 39-40: Attention Detection
**Tasks:**
- [ ] Implement `attention_tracker.py`
- [ ] Define attention criteria (|yaw| < 30°, |pitch| < 20°)
- [ ] Implement state machine (not_looking → looking → engaged)
- [ ] Track attention events per person
- [ ] Calculate total attention time

**State Machine:**
```
NOT_LOOKING ──(head faces display)──> LOOKING
LOOKING ──(duration > 2s)──> ENGAGED (qualified impression)
LOOKING ──(head turns away)──> NOT_LOOKING
ENGAGED ──(head turns away)──> NOT_LOOKING
```

**Deliverables:**
- Attention state tracking working
- Attention time calculated per person
- Qualified impressions counted (>2s attention)

**Testing:**
- Test with persons looking at and away from display
- Verify 2-second threshold for qualified impressions
- Test rapid head movements

---

#### Day 41-42: Dwell Time Calculation
**Tasks:**
- [ ] Implement `dwell_time_calculator.py`
- [ ] Track first_seen and last_seen timestamps per person
- [ ] Handle gaps (person leaves and returns)
- [ ] Calculate total dwell time
- [ ] Store in person database

**Logic:**
```python
class DwellTimeCalculator:
    def update(self, person_id, frame_timestamp):
        # Update last_seen
        # Calculate dwell time with gap tolerance
        pass

    def get_avg_dwell_time(self, time_window):
        # Query persons in time window
        # Calculate average dwell time
        return avg_dwell_time_seconds
```

**Deliverables:**
- Dwell time tracking working
- Average dwell time calculated
- Stored in database

**Testing:**
- Test with persons staying different durations
- Verify gap handling (person leaves and returns)
- Calculate statistics

---

**Phase 3 Milestone:**
✅ Face detection, demographics, head pose all working
✅ Attention detection operational
✅ Dwell time tracking accurate
✅ Analytics integrated with database

---

## Phase 4: Integration & Production (Weeks 7-8)

**Goal:** Finalize system controller integration, optimize for production, and deploy to Dubai.

### Week 7: Integration

#### Day 43-45: JSON Output System
**Tasks:**
- [ ] Implement `json_exporter.py`
- [ ] Define JSON output schema (see ARCHITECTURE.md)
- [ ] Query database for analytics summary
- [ ] Generate JSON file every 60 seconds
- [ ] Write to output directory

**JSON Schema:**
```json
{
  "timestamp": "2026-01-19T14:35:22Z",
  "device_id": "jetson_001",
  "cameras": [
    {
      "camera_id": 1,
      "analytics": {
        "total_persons_detected": 15,
        "unique_persons": 12,
        "qualified_impressions": 8,
        "avg_dwell_time": 4.2,
        "avg_attention_time": 2.8,
        "demographics": {...},
        "attention_rate": 0.67
      }
    }
  ],
  "system_status": {...}
}
```

**Deliverables:**
- JSON exporter working
- Analytics JSON generated every 60s
- Schema validated

**Testing:**
- Verify JSON format is correct
- Test with system controller (mock integration)

---

#### Day 46-47: System Controller Integration
**Tasks:**
- [ ] Define integration API with system controller
- [ ] Implement CLI arguments for system controller mode
- [ ] Test launching from system controller
- [ ] Test data exchange (analytics.json → system controller)
- [ ] Error handling and logging

**CLI Interface:**
```bash
python src/main.py \
  --config /etc/inverse_point/ai_config.yaml \
  --system-controller-mode \
  --output /var/inverse_point/analytics.json \
  --interval 60
```

**Deliverables:**
- System controller integration working
- CLI interface finalized
- Documentation for system controller team

**Testing:**
- Test launching from system controller
- Test JSON file exchange
- Test error scenarios (camera failure, etc.)

---

#### Day 48-49: Configuration & Deployment
**Tasks:**
- [ ] Finalize all configuration files
- [ ] Create deployment scripts (setup.sh, start.sh, stop.sh)
- [ ] Create systemd service file (for auto-start)
- [ ] Document installation procedure
- [ ] Test clean installation on fresh Jetson

**Deliverables:**
- Deployment package ready
- Installation guide complete
- Auto-start service configured

**Testing:**
- Fresh install on test Jetson
- Verify auto-start after reboot
- Test all configurations

---

### Week 8: Optimization & Dubai Deployment

#### Day 50-52: Performance Optimization
**Tasks:**
- [ ] Profile entire pipeline (end-to-end)
- [ ] Optimize bottlenecks (memory allocations, database queries)
- [ ] Enable all TensorRT optimizations (FP16, INT8 if needed)
- [ ] Reduce memory footprint
- [ ] Test 4-camera scenario at full load

**Target Metrics:**
- 30 FPS per camera (4 cameras)
- <4GB RAM usage
- 70-80% GPU utilization
- <500MB database size (30-day data)

**Deliverables:**
- Optimized pipeline
- Performance benchmark report
- Resource usage documented

**Testing:**
- 24-hour stress test (4 cameras)
- Monitor FPS, memory, GPU over time
- Verify no memory leaks

---

#### Day 53-55: Dubai Field Testing
**Tasks:**
- [ ] Ship/bring system to Dubai
- [ ] Install on Jetson Orin NX devices (5-10 units)
- [ ] Test with actual USB cameras in production environment
- [ ] Integrate with real system controllers
- [ ] Collect real-world performance data
- [ ] Fix any field issues

**Deliverables:**
- System deployed in Dubai
- Real-world testing complete
- Issues identified and fixed

**Testing:**
- Test all 5-10 devices
- Verify analytics accuracy in real scenarios
- Test system controller integration

---

#### Day 56: Final Documentation & Handoff
**Tasks:**
- [ ] Finalize all documentation (API, User Guide, Testing Guide)
- [ ] Create training materials for Dubai team
- [ ] Document known issues and workarounds
- [ ] Create maintenance guide
- [ ] Handoff to operations team

**Deliverables:**
- Complete documentation package
- Training materials
- Maintenance guide
- Project closure report

---

**Phase 4 Milestone:**
✅ System controller integration complete
✅ Production deployment successful
✅ Performance targets met
✅ Documentation complete
✅ System operational in Dubai

---

## Testing Strategy

### Unit Tests
- Test each module independently (database, analytics, pipeline)
- Use pytest framework
- Mock external dependencies
- Target: >80% code coverage

### Integration Tests
- Test full pipeline end-to-end
- Test with known ground truth videos
- Measure accuracy metrics
- Test error scenarios (camera failure, network issues)

### Performance Tests
- Benchmark FPS, latency, memory, GPU usage
- Stress test with long-duration runs (24+ hours)
- Test with maximum load (4 cameras @ 1080p)

### Acceptance Tests
- Test in production environment (Dubai)
- Verify analytics accuracy with real data
- User acceptance testing (system controller team)

---

## Dependencies & Requirements

### Python Dependencies (requirements.txt)
```
# Core
numpy>=1.21.0
opencv-python>=4.5.0
pyyaml>=5.4.0

# DeepStream
pyds  # DeepStream Python bindings (comes with DeepStream)

# Database
faiss-gpu>=1.7.0  # GPU-accelerated FAISS
sqlite3  # Built into Python

# ONNX/TensorRT (optional, for model conversion)
onnx>=1.10.0
onnxruntime-gpu>=1.10.0

# Utilities
structlog>=21.5.0  # Structured logging
pytest>=6.2.0  # Testing
```

### System Dependencies
```bash
# Already installed on Jetson with JetPack
- CUDA 12.6
- TensorRT 10.3
- cuDNN 9.0
- DeepStream 7.1
- GStreamer 1.20

# To install
sudo apt-get install -y \
  python3-pip \
  python3-dev \
  libgstreamer1.0-dev \
  libgstreamer-plugins-base1.0-dev \
  libglib2.0-dev

# Python packages
pip3 install -r requirements.txt
```

### AI Models
```
1. PeopleNet - NVIDIA NGC (free)
   wget --content-disposition https://api.ngc.nvidia.com/v2/models/.../peoplenet.etlt

2. RetinaFace - Convert from ONNX
   Source: https://github.com/biubug6/Pytorch_Retinaface

3. MiVOLO (Age/Gender) - Convert from ONNX
   Source: https://github.com/WildChlamydia/MiVOLO

4. 6DRepNet (Head Pose) - Convert from ONNX
   Source: https://github.com/thohemp/6DRepNet

5. OSNet - Built into DeepStream tracker (no download needed)
```

---

## Risk Mitigation

### Technical Risks
1. **FPS below 30 with 4 cameras**
   - Mitigation: Start with 2 cameras, optimize incrementally
   - Fallback: Reduce resolution to 720p, use INT8 quantization

2. **Model conversion issues (ONNX→TensorRT)**
   - Mitigation: Use pre-converted engines when available
   - Fallback: Use ONNX Runtime (slower but more compatible)

3. **RE-ID accuracy below 80%**
   - Mitigation: Tune threshold, add temporal smoothing
   - Fallback: Reduce threshold slightly, accept higher false positive rate

### Operational Risks
1. **Camera compatibility issues**
   - Mitigation: Test with multiple camera models early
   - Fallback: Maintain camera compatibility list

2. **Storage exhaustion**
   - Mitigation: Implement 30-day retention, monitoring
   - Fallback: Reduce retention to 15 days, compress embeddings

---

## Success Criteria

### Technical Success
- [x] 30 FPS per camera (up to 4 cameras)
- [x] RE-ID accuracy >80%
- [x] Demographics accuracy >75%
- [x] <4GB RAM usage
- [x] <500MB database size (30-day data)

### Business Success
- [x] System deployed in Dubai (5-10 devices)
- [x] Integrated with system controller
- [x] Real-world analytics validated
- [x] User acceptance testing passed

### Documentation Success
- [x] Architecture document complete
- [x] API reference complete
- [x] User guide complete
- [x] Testing guide complete
- [x] Training materials delivered

---

## Next Steps

1. **Review this plan** with stakeholders
2. **Begin Phase 1** (Core Pipeline Development)
3. **Weekly check-ins** to review progress
4. **Iterative testing** at each phase
5. **Dubai deployment** in Week 8

---

**Document Version:** 1.0
**Last Updated:** 2026-01-19
**Status:** ✅ Implementation Plan Ready
**Next Action:** Begin Phase 1 - Core Pipeline Development
