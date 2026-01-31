# IP AI v3 - Professional Implementation Plan

**Project:** Inverse Point AI Analytics System v3.0
**Platform:** NVIDIA Jetson Orin Nano (current) / Orin NX (production)
**Framework:** NVIDIA DeepStream SDK 7.1
**Created:** 2026-01-30
**Approach:** Incremental Development with Component Testing

---

## Executive Summary

This plan delivers a **production-grade, GPU-accelerated AI analytics system** using NVIDIA DeepStream SDK. The system provides real-time person detection, tracking, persistent re-identification, demographics analysis, and attention detection for indoor advertising displays.

### Key Differentiators from Previous Attempts

| Issue in Previous Attempts | Solution in v3 |
|---------------------------|----------------|
| CPU-based HOG features (5 FPS) | GPU-native OSNet embeddings (30 FPS) |
| Classical CV over-consolidation | Deep learning embeddings from tracker |
| Incomplete pipeline | Incremental build with testing |
| Complex model stack | DeepStream-native components |

### Target Metrics

| Metric | Target | Justification |
|--------|--------|---------------|
| **FPS** | 25-30 per camera | Real-time processing requirement |
| **ReID Accuracy** | >80% | Sufficient for analytics |
| **GPU Utilization** | 70-80% | Optimal resource usage |
| **Memory** | <4GB | Jetson Orin Nano constraint |
| **Latency** | <100ms | Real-time responsiveness |

---

## Technology Stack

### Core Framework
```
┌─────────────────────────────────────────────────────────┐
│                 NVIDIA DeepStream SDK 7.1               │
├─────────────────────────────────────────────────────────┤
│  Video Decode  │  nvinfer (TRT)  │  nvtracker (NvDCF)  │
│  nvstreammux   │  nvdsosd        │  Probe callbacks     │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│              Analytics & Storage Layer                   │
├─────────────────────────────────────────────────────────┤
│  FAISS (GPU)  │  SQLite  │  JSON Export  │  REST API   │
└─────────────────────────────────────────────────────────┘
```

### Hardware Requirements
- **Current:** Jetson Orin Nano (8GB RAM, 6 CPU cores)
- **Production:** Jetson Orin NX (16GB RAM, 8 CPU cores)
- **Cameras:** USB cameras (1-4), tested with Rapoo camera

### Software Stack
| Component | Version | Purpose |
|-----------|---------|---------|
| DeepStream SDK | 7.1 | Video pipeline framework |
| TensorRT | 10.3 | Model optimization |
| CUDA | 12.6 | GPU compute |
| Python | 3.10+ | Application code |
| GStreamer | 1.20+ | Pipeline backend |
| FAISS | 1.7+ | Vector similarity search |
| SQLite | 3.x | Metadata storage |

---

## System Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────────────┐
│                        IP AI v3 System                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  USB Camera(s)                                                       │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              DeepStream Pipeline (GPU)                       │   │
│  ├─────────────────────────────────────────────────────────────┤   │
│  │  1. v4l2src → nvvideoconvert → nvstreammux                   │   │
│  │  2. nvinfer (PeopleNet/YOLO) → Person Detection              │   │
│  │  3. nvtracker (NvDCF+OSNet) → Tracking + 128-dim Embeddings  │   │
│  │  4. [Optional] nvinfer (Face Det) → Face Crops               │   │
│  │  5. [Optional] nvinfer (Demographics) → Age/Gender           │   │
│  │  6. [Optional] nvinfer (HeadPose) → Attention                │   │
│  │  7. nvdsosd → Visualization (debug mode)                     │   │
│  │  8. Probe Callbacks → Metadata Extraction                    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              Analytics Engine (CPU)                          │   │
│  ├─────────────────────────────────────────────────────────────┤   │
│  │  • ReID Manager: Match embeddings against FAISS database     │   │
│  │  • Attention Tracker: State machine for engagement           │   │
│  │  • Demographics Aggregator: Majority vote per person         │   │
│  │  • Dwell Time Calculator: Track presence duration            │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              Storage & Output                                │   │
│  ├─────────────────────────────────────────────────────────────┤   │
│  │  • FAISS Index: 128-dim embeddings (GPU-accelerated)         │   │
│  │  • SQLite: Person metadata, appearances, analytics           │   │
│  │  • JSON Export: Analytics for system controller              │   │
│  │  • [Optional] REST API: Real-time queries                    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow Per Frame

```
Frame N arrives (30 FPS)
    │
    ├─► [GPU] Decode & Preprocess (~2ms)
    │
    ├─► [GPU] Person Detection (~15ms)
    │       │
    │       └─► Detected: [P1, P2, P3] with bounding boxes
    │
    ├─► [GPU] Multi-Object Tracking (~8ms)
    │       │
    │       └─► Track IDs + 128-dim OSNet embeddings
    │
    ├─► [CPU] ReID Matching (~1ms)
    │       │
    │       ├─► Query FAISS for nearest neighbors
    │       ├─► Match > 0.75 similarity → Existing person
    │       └─► No match → Create new person
    │
    ├─► [GPU] Face Detection (optional, ~5ms)
    │       │
    │       └─► Face crops for demographics
    │
    ├─► [GPU] Demographics + HeadPose (optional, ~10ms)
    │       │
    │       └─► Age group, gender, yaw/pitch/roll
    │
    ├─► [CPU] Analytics Update (~1ms)
    │       │
    │       ├─► Update dwell time
    │       ├─► Update attention state
    │       └─► Store appearance in database
    │
    └─► Continue to Frame N+1

Total: ~33ms/frame = 30 FPS achievable
```

---

## Incremental Development Phases

### Overview

```
Phase 1 (Foundation)     → Camera + Basic Detection
Phase 2 (Tracking)       → Add NvDCF Tracker + OSNet
Phase 3 (ReID)           → Persistent Database
Phase 4 (Demographics)   → Face + Age/Gender
Phase 5 (Attention)      → Head Pose + Engagement
Phase 6 (Integration)    → JSON Output + Production
```

---

## Phase 1: Foundation - Camera & Detection

**Goal:** Get USB camera working with person detection on DeepStream

### 1.1 Project Setup

```bash
# Directory structure
ip_ai_v3/
├── configs/
│   ├── pipeline_config.yaml      # Main configuration
│   ├── pgie_config.txt           # Person detection config
│   └── tracker_config.txt        # Tracker config (Phase 2)
├── src/
│   ├── __init__.py
│   ├── main.py                   # Entry point
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── camera_pipeline.py    # Camera-based pipeline
│   │   └── probe_manager.py      # Metadata extraction
│   ├── analytics/                # Phase 3+
│   ├── database/                 # Phase 3
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       └── config.py
├── models/
│   └── peoplenet/                # Detection model
├── output/
├── tests/
├── docs/
│   └── PROFESSIONAL_IMPLEMENTATION_PLAN.md
├── requirements.txt
└── README.md
```

### 1.2 Tasks

| Task | Description | Status |
|------|-------------|--------|
| 1.1.1 | Create project structure | Pending |
| 1.1.2 | Install dependencies | Pending |
| 1.1.3 | Download PeopleNet model from NGC | Pending |
| 1.1.4 | Create nvinfer config for detection | Pending |
| 1.1.5 | Implement basic camera pipeline | Pending |
| 1.1.6 | Add OSD visualization | Pending |
| 1.1.7 | Test with USB camera | Pending |
| 1.1.8 | Measure baseline FPS | Pending |

### 1.3 Detection Model Options

**Option A: PeopleNet (Recommended)**
- Source: NVIDIA NGC (TAO Toolkit)
- Optimized for person detection
- TensorRT engine available
- Classes: Person, Bag, Face

**Option B: YOLOv8n (Backup)**
- Already available from ip_ai_analytics
- General object detection
- Need to filter for person class only
- Good community support

### 1.4 Success Criteria
- [ ] USB camera stream displaying in real-time
- [ ] Person detection boxes showing on OSD
- [ ] FPS counter showing ≥25 FPS
- [ ] No memory leaks over 5-minute run

---

## Phase 2: Multi-Object Tracking with OSNet

**Goal:** Add NvDCF tracker with OSNet embeddings extraction

### 2.1 NvDCF Tracker Configuration

```ini
# configs/tracker_config.txt
[tracker]
tracker-width=640
tracker-height=384
ll-lib-file=/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so
ll-config-file=configs/nvdcf_config.yml
enable-batch-process=1
enable-past-frame=1          # CRITICAL: Enables embedding access
display-tracking-id=1
```

### 2.2 OSNet Embedding Extraction

The key insight from previous attempts: **Use the built-in OSNet embeddings from NvDCF tracker, NOT manual feature extraction.**

```python
# Access embeddings via NVDS_TRACKER_PAST_FRAME_META
def extract_osnet_embedding(obj_meta):
    """
    Extract 128-dim OSNet embedding from tracker metadata

    Key: NvDCF tracker with enable-past-frame=1 computes
    OSNet embeddings internally - we just need to extract them
    """
    # Access past frame user meta
    past_frame_meta = pyds.NvDsUserMeta.cast(
        obj_meta.obj_user_meta_list.data
    )

    # Extract embedding (128 float32 values)
    embedding = np.frombuffer(
        past_frame_meta.user_meta_data,
        dtype=np.float32,
        count=128
    )

    return embedding
```

### 2.3 Tasks

| Task | Description | Status |
|------|-------------|--------|
| 2.1.1 | Configure NvDCF tracker | Pending |
| 2.1.2 | Add tracker to pipeline | Pending |
| 2.1.3 | Implement embedding extraction probe | Pending |
| 2.1.4 | Verify embeddings are 128-dim vectors | Pending |
| 2.1.5 | Test track consistency (same ID across frames) | Pending |
| 2.1.6 | Measure FPS with tracker | Pending |

### 2.4 Success Criteria
- [ ] Track IDs stable across frames (person keeps same ID)
- [ ] OSNet embeddings extracted (128-dim float32)
- [ ] FPS still ≥25 FPS with tracker
- [ ] Re-association working (person leaving/returning within 5s)

---

## Phase 3: Persistent Re-Identification Database

**Goal:** Store embeddings in FAISS + SQLite for cross-session matching

### 3.1 Database Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Person Database                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────┐  ┌────────────────────────┐   │
│  │   FAISS Vector Index    │  │    SQLite Metadata     │   │
│  ├─────────────────────────┤  ├────────────────────────┤   │
│  │ • IndexFlatIP (cosine)  │  │ persons:               │   │
│  │ • 128-dim embeddings    │  │   - person_id          │   │
│  │ • GPU-accelerated       │  │   - embedding (blob)   │   │
│  │ • <1ms query time       │  │   - first_seen         │   │
│  │ • Save/load to disk     │  │   - last_seen          │   │
│  └─────────────────────────┘  │   - appearances        │   │
│                               │   - demographics       │   │
│                               │   - attention_time     │   │
│                               │                        │   │
│                               │ appearances:           │   │
│                               │   - person_id, frame,  │   │
│                               │     timestamp, bbox,   │   │
│                               │     confidence, etc.   │   │
│                               └────────────────────────┘   │
│                                                              │
│  Matching Algorithm:                                         │
│  1. Normalize embedding (L2)                                 │
│  2. Query FAISS top-5 neighbors                              │
│  3. If similarity > 0.75: match existing person              │
│  4. Else: create new person                                  │
│  5. Update embedding: 70% old + 30% new (moving average)     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 ReID Manager

```python
class ReIDManager:
    """
    Manages persistent person re-identification

    Key Features:
    - FAISS GPU index for fast similarity search
    - SQLite for metadata persistence
    - Moving average embedding updates
    - 30-day retention policy
    """

    def __init__(self, faiss_path, sqlite_path, threshold=0.75):
        self.threshold = threshold
        self.faiss_index = self._load_or_create_faiss(faiss_path)
        self.sqlite_conn = sqlite3.connect(sqlite_path)

    def match_or_create(self, embedding, tracker_id, camera_id,
                        bbox, confidence, frame_num):
        """
        Match embedding against database or create new person

        Returns: (persistent_person_id, is_new_person)
        """
        # 1. Normalize embedding
        embedding = embedding / np.linalg.norm(embedding)

        # 2. Query FAISS
        similarities, indices = self.faiss_index.search(
            embedding.reshape(1, -1), k=5
        )

        # 3. Check for match
        if similarities[0][0] > self.threshold:
            person_id = self.id_map[indices[0][0]]
            self._update_person(person_id, embedding)
            self._log_appearance(person_id, tracker_id, ...)
            return person_id, False

        # 4. Create new person
        person_id = self._create_person(embedding)
        self._log_appearance(person_id, tracker_id, ...)
        return person_id, True
```

### 3.3 Tasks

| Task | Description | Status |
|------|-------------|--------|
| 3.1.1 | Copy PersonDatabase from ip_ai_analytics | Pending |
| 3.1.2 | Adapt for OSNet 128-dim embeddings | Pending |
| 3.1.3 | Integrate with pipeline probe | Pending |
| 3.1.4 | Test cross-gap matching (person leaves/returns) | Pending |
| 3.1.5 | Tune similarity threshold (0.70-0.80) | Pending |
| 3.1.6 | Test persistence (stop/restart application) | Pending |
| 3.1.7 | Measure ReID accuracy | Pending |

### 3.4 Success Criteria
- [ ] Same person returns → gets same persistent ID
- [ ] Different people → get different persistent IDs
- [ ] Database persists across restarts
- [ ] ReID accuracy >80% on test videos
- [ ] FPS still ≥25 FPS

---

## Phase 4: Demographics Analysis

**Goal:** Add face detection and age/gender classification

### 4.1 Model Stack

```
Person Detection (PGIE)
        │
        ▼
Face Detection (SGIE-1)
        │
        ▼
Demographics (SGIE-2)
        │
        ▼
Age Group + Gender
```

### 4.2 Model Options

**Face Detection:**
- RetinaFace (MobileNet backbone) - lightweight, accurate
- OR PeopleNet face class (already detected)

**Age/Gender:**
- MiVOLO (SOTA, ONNX available)
- OR InsightFace AgeGender (simpler)

### 4.3 Tasks

| Task | Description | Status |
|------|-------------|--------|
| 4.1.1 | Download/convert face detection model | Pending |
| 4.1.2 | Configure as SGIE-1 | Pending |
| 4.1.3 | Download/convert demographics model | Pending |
| 4.1.4 | Configure as SGIE-2 | Pending |
| 4.1.5 | Implement DemographicsAggregator | Pending |
| 4.1.6 | Test accuracy on diverse faces | Pending |
| 4.1.7 | Optimize for FPS | Pending |

### 4.4 Success Criteria
- [ ] Face detection working on person crops
- [ ] Age/gender classification showing
- [ ] Demographics aggregated per person
- [ ] FPS ≥20 FPS with demographics enabled

---

## Phase 5: Attention Detection

**Goal:** Add head pose estimation for engagement tracking

### 5.1 Attention State Machine

```
                 ┌─────────────────┐
                 │   NOT_LOOKING   │
                 └────────┬────────┘
                          │ head faces display
                          │ (|yaw| < 30°, |pitch| < 20°)
                          ▼
                 ┌─────────────────┐
                 │     LOOKING     │
                 └────────┬────────┘
                          │ duration > 2 seconds
                          ▼
                 ┌─────────────────┐
                 │    ENGAGED      │ ← Qualified Impression
                 └─────────────────┘
```

### 5.2 Model

- **6DRepNet**: Yaw, pitch, roll estimation
- Input: Face crop (224x224)
- Output: 3 angles in degrees
- TensorRT optimized

### 5.3 Tasks

| Task | Description | Status |
|------|-------------|--------|
| 5.1.1 | Download/convert 6DRepNet model | Pending |
| 5.1.2 | Configure as SGIE-3 | Pending |
| 5.1.3 | Implement AttentionTracker | Pending |
| 5.1.4 | Track attention time per person | Pending |
| 5.1.5 | Count qualified impressions | Pending |
| 5.1.6 | Test with people looking at/away from camera | Pending |

### 5.4 Success Criteria
- [ ] Head pose angles extracted
- [ ] Attention state tracked per person
- [ ] Qualified impressions counted
- [ ] Total attention time calculated

---

## Phase 6: Integration & Production

**Goal:** JSON output, system controller integration, optimization

### 6.1 JSON Output Format

```json
{
  "timestamp": "2026-01-30T14:35:22Z",
  "device_id": "jetson_orin_nano_001",
  "cameras": [
    {
      "camera_id": 1,
      "analytics": {
        "total_persons_detected": 15,
        "unique_persons": 12,
        "qualified_impressions": 8,
        "avg_dwell_time": 4.2,
        "avg_attention_time": 2.8,
        "demographics": {
          "age_distribution": {
            "20-29": 0.33,
            "30-39": 0.42,
            "40-49": 0.17,
            "50-59": 0.08
          },
          "gender_distribution": {
            "Male": 0.58,
            "Female": 0.42
          }
        },
        "attention_rate": 0.67
      }
    }
  ],
  "system_status": {
    "fps": 28.5,
    "gpu_usage_percent": 72,
    "memory_usage_mb": 3200
  }
}
```

### 6.2 Tasks

| Task | Description | Status |
|------|-------------|--------|
| 6.1.1 | Implement JSON exporter | Pending |
| 6.1.2 | Add analytics summary queries | Pending |
| 6.1.3 | Create CLI interface | Pending |
| 6.1.4 | Optimize for 4-camera support | Pending |
| 6.1.5 | Add systemd service file | Pending |
| 6.1.6 | Create deployment scripts | Pending |
| 6.1.7 | Write user documentation | Pending |

### 6.3 Success Criteria
- [ ] JSON output generated every 60 seconds
- [ ] CLI supports all configuration options
- [ ] Auto-start on device boot
- [ ] 4 cameras @ ≥20 FPS each

---

## Privacy Compliance

### What We Store (Privacy-Safe)
- 128-dim embedding vectors (not reversible to images)
- Aggregate demographics (age groups, gender percentages)
- Anonymous metrics (counts, durations)

### What We DO NOT Store
- Raw video frames
- Face images
- Personal identifiers
- Biometric templates

### Compliance
- GDPR Article 25: Privacy by design
- UAE Data Protection: No biometric storage
- 30-day retention with auto-delete

---

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| FPS below 25 | Medium | High | Start with 720p, optimize models |
| ReID accuracy <80% | Medium | Medium | Tune threshold, add temporal smoothing |
| USB camera issues | Low | Medium | Test multiple camera models |
| Memory overflow | Low | High | Monitor and optimize buffers |
| Model conversion fails | Low | Medium | Use pre-built TRT engines |

---

## Development Guidelines

### Code Standards
- Python 3.10+ with type hints
- Google-style docstrings
- Black formatter, pylint linting
- Logging via structlog
- Configuration via YAML

### Testing Strategy
- Unit tests for each module
- Integration tests for pipeline
- Performance benchmarks
- Accuracy validation with ground truth

### Git Workflow
- Main branch: stable releases
- Dev branch: active development
- Feature branches for each phase

---

## Estimated Timeline

| Phase | Components | Estimated Effort |
|-------|------------|------------------|
| Phase 1 | Camera + Detection | 2-3 days |
| Phase 2 | Tracking + OSNet | 2-3 days |
| Phase 3 | ReID Database | 3-4 days |
| Phase 4 | Demographics | 3-4 days |
| Phase 5 | Attention | 2-3 days |
| Phase 6 | Integration | 3-4 days |

**Total: 15-21 days** for full implementation

---

## Next Steps

1. **Create project structure** (ip_ai_v3/)
2. **Set up development environment**
3. **Begin Phase 1: Camera + Detection**
4. **Test with USB camera before proceeding**

---

**Document Version:** 1.0
**Created:** 2026-01-30
**Status:** Ready for Implementation
