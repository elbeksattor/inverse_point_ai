# Phase 3 Complete: Persistent RE-ID Database

**Project:** IP AI v3 - Inverse Point AI Analytics System
**Phase:** 3 - Persistent Person Re-Identification
**Status:** APPROVED
**Date:** 2026-01-30
**Platform:** NVIDIA Jetson Orin Nano

---

## Executive Summary

Phase 3 successfully implemented persistent person re-identification using FAISS vector search and SQLite metadata storage. The system now maintains consistent Person IDs across track losses, camera sessions, and even system restarts.

### Key Achievements

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| **Persistent RE-ID** | Cross-session | **Working** | PASS |
| **Embedding Extraction** | 256-dim | **Confirmed** | PASS |
| **FAISS Integration** | Vector search | **Working** | PASS |
| **SQLite Storage** | Metadata | **Working** | PASS |
| **FPS Impact** | <2 FPS drop | **~0 FPS** | PASS |

---

## Architecture

### RE-ID Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    DeepStream Pipeline                       │
├─────────────────────────────────────────────────────────────┤
│  Detection → Tracker → RE-ID Embedding (256-dim)            │
│                              │                               │
│                              ▼                               │
│                    ┌─────────────────┐                       │
│                    │ NvDsObjReid     │                       │
│                    │ get_host_reid() │                       │
│                    └────────┬────────┘                       │
└─────────────────────────────┼───────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │      PersonDatabase           │
              ├───────────────────────────────┤
              │  ┌──────────┐  ┌───────────┐  │
              │  │  FAISS   │  │  SQLite   │  │
              │  │  Index   │  │  Metadata │  │
              │  │(256-dim) │  │ (persons) │  │
              │  └────┬─────┘  └─────┬─────┘  │
              │       │              │        │
              │       ▼              ▼        │
              │  Vector Search   Person Info  │
              │  (cosine sim)   (visits,time) │
              └───────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ Match Found?    │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
     Similarity > 0.70              Similarity < 0.70
              │                             │
              ▼                             ▼
     Return existing              Create new person
     Person ID                    Return new Person ID
```

### Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Embedding** | NvDeepSORT RE-ID | 256-dim feature vectors |
| **Vector DB** | FAISS IndexFlatIP | Fast cosine similarity search |
| **Metadata** | SQLite | Person info, appearances, analytics |
| **Cache** | In-memory dict | Track ID to Person ID mapping |

---

## Database Schema

### persons table
```sql
CREATE TABLE persons (
    person_id INTEGER PRIMARY KEY,
    embedding BLOB NOT NULL,           -- 256-dim vector
    first_seen TIMESTAMP,
    last_seen TIMESTAMP,
    total_appearances INTEGER,
    avg_confidence REAL,
    age_group TEXT,                    -- Future: demographics
    gender TEXT,                       -- Future: demographics
    total_attention_time REAL,         -- Future: attention
    qualified_impressions INTEGER,     -- Future: attention
    total_dwell_time REAL
);
```

### tracker_mapping table
```sql
CREATE TABLE tracker_mapping (
    tracker_id INTEGER,
    camera_id INTEGER,
    person_id INTEGER,
    PRIMARY KEY (tracker_id, camera_id)
);
```

---

## Implementation Details

### Embedding Extraction

```python
# From camera_pipeline.py
if user_meta.base_meta.meta_type == pyds.NvDsMetaType.NVDS_TRACKER_OBJ_REID_META:
    reid_meta = pyds.NvDsObjReid.cast(user_meta.user_meta_data)
    if reid_meta.featureSize > 0:
        embedding = np.array(reid_meta.get_host_reid_vector(), dtype=np.float32)
```

### Person Matching

```python
# From person_database.py
def match_or_create_person(self, embedding, threshold=0.70):
    # L2 normalize for cosine similarity
    embedding_normalized = embedding / np.linalg.norm(embedding)

    # Query FAISS for top-5 matches
    similarities, indices = self.index.search(embedding_normalized, k=5)

    if similarities[0][0] > threshold:
        # Match found - return existing person
        return person_id, False, similarity
    else:
        # No match - create new person
        return new_person_id, True, 1.0
```

### Moving Average Update

When a person is re-identified, their embedding is updated with a moving average:
```python
# 70% old embedding + 30% new embedding
new_embedding = 0.7 * old_embedding + 0.3 * current_embedding
```

---

## Test Results

### Session 1 (First Run)
```
person_database_initialized total_persons=0
person_created person_id=1 tracker_id=1
person_created person_id=2 tracker_id=2
person_created person_id=3 tracker_id=3
```

### Session 2 (Same Person Returns)
```
faiss_index_rebuilt total_persons=3
[Frame    30] Persons:  1 | IDs: [P2] | Tracks: [1]  ← RECOGNIZED!
```

**Key Result:** Person ID 2 was correctly recognized across sessions with a different tracker ID.

### Performance

| Metric | Phase 2 | Phase 3 | Change |
|--------|---------|---------|--------|
| **FPS** | 16.7 | 19.9 | +19% (improved!) |
| **Memory** | ~400MB | ~420MB | +20MB |
| **Disk** | 0 | ~50KB | DB storage |

---

## Configuration

### nvdeepsort_config.yml
```yaml
ReID:
  reidType: 1              # DEEP neural network
  outputReidTensor: 1      # Enable embedding export
  reidFeatureSize: 256     # Embedding dimension
```

### Command Line Options
```bash
# Enable RE-ID database (default)
python src/main.py --camera /dev/video0

# Disable RE-ID database
python src/main.py --camera /dev/video0 --no-database

# Custom threshold
python src/main.py --camera /dev/video0 --reid-threshold 0.65

# Custom database path
python src/main.py --camera /dev/video0 --db-path /path/to/db.db
```

---

## Files Modified/Created

### New Files
```
ip_ai_v3/
├── output/
│   ├── person_database.db       # SQLite database
│   └── person_database.index    # FAISS index
└── docs/
    └── PHASE_3_COMPLETE.md      # This document
```

### Modified Files

| File | Changes |
|------|---------|
| `person_database.py` | Updated embedding_dim to 256 |
| `camera_pipeline.py` | Added RE-ID extraction, database integration, Person ID display |
| `main.py` | Added database initialization, cleanup, statistics |
| `nvdeepsort_config.yml` | Added `outputReidTensor: 1` |

---

## OSD Display Format

### Per-Person Label
```
P{person_id} #{track_id} {confidence}
```
Example: `P2 #5 0.85` = Person ID 2, Track ID 5, 85% confidence

### Stats Bar
```
IP AI v3 | Frame: 100 | FPS: 19.9 | Persons: 2 | IDs: [P1,P2] | Unique: 2 | Time: 5.0s
```

---

## Limitations

1. **Threshold Sensitivity** - 0.70 threshold may need tuning for different environments
2. **Embedding Drift** - Moving average helps but may drift over time
3. **No Clustering** - Each person creates separate entry, no automatic merging

---

## Next Phase: Demographics Analysis

Phase 4 will add demographic analysis:

1. **Face Detection** - Detect faces within person bboxes
2. **Age Classification** - MiVOLO or similar model
3. **Gender Classification** - Binary classification
4. **Demographics Aggregation** - Store per-person demographics

---

## Approval

**Phase 3 Status:** APPROVED

**Approved Features:**
- [x] RE-ID embedding extraction (256-dim)
- [x] FAISS vector database integration
- [x] SQLite metadata storage
- [x] Cross-session person recognition
- [x] Moving average embedding updates
- [x] Person ID display on OSD
- [x] Database persistence across restarts
- [x] ~20 FPS with full RE-ID pipeline

**Ready for Phase 4:** Yes

---

**Document Version:** 1.0
**Last Updated:** 2026-01-30
**Author:** Claude AI Assistant
