#!/usr/bin/env python3
"""
IP AI v3 - Persistent Person RE-ID Database

Combines FAISS vector search with SQLite metadata storage.
Adapted from ip_ai_analytics with improvements for OSNet embeddings.

v0.4.1: Multi-shot embedding storage for improved RE-ID stability
- Stores multiple embeddings per person (gallery approach)
- Gallery-based matching with max similarity
- Temporal boost for recently seen persons
- Reduced false positive new person creation
"""

import sqlite3
import numpy as np
import faiss
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import structlog
import time

logger = structlog.get_logger(__name__)

# Configuration for multi-shot RE-ID
MAX_EMBEDDINGS_PER_PERSON = 10  # Maximum embeddings to store per person (gallery size)
EMBEDDING_DIVERSITY_THRESHOLD = 0.70  # Only add embedding if different enough from existing
RECENT_SEEN_BOOST_SECONDS = 120.0  # Extended window for temporal boost (2 minutes)
RECENT_SEEN_THRESHOLD_REDUCTION = 0.10  # Reduce threshold for recent persons (reduced from 0.15)

# Gallery vote matching - disabled for now (margin check is sufficient)
GALLERY_VOTE_THRESHOLD = 0.35  # Minimum similarity for a vote (lowered)
MIN_GALLERY_VOTES = 0  # Disabled - rely on margin check instead
GALLERY_VOTE_RATIO = 0.0  # Disabled

# Margin-based separation - best match must be better than second best
MATCH_MARGIN_REQUIRED = 0.08  # Best match must exceed second best by this margin
HIGH_CONFIDENCE_THRESHOLD = 0.60  # Skip margin check if similarity is this high

# Adaptive threshold based on database size
# When only 1 person exists, use higher threshold to prevent merging different people
# When 2+ people exist, margin check handles separation so lower threshold is safe
SINGLE_PERSON_THRESHOLD = 0.55  # Higher threshold when only 1 person in database
MULTI_PERSON_THRESHOLD = 0.48  # Lower threshold when margin check can be applied

# Default threshold (used as base, may be adjusted adaptively)
DEFAULT_REID_THRESHOLD = 0.50


class PersonDatabase:
    """
    Manages persistent storage of person RE-ID embeddings and metadata.

    Uses FAISS for fast vector similarity search and SQLite for metadata.
    Optimized for 256-dim ResNet50 embeddings from DeepStream NvDeepSORT tracker.
    """

    def __init__(
        self,
        db_path: str = "output/person_database.db",
        index_path: str = "output/person_embeddings.index",
        embedding_dim: int = 256,  # ResNet50 Market1501 uses 256-dim
        use_gpu: bool = False
    ):
        """
        Initialize person database.

        Args:
            db_path: Path to SQLite database file
            index_path: Path to FAISS index file
            embedding_dim: Dimension of embedding vectors (128 for OSNet)
            use_gpu: Use GPU-accelerated FAISS (if available)
        """
        self.db_path = Path(db_path)
        self.index_path = Path(index_path)
        self.embedding_dim = embedding_dim
        self.use_gpu = use_gpu

        # Create directories
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize SQLite connection (thread-safe for GStreamer)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_schema()

        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner Product for cosine similarity
        self.index_to_person_id = []  # Maps FAISS index position to person_id

        # Multi-shot gallery: person_id -> list of embeddings
        self.person_gallery: Dict[int, List[np.ndarray]] = {}

        # Last seen timestamp per person for temporal boost
        self.person_last_seen: Dict[int, float] = {}

        # Load existing index if available
        self._load_index()

        # Clear old tracker mappings from previous sessions
        self.clear_tracker_mappings()

        logger.info(
            "person_database_initialized",
            db_path=str(self.db_path),
            total_persons=len(self.index_to_person_id),
            embedding_dim=self.embedding_dim,
            gallery_mode="multi_shot"
        )

    def _create_schema(self):
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()

        # Person registry table (digital passport)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS persons (
                person_id INTEGER PRIMARY KEY AUTOINCREMENT,
                embedding BLOB NOT NULL,
                first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_appearances INTEGER DEFAULT 1,
                avg_confidence REAL DEFAULT 0.0,
                age_group TEXT,
                gender TEXT,
                total_attention_time REAL DEFAULT 0.0,
                qualified_impressions INTEGER DEFAULT 0,
                total_dwell_time REAL DEFAULT 0.0,
                last_camera_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Appearance log table (for analytics)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS appearances (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER NOT NULL,
                camera_id INTEGER NOT NULL,
                tracker_id INTEGER,
                frame_number INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                confidence REAL,
                bbox_x REAL,
                bbox_y REAL,
                bbox_width REAL,
                bbox_height REAL,
                age_group TEXT,
                gender TEXT,
                attention_state TEXT,
                head_yaw REAL,
                head_pitch REAL,
                head_roll REAL,
                FOREIGN KEY (person_id) REFERENCES persons(person_id)
            )
        ''')

        # Analytics summary table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analytics_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id INTEGER NOT NULL,
                window_start TIMESTAMP,
                window_end TIMESTAMP,
                total_detections INTEGER DEFAULT 0,
                unique_persons INTEGER DEFAULT 0,
                qualified_impressions INTEGER DEFAULT 0,
                avg_dwell_time REAL DEFAULT 0.0,
                avg_attention_time REAL DEFAULT 0.0,
                demographics_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Tracker ID to Person ID mapping (for current session)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tracker_mapping (
                tracker_id INTEGER NOT NULL,
                camera_id INTEGER NOT NULL,
                person_id INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (tracker_id, camera_id)
            )
        ''')

        # Create indexes for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_person_last_seen ON persons(last_seen)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_appearance_person ON appearances(person_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_appearance_timestamp ON appearances(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_appearance_camera ON appearances(camera_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_analytics_camera ON analytics_summary(camera_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_analytics_window ON analytics_summary(window_start, window_end)')

        self.conn.commit()
        logger.info("database_schema_created")

    def _load_index(self):
        """Load FAISS index, mapping, and gallery from disk if exists."""
        index_file = self.index_path
        mapping_file = Path(str(self.index_path) + ".pkl")
        gallery_file = Path(str(self.index_path) + ".gallery.pkl")

        if index_file.exists() and mapping_file.exists():
            try:
                # Load FAISS index
                self.index = faiss.read_index(str(index_file))

                # Load ID mapping
                with open(mapping_file, 'rb') as f:
                    self.index_to_person_id = pickle.load(f)

                # Load gallery if exists
                if gallery_file.exists():
                    with open(gallery_file, 'rb') as f:
                        self.person_gallery = pickle.load(f)
                    logger.info(
                        "gallery_loaded",
                        total_persons_with_gallery=len(self.person_gallery)
                    )

                # Verify consistency between FAISS index and mapping
                if self.index.ntotal != len(self.index_to_person_id):
                    logger.warning(
                        "faiss_index_mapping_size_mismatch_on_load",
                        index_total=self.index.ntotal,
                        mapping_len=len(self.index_to_person_id)
                    )
                    # Rebuild from database to fix
                    self.index = faiss.IndexFlatIP(self.embedding_dim)
                    self.index_to_person_id = []
                    self._rebuild_index_from_db()
                else:
                    logger.info(
                        "faiss_index_loaded",
                        total_embeddings=self.index.ntotal,
                        path=str(index_file)
                    )
            except Exception as e:
                logger.warning("faiss_index_load_failed", error=str(e))
                self.index = faiss.IndexFlatIP(self.embedding_dim)
                self.index_to_person_id = []
                self.person_gallery = {}
                self._rebuild_index_from_db()
        else:
            # Rebuild index from database embeddings
            logger.info("faiss_index_not_found_rebuilding")
            self._rebuild_index_from_db()

    def _rebuild_index_from_db(self):
        """Rebuild FAISS index from database embeddings."""
        # Reset the index first to ensure clean rebuild
        self.index.reset()
        self.index_to_person_id = []

        cursor = self.conn.cursor()
        cursor.execute('SELECT person_id, embedding FROM persons ORDER BY person_id')

        embeddings = []
        person_ids = []

        for row in cursor.fetchall():
            person_id = row['person_id']
            embedding_blob = row['embedding']
            embedding = np.frombuffer(embedding_blob, dtype=np.float32)

            # Normalize for cosine similarity
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

            embeddings.append(embedding)
            person_ids.append(person_id)

        if embeddings:
            embeddings_array = np.vstack(embeddings).astype('float32')
            self.index.add(embeddings_array)
            self.index_to_person_id = person_ids
            logger.info("faiss_index_rebuilt", total_persons=len(person_ids))

    def save_index(self):
        """Save FAISS index, mapping, and gallery to disk."""
        try:
            # Rebuild index from database to ensure embeddings are synchronized
            logger.info("rebuilding_index_before_save", current_total=self.index.ntotal)
            self._rebuild_index_from_db()

            faiss.write_index(self.index, str(self.index_path))

            with open(str(self.index_path) + ".pkl", 'wb') as f:
                pickle.dump(self.index_to_person_id, f)

            # Save gallery
            gallery_file = Path(str(self.index_path) + ".gallery.pkl")
            with open(gallery_file, 'wb') as f:
                pickle.dump(self.person_gallery, f)

            logger.info(
                "faiss_index_saved",
                path=str(self.index_path),
                total=self.index.ntotal,
                gallery_persons=len(self.person_gallery)
            )
        except Exception as e:
            logger.error("faiss_index_save_failed", error=str(e))

    def _match_against_gallery(
        self,
        embedding_normalized: np.ndarray,
        person_id: int
    ) -> Tuple[float, int, int]:
        """
        Match embedding against a person's gallery of stored embeddings.

        Returns:
            Tuple of (max_similarity, vote_count, gallery_size)
            - max_similarity: Maximum similarity across all gallery embeddings
            - vote_count: Number of gallery embeddings above GALLERY_VOTE_THRESHOLD
            - gallery_size: Total size of gallery
        """
        if person_id not in self.person_gallery:
            return 0.0, 0, 0

        gallery = self.person_gallery[person_id]
        if not gallery:
            return 0.0, 0, 0

        # Compute similarity with each gallery embedding
        max_sim = 0.0
        vote_count = 0
        for gallery_emb in gallery:
            sim = float(np.dot(embedding_normalized.flatten(), gallery_emb.flatten()))
            max_sim = max(max_sim, sim)
            if sim >= GALLERY_VOTE_THRESHOLD:
                vote_count += 1

        return max_sim, vote_count, len(gallery)

    def _add_to_gallery(self, person_id: int, embedding_normalized: np.ndarray):
        """
        Add embedding to person's gallery if diverse enough.

        Maintains gallery size by removing oldest if at capacity.
        """
        if person_id not in self.person_gallery:
            self.person_gallery[person_id] = []

        gallery = self.person_gallery[person_id]

        # Check if embedding is diverse enough (not too similar to existing)
        for existing_emb in gallery:
            sim = float(np.dot(embedding_normalized.flatten(), existing_emb.flatten()))
            if sim > EMBEDDING_DIVERSITY_THRESHOLD:
                # Too similar to existing, skip
                return

        # Add to gallery
        gallery.append(embedding_normalized.flatten().copy())

        # Limit gallery size (remove oldest)
        if len(gallery) > MAX_EMBEDDINGS_PER_PERSON:
            gallery.pop(0)

        logger.debug(
            "gallery_updated",
            person_id=person_id,
            gallery_size=len(gallery)
        )

    def match_or_create_person(
        self,
        embedding: np.ndarray,
        threshold: float = 0.75,
        confidence: float = 0.0,
        tracker_id: Optional[int] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        frame_number: Optional[int] = None,
        camera_id: int = 1
    ) -> Tuple[int, bool, float]:
        """
        Match embedding against database using multi-shot gallery matching.

        Uses gallery-based matching for improved stability:
        1. FAISS provides initial candidate person IDs
        2. Match against full gallery for each candidate
        3. Apply temporal boost for recently seen persons
        4. Select best match above threshold

        Args:
            embedding: RE-ID embedding vector (256-dim for ResNet50)
            threshold: Cosine similarity threshold (0.0-1.0)
            confidence: Detection confidence
            tracker_id: Tracker-assigned ID (for mapping)
            bbox: Bounding box (x, y, width, height)
            frame_number: Current frame number
            camera_id: Camera ID

        Returns:
            Tuple of (person_id, is_new_person, similarity_score)
        """
        current_time = time.time()

        # Normalize embedding for cosine similarity
        embedding_normalized = embedding / (np.linalg.norm(embedding) + 1e-8)
        embedding_normalized = embedding_normalized.reshape(1, -1).astype('float32')

        # Adaptive threshold based on database size
        # When only 1 person exists, use higher threshold to prevent merging different people
        # When 2+ people exist, margin check can help separate, so use lower threshold
        num_persons = self.index.ntotal
        if num_persons <= 1:
            adaptive_threshold = max(threshold, SINGLE_PERSON_THRESHOLD)
        else:
            adaptive_threshold = max(threshold, MULTI_PERSON_THRESHOLD)

        # Use adaptive threshold instead of passed threshold
        threshold = adaptive_threshold

        # Find best match using gallery-based approach
        best_person_id = None
        best_similarity = 0.0
        best_effective_threshold = threshold

        # Track all candidate matches for margin check
        candidate_matches = []  # List of (person_id, similarity, votes, gallery_size, effective_threshold)

        if self.index.ntotal > 0:
            # Query FAISS for top candidates
            k = min(10, self.index.ntotal)
            similarities, indices = self.index.search(embedding_normalized, k=k)

            # Get unique person IDs from candidates
            candidate_person_ids = set()
            for idx in indices[0]:
                if idx >= 0 and idx < len(self.index_to_person_id):
                    candidate_person_ids.add(self.index_to_person_id[idx])

            # Also check recently seen persons (within boost window)
            for pid, last_seen in self.person_last_seen.items():
                if current_time - last_seen < RECENT_SEEN_BOOST_SECONDS:
                    candidate_person_ids.add(pid)

            # Match against each candidate's gallery
            for pid in candidate_person_ids:
                # Get gallery-based similarity with vote counting
                gallery_sim, vote_count, gallery_size = self._match_against_gallery(embedding_normalized, pid)

                # Also check FAISS embedding (primary embedding in database)
                faiss_sim = 0.0
                try:
                    # Find this person's FAISS index
                    if pid in self.index_to_person_id:
                        faiss_idx = self.index_to_person_id.index(pid)
                        # Get similarity from FAISS results if available
                        for i, idx in enumerate(indices[0]):
                            if idx == faiss_idx:
                                faiss_sim = float(similarities[0][i])
                                break
                except (ValueError, IndexError):
                    pass

                # Use maximum of gallery and FAISS similarity
                max_sim = max(gallery_sim, faiss_sim)

                # Count FAISS as a vote if above threshold
                if faiss_sim >= GALLERY_VOTE_THRESHOLD:
                    vote_count += 1
                    gallery_size += 1  # Count FAISS embedding as part of "gallery"

                # Apply temporal boost for recently seen persons
                effective_threshold = threshold
                if pid in self.person_last_seen:
                    time_since_seen = current_time - self.person_last_seen[pid]
                    if time_since_seen < RECENT_SEEN_BOOST_SECONDS:
                        # Reduce threshold for recently seen persons
                        effective_threshold = threshold - RECENT_SEEN_THRESHOLD_REDUCTION
                        logger.debug(
                            "temporal_boost_applied",
                            person_id=pid,
                            time_since_seen=f"{time_since_seen:.1f}s",
                            effective_threshold=f"{effective_threshold:.3f}"
                        )

                # Calculate required votes based on gallery size
                required_votes = max(MIN_GALLERY_VOTES, int(gallery_size * GALLERY_VOTE_RATIO))

                # Store candidate info
                candidate_matches.append({
                    "person_id": pid,
                    "similarity": max_sim,
                    "votes": vote_count,
                    "gallery_size": gallery_size,
                    "required_votes": required_votes,
                    "effective_threshold": effective_threshold,
                    "vote_passed": vote_count >= required_votes
                })

            # Sort candidates by similarity (descending)
            candidate_matches.sort(key=lambda x: x["similarity"], reverse=True)

            # Check best match with vote and margin requirements
            if candidate_matches:
                best_candidate = candidate_matches[0]
                second_best_sim = candidate_matches[1]["similarity"] if len(candidate_matches) > 1 else 0.0

                # Check if best match passes all criteria:
                # 1. Above effective threshold
                # 2. Has enough gallery votes
                # 3. Has sufficient margin over second best (unless recently seen)
                passes_threshold = best_candidate["similarity"] > best_candidate["effective_threshold"]
                passes_votes = best_candidate["vote_passed"] or best_candidate["gallery_size"] < 2  # Skip vote check for new galleries

                # Margin check - relaxed for high-confidence or recently seen persons
                margin = best_candidate["similarity"] - second_best_sim
                margin_required = MATCH_MARGIN_REQUIRED

                # High-confidence matches skip margin check entirely
                if best_candidate["similarity"] >= HIGH_CONFIDENCE_THRESHOLD:
                    passes_margin = True
                elif best_candidate["person_id"] in self.person_last_seen:
                    time_since = current_time - self.person_last_seen[best_candidate["person_id"]]
                    if time_since < RECENT_SEEN_BOOST_SECONDS:
                        margin_required = MATCH_MARGIN_REQUIRED / 2  # Relax margin for recent persons
                    passes_margin = margin >= margin_required or len(candidate_matches) == 1
                else:
                    passes_margin = margin >= margin_required or len(candidate_matches) == 1

                if passes_threshold and passes_votes and passes_margin:
                    best_person_id = best_candidate["person_id"]
                    best_similarity = best_candidate["similarity"]
                    best_effective_threshold = best_candidate["effective_threshold"]

            # Log similarity check
            logger.info(
                "reid_gallery_check",
                best_similarity=f"{best_similarity:.4f}",
                threshold=f"{threshold:.4f}",
                effective_threshold=f"{best_effective_threshold:.4f}",
                candidates_checked=len(candidate_person_ids),
                index_total=self.index.ntotal,
                top_5_sims=[f"{s:.4f}" for s in similarities[0][:min(5, len(similarities[0]))]],
                best_votes=candidate_matches[0]["votes"] if candidate_matches else 0,
                best_gallery_size=candidate_matches[0]["gallery_size"] if candidate_matches else 0,
                margin=f"{(candidate_matches[0]['similarity'] - candidate_matches[1]['similarity']):.4f}" if len(candidate_matches) > 1 else "N/A"
            )

        # If we found a match above threshold
        if best_person_id is not None:
            # Update person record
            self._update_person(best_person_id, embedding, confidence, camera_id)

            # Add to gallery for improved future matching
            self._add_to_gallery(best_person_id, embedding_normalized)

            # Update last seen time
            self.person_last_seen[best_person_id] = current_time

            # Update tracker mapping
            if tracker_id is not None:
                self._update_tracker_mapping(tracker_id, camera_id, best_person_id)

            # Log appearance
            if bbox is not None and frame_number is not None:
                self._log_appearance(
                    best_person_id, frame_number, confidence, bbox,
                    camera_id, tracker_id
                )

            logger.info(
                "person_matched",
                person_id=best_person_id,
                tracker_id=tracker_id,
                similarity=f"{best_similarity:.4f}",
                threshold=f"{best_effective_threshold:.4f}",
                gallery_size=len(self.person_gallery.get(best_person_id, []))
            )

            return best_person_id, False, best_similarity

        # No match - create new person
        person_id = self._create_person(embedding, confidence, camera_id)

        # Add to FAISS index
        self.index.add(embedding_normalized)
        self.index_to_person_id.append(person_id)

        # Initialize gallery with first embedding
        self.person_gallery[person_id] = [embedding_normalized.flatten().copy()]

        # Update last seen time
        self.person_last_seen[person_id] = current_time

        # Update tracker mapping
        if tracker_id is not None:
            self._update_tracker_mapping(tracker_id, camera_id, person_id)

        # Log appearance
        if bbox is not None and frame_number is not None:
            self._log_appearance(
                person_id, frame_number, confidence, bbox,
                camera_id, tracker_id
            )

        logger.info(
            "person_created",
            person_id=person_id,
            tracker_id=tracker_id,
            camera_id=camera_id
        )

        return person_id, True, 1.0

    def _create_person(self, embedding: np.ndarray, confidence: float, camera_id: int) -> int:
        """Create new person entry in database."""
        cursor = self.conn.cursor()

        embedding_blob = embedding.astype('float32').tobytes()

        cursor.execute('''
            INSERT INTO persons (embedding, avg_confidence, last_camera_id)
            VALUES (?, ?, ?)
        ''', (embedding_blob, confidence, camera_id))

        self.conn.commit()
        person_id = cursor.lastrowid

        return person_id

    def _update_person(
        self,
        person_id: int,
        embedding: np.ndarray,
        confidence: float,
        camera_id: int,
        update_weight: float = 0.3
    ):
        """
        Update existing person record with new appearance.

        Uses moving average for embedding: 70% old + 30% new.
        """
        cursor = self.conn.cursor()

        # Get current data
        cursor.execute('''
            SELECT embedding, total_appearances, avg_confidence
            FROM persons WHERE person_id = ?
        ''', (person_id,))

        row = cursor.fetchone()
        if not row:
            return

        old_embedding = np.frombuffer(row['embedding'], dtype=np.float32)
        total_appearances = row['total_appearances']
        avg_confidence = row['avg_confidence']

        # Update embedding with moving average
        new_embedding = (1.0 - update_weight) * old_embedding + update_weight * embedding
        new_embedding_blob = new_embedding.astype('float32').tobytes()

        # Update confidence
        new_avg_confidence = (avg_confidence * total_appearances + confidence) / (total_appearances + 1)

        # Update database
        cursor.execute('''
            UPDATE persons
            SET embedding = ?,
                last_seen = CURRENT_TIMESTAMP,
                total_appearances = total_appearances + 1,
                avg_confidence = ?,
                last_camera_id = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE person_id = ?
        ''', (new_embedding_blob, new_avg_confidence, camera_id, person_id))

        self.conn.commit()

    def _update_tracker_mapping(self, tracker_id: int, camera_id: int, person_id: int):
        """Update tracker ID to person ID mapping."""
        cursor = self.conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO tracker_mapping (tracker_id, camera_id, person_id)
            VALUES (?, ?, ?)
        ''', (tracker_id, camera_id, person_id))

        self.conn.commit()

    def get_person_by_tracker(self, tracker_id: int, camera_id: int) -> Optional[int]:
        """Get person ID for a tracker ID (current session only)."""
        cursor = self.conn.cursor()

        cursor.execute('''
            SELECT person_id FROM tracker_mapping
            WHERE tracker_id = ? AND camera_id = ?
        ''', (tracker_id, camera_id))

        row = cursor.fetchone()
        return row['person_id'] if row else None

    def _log_appearance(
        self,
        person_id: int,
        frame_number: int,
        confidence: float,
        bbox: Tuple[float, float, float, float],
        camera_id: int,
        tracker_id: Optional[int] = None
    ):
        """Log person appearance."""
        cursor = self.conn.cursor()

        cursor.execute('''
            INSERT INTO appearances (
                person_id, camera_id, tracker_id, frame_number, confidence,
                bbox_x, bbox_y, bbox_width, bbox_height
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (person_id, camera_id, tracker_id, frame_number, confidence,
              bbox[0], bbox[1], bbox[2], bbox[3]))

        self.conn.commit()

    def update_demographics(
        self,
        person_id: int,
        age_group: Optional[str] = None,
        gender: Optional[str] = None,
        attention_time: Optional[float] = None,
        attention_state: Optional[str] = None
    ):
        """
        Update demographics and attention info for a person.

        Args:
            person_id: Person ID to update
            age_group: Age group category (e.g., "20-29")
            gender: "Male" or "Female"
            attention_time: Additional attention time to add (seconds)
            attention_state: Current attention state
        """
        cursor = self.conn.cursor()

        updates = []
        params = []

        if gender is not None:
            updates.append("gender = ?")
            params.append(gender)

        if age_group is not None:
            updates.append("age_group = ?")
            params.append(age_group)

        if attention_time is not None:
            updates.append("total_attention_time = total_attention_time + ?")
            params.append(attention_time)

        if updates:
            updates.append("updated_at = CURRENT_TIMESTAMP")
            params.append(person_id)

            cursor.execute(f'''
                UPDATE persons
                SET {", ".join(updates)}
                WHERE person_id = ?
            ''', params)

            self.conn.commit()

    def increment_qualified_impression(self, person_id: int):
        """Increment qualified impressions count for a person."""
        cursor = self.conn.cursor()

        cursor.execute('''
            UPDATE persons
            SET qualified_impressions = qualified_impressions + 1,
                updated_at = CURRENT_TIMESTAMP
            WHERE person_id = ?
        ''', (person_id,))

        self.conn.commit()

    def update_dwell_time(self, person_id: int, dwell_time: float):
        """Update total dwell time for a person."""
        cursor = self.conn.cursor()

        cursor.execute('''
            UPDATE persons
            SET total_dwell_time = total_dwell_time + ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE person_id = ?
        ''', (dwell_time, person_id))

        self.conn.commit()

    def get_analytics_summary(
        self,
        camera_id: Optional[int] = None,
        since: Optional[datetime] = None
    ) -> Dict:
        """
        Get analytics summary.

        Returns:
            Dictionary with analytics data
        """
        cursor = self.conn.cursor()

        where_clauses = []
        params = []

        if camera_id is not None:
            where_clauses.append("last_camera_id = ?")
            params.append(camera_id)

        if since is not None:
            where_clauses.append("last_seen >= ?")
            params.append(since)

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        # Get unique persons count
        cursor.execute(f'''
            SELECT COUNT(*) as unique_persons,
                   SUM(total_appearances) as total_detections,
                   SUM(qualified_impressions) as qualified_impressions,
                   AVG(total_dwell_time) as avg_dwell_time,
                   AVG(total_attention_time) as avg_attention_time
            FROM persons {where_sql}
        ''', params)

        row = cursor.fetchone()

        summary = {
            "unique_persons": row['unique_persons'] or 0,
            "total_detections": row['total_detections'] or 0,
            "qualified_impressions": row['qualified_impressions'] or 0,
            "avg_dwell_time": round(row['avg_dwell_time'] or 0, 2),
            "avg_attention_time": round(row['avg_attention_time'] or 0, 2)
        }

        # Get demographics distribution
        cursor.execute(f'''
            SELECT gender, age_group, COUNT(*) as count
            FROM persons {where_sql}
            GROUP BY gender, age_group
        ''', params)

        gender_dist = {}
        age_dist = {}
        total = 0

        for row in cursor.fetchall():
            gender = row['gender'] or 'Unknown'
            age_group = row['age_group'] or 'Unknown'
            count = row['count']
            total += count

            gender_dist[gender] = gender_dist.get(gender, 0) + count
            age_dist[age_group] = age_dist.get(age_group, 0) + count

        # Convert to percentages
        if total > 0:
            summary["demographics"] = {
                "gender_distribution": {
                    k: round(v / total, 2) for k, v in gender_dist.items()
                },
                "age_distribution": {
                    k: round(v / total, 2) for k, v in age_dist.items()
                }
            }
        else:
            summary["demographics"] = {
                "gender_distribution": {},
                "age_distribution": {}
            }

        return summary

    def get_unique_persons_count(
        self,
        camera_id: Optional[int] = None,
        since: Optional[datetime] = None
    ) -> int:
        """Get count of unique persons."""
        cursor = self.conn.cursor()

        if camera_id is not None and since is not None:
            cursor.execute('''
                SELECT COUNT(DISTINCT person_id)
                FROM appearances
                WHERE camera_id = ? AND timestamp >= ?
            ''', (camera_id, since))
        elif camera_id is not None:
            cursor.execute('''
                SELECT COUNT(DISTINCT person_id)
                FROM appearances
                WHERE camera_id = ?
            ''', (camera_id,))
        else:
            cursor.execute('SELECT COUNT(*) FROM persons')

        return cursor.fetchone()[0]

    def cleanup_old_persons(self, retention_days: int = 30):
        """Delete persons not seen for N days."""
        cursor = self.conn.cursor()

        cutoff_date = datetime.now() - timedelta(days=retention_days)

        # Get persons to delete
        cursor.execute('''
            SELECT person_id FROM persons
            WHERE last_seen < ?
        ''', (cutoff_date,))

        persons_to_delete = [row[0] for row in cursor.fetchall()]

        if persons_to_delete:
            placeholders = ','.join('?' * len(persons_to_delete))

            # Delete appearances
            cursor.execute(f'''
                DELETE FROM appearances
                WHERE person_id IN ({placeholders})
            ''', persons_to_delete)

            # Delete tracker mappings
            cursor.execute(f'''
                DELETE FROM tracker_mapping
                WHERE person_id IN ({placeholders})
            ''', persons_to_delete)

            # Delete persons
            cursor.execute(f'''
                DELETE FROM persons
                WHERE person_id IN ({placeholders})
            ''', persons_to_delete)

            self.conn.commit()

            # Clean up gallery for deleted persons
            for pid in persons_to_delete:
                self.person_gallery.pop(pid, None)
                self.person_last_seen.pop(pid, None)

            # Rebuild FAISS index
            self._rebuild_index_from_db()

            logger.info(
                "old_persons_cleaned",
                deleted_count=len(persons_to_delete),
                retention_days=retention_days
            )

    def clear_tracker_mappings(self):
        """Clear tracker mappings (for session restart)."""
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM tracker_mapping')
        self.conn.commit()
        logger.info("tracker_mappings_cleared")

    def get_gallery_stats(self) -> Dict:
        """Get statistics about the multi-shot gallery."""
        if not self.person_gallery:
            return {
                "total_persons": 0,
                "total_embeddings": 0,
                "avg_embeddings_per_person": 0.0
            }

        total_embeddings = sum(len(g) for g in self.person_gallery.values())
        return {
            "total_persons": len(self.person_gallery),
            "total_embeddings": total_embeddings,
            "avg_embeddings_per_person": total_embeddings / len(self.person_gallery) if self.person_gallery else 0.0,
            "max_gallery_size": max(len(g) for g in self.person_gallery.values()) if self.person_gallery else 0,
            "recently_seen_count": sum(
                1 for t in self.person_last_seen.values()
                if time.time() - t < RECENT_SEEN_BOOST_SECONDS
            )
        }

    def get_attention_analytics(
        self,
        camera_id: Optional[int] = None,
        since: Optional[datetime] = None
    ) -> Dict:
        """
        Get attention-focused analytics summary.

        Returns:
            Dictionary with attention metrics:
            - total_qualified_impressions: Total QI across all persons
            - unique_persons_with_qi: Persons who had at least one QI
            - avg_attention_time: Average attention time per person
            - max_attention_time: Maximum attention time observed
            - attention_by_demographics: QI breakdown by gender/age
        """
        cursor = self.conn.cursor()

        where_clauses = []
        params = []

        if camera_id is not None:
            where_clauses.append("last_camera_id = ?")
            params.append(camera_id)

        if since is not None:
            where_clauses.append("last_seen >= ?")
            params.append(since)

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        # Get attention summary
        cursor.execute(f'''
            SELECT
                SUM(qualified_impressions) as total_qi,
                COUNT(CASE WHEN qualified_impressions > 0 THEN 1 END) as persons_with_qi,
                AVG(total_attention_time) as avg_attention_time,
                MAX(total_attention_time) as max_attention_time,
                SUM(total_dwell_time) as total_dwell_time,
                COUNT(*) as total_persons
            FROM persons {where_sql}
        ''', params)

        row = cursor.fetchone()

        summary = {
            "total_qualified_impressions": row['total_qi'] or 0,
            "unique_persons_with_qi": row['persons_with_qi'] or 0,
            "avg_attention_time": round(row['avg_attention_time'] or 0, 2),
            "max_attention_time": round(row['max_attention_time'] or 0, 2),
            "total_dwell_time": round(row['total_dwell_time'] or 0, 2),
            "total_persons": row['total_persons'] or 0
        }

        # Calculate QI rate (persons with QI / total persons)
        if summary["total_persons"] > 0:
            summary["qi_rate"] = round(
                summary["unique_persons_with_qi"] / summary["total_persons"], 3
            )
        else:
            summary["qi_rate"] = 0.0

        # Get QI breakdown by gender
        cursor.execute(f'''
            SELECT
                gender,
                SUM(qualified_impressions) as qi_count,
                COUNT(*) as person_count
            FROM persons {where_sql}
            GROUP BY gender
        ''', params)

        gender_qi = {}
        for row in cursor.fetchall():
            gender = row['gender'] or 'Unknown'
            gender_qi[gender] = {
                "qualified_impressions": row['qi_count'] or 0,
                "person_count": row['person_count'] or 0
            }

        summary["qi_by_gender"] = gender_qi

        # Get QI breakdown by age group
        cursor.execute(f'''
            SELECT
                age_group,
                SUM(qualified_impressions) as qi_count,
                COUNT(*) as person_count
            FROM persons {where_sql}
            GROUP BY age_group
        ''', params)

        age_qi = {}
        for row in cursor.fetchall():
            age_group = row['age_group'] or 'Unknown'
            age_qi[age_group] = {
                "qualified_impressions": row['qi_count'] or 0,
                "person_count": row['person_count'] or 0
            }

        summary["qi_by_age_group"] = age_qi

        return summary

    def finalize_person_session(
        self,
        person_id: int,
        dwell_time: float,
        attention_time: float,
        qualified_impression: bool = False
    ):
        """
        Finalize a person's session when they leave the camera view.

        Args:
            person_id: Person ID
            dwell_time: Total time person was visible (seconds)
            attention_time: Time person spent looking at camera (seconds)
            qualified_impression: Whether this session resulted in a QI
        """
        cursor = self.conn.cursor()

        if qualified_impression:
            cursor.execute('''
                UPDATE persons
                SET total_dwell_time = total_dwell_time + ?,
                    total_attention_time = total_attention_time + ?,
                    qualified_impressions = qualified_impressions + 1,
                    updated_at = CURRENT_TIMESTAMP
                WHERE person_id = ?
            ''', (dwell_time, attention_time, person_id))
        else:
            cursor.execute('''
                UPDATE persons
                SET total_dwell_time = total_dwell_time + ?,
                    total_attention_time = total_attention_time + ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE person_id = ?
            ''', (dwell_time, attention_time, person_id))

        self.conn.commit()

        logger.info(
            "person_session_finalized",
            person_id=person_id,
            dwell_time=f"{dwell_time:.1f}s",
            attention_time=f"{attention_time:.1f}s",
            qualified_impression=qualified_impression
        )

    def log_appearance_with_attention(
        self,
        person_id: int,
        frame_number: int,
        confidence: float,
        bbox: Tuple[float, float, float, float],
        camera_id: int,
        tracker_id: Optional[int] = None,
        attention_state: Optional[str] = None,
        head_yaw: Optional[float] = None,
        head_pitch: Optional[float] = None,
        head_roll: Optional[float] = None,
        age_group: Optional[str] = None,
        gender: Optional[str] = None
    ):
        """
        Log person appearance with full attention data.

        Args:
            person_id: Person ID
            frame_number: Current frame number
            confidence: Detection confidence
            bbox: Bounding box (x, y, width, height)
            camera_id: Camera ID
            tracker_id: Tracker ID
            attention_state: Current attention state (NOT_LOOKING, LOOKING, ENGAGED)
            head_yaw: Head yaw angle in degrees
            head_pitch: Head pitch angle in degrees
            head_roll: Head roll angle in degrees
            age_group: Age group
            gender: Gender
        """
        cursor = self.conn.cursor()

        cursor.execute('''
            INSERT INTO appearances (
                person_id, camera_id, tracker_id, frame_number, confidence,
                bbox_x, bbox_y, bbox_width, bbox_height,
                attention_state, head_yaw, head_pitch, head_roll,
                age_group, gender
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            person_id, camera_id, tracker_id, frame_number, confidence,
            bbox[0], bbox[1], bbox[2], bbox[3],
            attention_state, head_yaw, head_pitch, head_roll,
            age_group, gender
        ))

        self.conn.commit()

    def close(self):
        """Close database and save index."""
        # Log gallery stats before closing
        gallery_stats = self.get_gallery_stats()
        logger.info("gallery_stats_at_close", **gallery_stats)

        self.save_index()
        self.conn.close()
        logger.info("person_database_closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
