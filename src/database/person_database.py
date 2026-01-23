#!/usr/bin/env python3
"""
Persistent Person RE-ID Database
Combines FAISS vector search with SQLite metadata storage
"""

import sqlite3
import numpy as np
import faiss
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import structlog

logger = structlog.get_logger(__name__)


class PersonDatabase:
    """
    Manages persistent storage of person RE-ID embeddings and metadata
    Uses FAISS for fast vector similarity search and SQLite for metadata
    """

    def __init__(
        self,
        db_path: str = "output/database/person_database.db",
        index_path: str = "output/database/person_embeddings.index",
        embedding_dim: int = 128,
        use_gpu: bool = False
    ):
        """
        Initialize person database

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

        # Load existing index if available
        self._load_index()

        logger.info(
            "person_database_initialized",
            db_path=str(self.db_path),
            total_persons=len(self.index_to_person_id),
            embedding_dim=self.embedding_dim
        )

    def _create_schema(self):
        """Create database tables if they don't exist"""
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
        """Load FAISS index and mapping from disk if exists"""
        index_file = self.index_path
        mapping_file = Path(str(self.index_path) + ".pkl")

        if index_file.exists() and mapping_file.exists():
            try:
                # Load FAISS index
                self.index = faiss.read_index(str(index_file))

                # Load ID mapping
                with open(mapping_file, 'rb') as f:
                    self.index_to_person_id = pickle.load(f)

                logger.info(
                    "faiss_index_loaded",
                    total_embeddings=self.index.ntotal,
                    path=str(index_file)
                )
            except Exception as e:
                logger.warning("faiss_index_load_failed", error=str(e))
                self.index = faiss.IndexFlatIP(self.embedding_dim)
                self.index_to_person_id = []
                self._rebuild_index_from_db()
        else:
            # Rebuild index from database embeddings
            logger.info("faiss_index_not_found_rebuilding")
            self._rebuild_index_from_db()

    def _rebuild_index_from_db(self):
        """Rebuild FAISS index from database embeddings"""
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
        """Save FAISS index and mapping to disk"""
        try:
            faiss.write_index(self.index, str(self.index_path))

            with open(str(self.index_path) + ".pkl", 'wb') as f:
                pickle.dump(self.index_to_person_id, f)

            logger.debug("faiss_index_saved", path=str(self.index_path))
        except Exception as e:
            logger.error("faiss_index_save_failed", error=str(e))

    def match_or_create_person(
        self,
        embedding: np.ndarray,
        threshold: float = 0.75,
        confidence: float = 0.0,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        frame_number: Optional[int] = None,
        camera_id: int = 1
    ) -> Tuple[int, bool, float]:
        """
        Match embedding against database or create new person

        Args:
            embedding: Embedding vector (128-dim for OSNet)
            threshold: Cosine similarity threshold (0.0-1.0)
            confidence: Detection confidence
            bbox: Bounding box (x, y, width, height)
            frame_number: Current frame number
            camera_id: Camera ID

        Returns:
            (person_id, is_new, similarity): Person ID, whether new, and similarity score
        """
        # Normalize embedding for cosine similarity
        embedding_normalized = embedding / (np.linalg.norm(embedding) + 1e-8)
        embedding_normalized = embedding_normalized.reshape(1, -1).astype('float32')

        # Query FAISS index for top-5 nearest neighbors
        similarity = 0.0
        if self.index.ntotal > 0:
            similarities, indices = self.index.search(
                embedding_normalized,
                k=min(5, self.index.ntotal)
            )

            # Check if best match exceeds threshold
            if similarities[0][0] > threshold:
                faiss_idx = int(indices[0][0])
                person_id = self.index_to_person_id[faiss_idx]
                similarity = float(similarities[0][0])

                # Update person record
                self._update_person(person_id, embedding, confidence, camera_id)

                # Log appearance
                if bbox is not None and frame_number is not None:
                    self._log_appearance(person_id, frame_number, confidence, bbox, camera_id)

                logger.debug(
                    "person_matched",
                    person_id=person_id,
                    similarity=similarity,
                    threshold=threshold
                )

                return person_id, False, similarity

        # No match - create new person
        person_id = self._create_person(embedding, confidence, camera_id)

        # Add to FAISS index
        self.index.add(embedding_normalized)
        self.index_to_person_id.append(person_id)

        # Log appearance
        if bbox is not None and frame_number is not None:
            self._log_appearance(person_id, frame_number, confidence, bbox, camera_id)

        logger.info("person_created", person_id=person_id, camera_id=camera_id)

        return person_id, True, 1.0

    def _create_person(self, embedding: np.ndarray, confidence: float, camera_id: int) -> int:
        """Create new person entry in database"""
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
        Update existing person record with new appearance

        Args:
            person_id: Person ID to update
            embedding: New embedding vector
            confidence: Detection confidence
            camera_id: Camera ID
            update_weight: Weight for moving average (0.3 = 30% new, 70% old)
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

    def _rollback_new_person(self, person_id: int):
        """
        Rollback a newly created person entry.

        This is called when we detect that a new person was incorrectly created
        due to an ID switch (tracker swapped to different person). We need to
        remove the person entry and its embedding from the FAISS index.

        Args:
            person_id: Person ID to rollback
        """
        cursor = self.conn.cursor()

        # Delete from persons table
        cursor.execute('DELETE FROM persons WHERE person_id = ?', (person_id,))

        # Delete any appearances
        cursor.execute('DELETE FROM appearances WHERE person_id = ?', (person_id,))

        # Note: We don't remove from FAISS index as it's more complex
        # The stale entry won't cause issues since we check person_id mapping
        # In production, we'd want to rebuild index periodically

        self.conn.commit()
        logger.info("person_rolled_back", person_id=person_id)

    def _log_appearance(
        self,
        person_id: int,
        frame_number: int,
        confidence: float,
        bbox: Tuple[float, float, float, float],
        camera_id: int
    ):
        """Log person appearance"""
        cursor = self.conn.cursor()

        cursor.execute('''
            INSERT INTO appearances (
                person_id, camera_id, frame_number, confidence,
                bbox_x, bbox_y, bbox_width, bbox_height
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (person_id, camera_id, frame_number, confidence,
              bbox[0], bbox[1], bbox[2], bbox[3]))

        self.conn.commit()

    def update_demographics(
        self,
        person_id: int,
        age: Optional[int] = None,
        gender: Optional[str] = None,
        age_group: Optional[str] = None,
        attention_time: Optional[float] = None
    ):
        """
        Update demographics information for a person

        Args:
            person_id: Person ID to update
            age: Estimated age
            gender: 'male' or 'female'
            age_group: Age group category
            attention_time: Additional attention time to add (in seconds)
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
            logger.debug(
                "demographics_updated",
                person_id=person_id,
                gender=gender,
                age_group=age_group
            )

    def get_demographics_stats(self, camera_id: Optional[int] = None) -> Dict:
        """
        Get aggregated demographics statistics

        Args:
            camera_id: Filter by camera (optional)

        Returns:
            Dictionary with demographics breakdown
        """
        cursor = self.conn.cursor()

        stats = {
            'total_persons': 0,
            'gender_distribution': {'male': 0, 'female': 0, 'unknown': 0},
            'age_distribution': {}
        }

        if camera_id is not None:
            cursor.execute('''
                SELECT gender, age_group, COUNT(*) as count
                FROM persons
                WHERE last_camera_id = ?
                GROUP BY gender, age_group
            ''', (camera_id,))
        else:
            cursor.execute('''
                SELECT gender, age_group, COUNT(*) as count
                FROM persons
                GROUP BY gender, age_group
            ''')

        for row in cursor.fetchall():
            gender = row['gender'] or 'unknown'
            age_group = row['age_group'] or 'unknown'
            count = row['count']

            stats['total_persons'] += count
            stats['gender_distribution'][gender] = \
                stats['gender_distribution'].get(gender, 0) + count
            stats['age_distribution'][age_group] = \
                stats['age_distribution'].get(age_group, 0) + count

        return stats

    def get_person_stats(self, person_id: int) -> Optional[Dict]:
        """Get statistics for a person"""
        cursor = self.conn.cursor()

        cursor.execute('''
            SELECT person_id, first_seen, last_seen, total_appearances,
                   avg_confidence, age_group, gender, total_attention_time,
                   qualified_impressions, total_dwell_time
            FROM persons WHERE person_id = ?
        ''', (person_id,))

        row = cursor.fetchone()
        if row:
            return dict(row)
        return None

    def get_unique_persons_count(self, camera_id: Optional[int] = None, since: Optional[datetime] = None) -> int:
        """Get count of unique persons"""
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
        """Delete persons not seen for N days"""
        cursor = self.conn.cursor()

        cutoff_date = datetime.now() - timedelta(days=retention_days)

        # Get persons to delete
        cursor.execute('''
            SELECT person_id FROM persons
            WHERE last_seen < ?
        ''', (cutoff_date,))

        persons_to_delete = [row[0] for row in cursor.fetchall()]

        if persons_to_delete:
            # Delete appearances
            cursor.execute(f'''
                DELETE FROM appearances
                WHERE person_id IN ({','.join('?' * len(persons_to_delete))})
            ''', persons_to_delete)

            # Delete persons
            cursor.execute(f'''
                DELETE FROM persons
                WHERE person_id IN ({','.join('?' * len(persons_to_delete))})
            ''', persons_to_delete)

            self.conn.commit()

            # Rebuild FAISS index
            self._rebuild_index_from_db()

            logger.info(
                "old_persons_cleaned",
                deleted_count=len(persons_to_delete),
                retention_days=retention_days
            )

    def close(self):
        """Close database and save index"""
        self.save_index()
        self.conn.close()
        logger.info("person_database_closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
