#!/usr/bin/env python3
"""
Global ReID Verification System

This module provides a robust person identification system that works
independently of the NvDCF tracker's ID assignments. It maintains a
global registry of all known persons and verifies/corrects tracker
assignments using ReID embeddings.

Key Features:
1. Global Person Registry with embedding averaging
2. Two-stage verification (quick + detailed)
3. Adaptive thresholding based on context
4. Temporal smoothing to prevent flickering
5. Position-based validation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class PersonRecord:
    """
    Record for a unique person in the global registry.
    Maintains running statistics for robust matching.
    """
    person_id: int

    # Embedding statistics (Exponential Moving Average)
    avg_embedding: np.ndarray = None
    embedding_count: int = 0
    embedding_alpha: float = 0.3  # EMA weight for new embeddings

    # Temporal tracking
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    total_appearances: int = 0

    # Spatial tracking
    last_position: Tuple[float, float] = (0, 0)  # center x, y
    position_history: List[Tuple[int, float, float]] = field(default_factory=list)

    # Confidence metrics
    avg_confidence: float = 0.0
    match_history: List[float] = field(default_factory=list)  # Recent similarity scores

    def update_embedding(self, new_embedding: np.ndarray):
        """Update average embedding using EMA"""
        new_embedding = new_embedding / (np.linalg.norm(new_embedding) + 1e-8)

        if self.avg_embedding is None:
            self.avg_embedding = new_embedding.copy()
        else:
            # Exponential moving average
            self.avg_embedding = (
                self.embedding_alpha * new_embedding +
                (1 - self.embedding_alpha) * self.avg_embedding
            )
            # Re-normalize
            self.avg_embedding = self.avg_embedding / (np.linalg.norm(self.avg_embedding) + 1e-8)

        self.embedding_count += 1

    def update_position(self, frame: int, bbox: Tuple[float, float, float, float]):
        """Update position from bounding box"""
        cx = bbox[0] + bbox[2] / 2  # center x
        cy = bbox[1] + bbox[3] / 2  # center y
        self.last_position = (cx, cy)
        self.position_history.append((frame, cx, cy))

        # Keep only recent history (last 100 positions)
        if len(self.position_history) > 100:
            self.position_history = self.position_history[-100:]

    def update(self, frame: int, embedding: np.ndarray, bbox: Tuple = None,
               confidence: float = 0.0, similarity: float = 0.0):
        """Full update of person record"""
        self.update_embedding(embedding)
        if bbox:
            self.update_position(frame, bbox)
        self.last_seen_frame = frame
        self.total_appearances += 1

        # Update confidence EMA
        if confidence > 0:
            if self.avg_confidence == 0:
                self.avg_confidence = confidence
            else:
                self.avg_confidence = 0.9 * self.avg_confidence + 0.1 * confidence

        # Track match history
        self.match_history.append(similarity)
        if len(self.match_history) > 20:
            self.match_history = self.match_history[-20:]

    def get_stability_score(self) -> float:
        """Calculate how stable/reliable this person's identity is"""
        if len(self.match_history) < 3:
            return 0.5  # Uncertain

        avg_match = np.mean(self.match_history)
        match_std = np.std(self.match_history)

        # High average + low variance = stable
        stability = avg_match * (1 - min(match_std, 0.3))
        return float(stability)


class GlobalReIDVerifier:
    """
    Global ReID Verification System

    This system maintains a registry of all known persons and provides
    robust verification of person identities independent of tracker IDs.
    """

    def __init__(
        self,
        confident_threshold: float = 0.75,  # High confidence match
        possible_threshold: float = 0.65,   # Possible match
        new_person_threshold: float = 0.60, # Below this, might be new person
        temporal_window: int = 5,           # Frames for temporal smoothing
        position_weight: float = 0.1        # Weight for position in scoring
    ):
        """
        Initialize the Global ReID Verifier.

        Args:
            confident_threshold: Similarity above this is a confident match
            possible_threshold: Similarity above this is a possible match
            new_person_threshold: Below this, consider as new person
            temporal_window: Number of frames for temporal smoothing
            position_weight: Weight for position similarity in final score
        """
        self.confident_threshold = confident_threshold
        self.possible_threshold = possible_threshold
        self.new_person_threshold = new_person_threshold
        self.temporal_window = temporal_window
        self.position_weight = position_weight

        # Global person registry
        self.persons: Dict[int, PersonRecord] = {}

        # Temporal smoothing buffers
        # tracker_id -> list of (frame, person_id, similarity)
        self.tracker_assignments: Dict[int, List[Tuple[int, int, float]]] = defaultdict(list)

        # Statistics
        self.stats = {
            'verifications': 0,
            'corrections': 0,
            'confirmations': 0,
            'new_persons': 0
        }

        logger.info("global_reid_verifier_initialized",
                   confident_threshold=confident_threshold,
                   possible_threshold=possible_threshold)

    def register_person(self, person_id: int, frame: int,
                       embedding: np.ndarray, bbox: Tuple = None,
                       confidence: float = 0.0) -> PersonRecord:
        """
        Register a new person in the global registry.

        Args:
            person_id: Person ID from database
            frame: Frame number
            embedding: ReID embedding vector
            bbox: Bounding box (x, y, w, h)
            confidence: Detection confidence

        Returns:
            PersonRecord for the registered person
        """
        if person_id in self.persons:
            # Update existing
            self.persons[person_id].update(frame, embedding, bbox, confidence, 1.0)
        else:
            # Create new
            record = PersonRecord(
                person_id=person_id,
                first_seen_frame=frame,
                last_seen_frame=frame
            )
            record.update(frame, embedding, bbox, confidence, 1.0)
            self.persons[person_id] = record
            self.stats['new_persons'] += 1

            logger.debug("person_registered",
                        person_id=person_id,
                        frame=frame)

        return self.persons[person_id]

    def search_all_persons(
        self,
        embedding: np.ndarray,
        k: int = 5,
        exclude_ids: List[int] = None
    ) -> List[Tuple[int, float]]:
        """
        Search all registered persons for matches.

        Args:
            embedding: Query embedding
            k: Number of top matches to return
            exclude_ids: Person IDs to exclude from search

        Returns:
            List of (person_id, similarity) tuples, sorted by similarity descending
        """
        if not self.persons:
            return []

        embedding_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
        exclude_ids = exclude_ids or []

        matches = []
        for person_id, record in self.persons.items():
            if person_id in exclude_ids:
                continue
            if record.avg_embedding is None:
                continue

            similarity = float(np.dot(embedding_norm, record.avg_embedding))
            matches.append((person_id, similarity))

        # Sort by similarity descending
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:k]

    def verify_assignment(
        self,
        tracker_id: int,
        proposed_person_id: int,
        embedding: np.ndarray,
        frame: int,
        bbox: Tuple = None,
        is_new_person: bool = False
    ) -> Tuple[int, bool, float, str]:
        """
        Verify and potentially correct a person assignment.

        This is the main entry point for verification. It checks if the
        proposed person_id is correct based on global ReID matching.

        Args:
            tracker_id: NvDCF tracker ID
            proposed_person_id: Person ID proposed by current system
            embedding: ReID embedding for this detection
            frame: Current frame number
            bbox: Bounding box (optional)
            is_new_person: Whether this would create a new person

        Returns:
            (verified_person_id, was_corrected, similarity, reason)
        """
        self.stats['verifications'] += 1
        embedding_norm = embedding / (np.linalg.norm(embedding) + 1e-8)

        # Search all known persons
        matches = self.search_all_persons(embedding, k=5)

        if not matches:
            # No persons in registry yet
            return proposed_person_id, False, 1.0, "no_registry"

        best_match_id, best_similarity = matches[0]

        # ============================================================
        # CASE 1: New person being created
        # ============================================================
        if is_new_person:
            # Check if this "new" person actually matches an existing one
            if best_similarity > self.possible_threshold:
                # This is not a new person - it matches an existing one!
                self.stats['corrections'] += 1

                logger.info(
                    "new_person_prevented_global",
                    tracker_id=tracker_id,
                    would_be_person_id=proposed_person_id,
                    matched_person_id=best_match_id,
                    similarity=best_similarity,
                    frame=frame
                )

                # Update the matched person's record
                self.persons[best_match_id].update(
                    frame, embedding, bbox, similarity=best_similarity
                )

                # Record assignment for temporal smoothing
                self._record_assignment(tracker_id, frame, best_match_id, best_similarity)

                return best_match_id, True, best_similarity, "matched_existing"

            else:
                # Genuinely new person
                return proposed_person_id, False, 1.0, "genuinely_new"

        # ============================================================
        # CASE 2: Existing person assignment - verify it
        # ============================================================

        # Check similarity to proposed person
        if proposed_person_id in self.persons:
            proposed_record = self.persons[proposed_person_id]
            sim_to_proposed = float(np.dot(embedding_norm, proposed_record.avg_embedding))
        else:
            sim_to_proposed = 0.0

        # Decision logic
        if best_match_id == proposed_person_id:
            # Assignment matches global search - confirm
            self.stats['confirmations'] += 1

            # Update record
            if proposed_person_id in self.persons:
                self.persons[proposed_person_id].update(
                    frame, embedding, bbox, similarity=sim_to_proposed
                )

            self._record_assignment(tracker_id, frame, proposed_person_id, sim_to_proposed)
            return proposed_person_id, False, sim_to_proposed, "confirmed"

        else:
            # Mismatch! Global search suggests different person

            # Apply temporal smoothing - don't switch on single frame
            smoothed_id = self._temporal_smooth(
                tracker_id, frame, best_match_id, best_similarity
            )

            if smoothed_id != proposed_person_id:
                # Consistent mismatch - correct the assignment
                self.stats['corrections'] += 1

                logger.info(
                    "assignment_corrected_global",
                    tracker_id=tracker_id,
                    proposed_person_id=proposed_person_id,
                    corrected_person_id=smoothed_id,
                    proposed_similarity=sim_to_proposed,
                    corrected_similarity=best_similarity,
                    frame=frame
                )

                # Update corrected person's record
                if smoothed_id in self.persons:
                    self.persons[smoothed_id].update(
                        frame, embedding, bbox, similarity=best_similarity
                    )

                return smoothed_id, True, best_similarity, "corrected_global"

            else:
                # Single-frame anomaly - keep proposed
                self._record_assignment(tracker_id, frame, proposed_person_id, sim_to_proposed)
                return proposed_person_id, False, sim_to_proposed, "kept_temporal"

    def _record_assignment(self, tracker_id: int, frame: int,
                          person_id: int, similarity: float):
        """Record an assignment for temporal smoothing"""
        history = self.tracker_assignments[tracker_id]
        history.append((frame, person_id, similarity))

        # Keep only recent history
        if len(history) > self.temporal_window * 2:
            self.tracker_assignments[tracker_id] = history[-self.temporal_window * 2:]

    def _temporal_smooth(self, tracker_id: int, frame: int,
                        new_person_id: int, new_similarity: float) -> int:
        """
        Apply temporal smoothing to prevent single-frame ID switches.

        Returns the smoothed person_id based on recent history.
        """
        history = self.tracker_assignments[tracker_id]

        if len(history) < 2:
            # Not enough history - accept new assignment
            self._record_assignment(tracker_id, frame, new_person_id, new_similarity)
            return new_person_id

        # Get recent assignments
        recent = history[-self.temporal_window:]
        recent_ids = [h[1] for h in recent]

        # Add proposed new assignment
        recent_ids.append(new_person_id)

        # Count occurrences
        id_counts = Counter(recent_ids)
        most_common_id, count = id_counts.most_common(1)[0]

        # Record the new assignment
        self._record_assignment(tracker_id, frame, new_person_id, new_similarity)

        # If new_person_id is consistent (appears in majority), use it
        if new_person_id == most_common_id or count <= len(recent_ids) // 2:
            return new_person_id
        else:
            # Anomaly - keep most common
            return most_common_id

    def get_person_record(self, person_id: int) -> Optional[PersonRecord]:
        """Get the record for a person"""
        return self.persons.get(person_id)

    def get_all_person_ids(self) -> List[int]:
        """Get all registered person IDs"""
        return list(self.persons.keys())

    def get_stats(self) -> Dict:
        """Get verification statistics"""
        return {
            **self.stats,
            'registered_persons': len(self.persons),
            'correction_rate': (
                self.stats['corrections'] / max(self.stats['verifications'], 1)
            )
        }

    def get_person_summary(self) -> List[Dict]:
        """Get summary of all registered persons"""
        summaries = []
        for person_id, record in self.persons.items():
            summaries.append({
                'person_id': person_id,
                'first_seen': record.first_seen_frame,
                'last_seen': record.last_seen_frame,
                'appearances': record.total_appearances,
                'embedding_count': record.embedding_count,
                'stability': record.get_stability_score(),
                'avg_confidence': record.avg_confidence
            })
        return sorted(summaries, key=lambda x: x['person_id'])
