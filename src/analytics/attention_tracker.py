#!/usr/bin/env python3
"""
IP AI v3 - Attention Tracker

Tracks head pose and engagement state per person using a state machine.
Counts qualified impressions (>2 seconds looking at display).
"""

import time
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import structlog

logger = structlog.get_logger(__name__)


class AttentionState(Enum):
    """Attention state machine states."""
    NOT_LOOKING = auto()   # Head not facing display
    LOOKING = auto()       # Head facing display, duration < threshold
    ENGAGED = auto()       # Head facing display, duration >= threshold (qualified)


@dataclass
class PersonAttention:
    """Attention tracking data for a single person."""
    person_id: int
    state: AttentionState = AttentionState.NOT_LOOKING

    # Current looking session
    looking_start_time: Optional[float] = None  # When started looking

    # Accumulated metrics
    total_attention_time: float = 0.0           # Total time looking (seconds)
    qualified_impressions: int = 0              # Count of ENGAGED transitions

    # Last known head pose
    last_yaw: float = 0.0
    last_pitch: float = 0.0
    last_roll: float = 0.0

    # Dwell time tracking
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    @property
    def dwell_time(self) -> float:
        """Total time person has been tracked (seconds)."""
        return self.last_seen - self.first_seen

    @property
    def current_looking_duration(self) -> float:
        """Current continuous looking duration (seconds)."""
        if self.looking_start_time is None:
            return 0.0
        return time.time() - self.looking_start_time


class AttentionTracker:
    """
    Tracks attention state for multiple persons using head pose angles.

    State Machine:
        NOT_LOOKING -> LOOKING: When head faces display (|yaw| < threshold, |pitch| < threshold)
        LOOKING -> ENGAGED: After looking for > engagement_threshold seconds
        LOOKING/ENGAGED -> NOT_LOOKING: When head turns away

    A "Qualified Impression" is counted each time a person transitions to ENGAGED.
    """

    def __init__(
        self,
        yaw_threshold: float = 30.0,       # degrees
        pitch_threshold: float = 20.0,     # degrees
        engagement_threshold: float = 2.0  # seconds for qualified impression
    ):
        """
        Initialize attention tracker.

        Args:
            yaw_threshold: Maximum absolute yaw angle to be considered "looking"
            pitch_threshold: Maximum absolute pitch angle to be considered "looking"
            engagement_threshold: Seconds of continuous looking for ENGAGED state
        """
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.engagement_threshold = engagement_threshold

        # Track attention per person_id
        self._persons: Dict[int, PersonAttention] = {}

        logger.info(
            "attention_tracker_initialized",
            yaw_threshold=yaw_threshold,
            pitch_threshold=pitch_threshold,
            engagement_threshold=engagement_threshold
        )

    def _is_looking(self, yaw: float, pitch: float) -> bool:
        """
        Determine if head pose indicates looking at display.

        Args:
            yaw: Head yaw angle in degrees (left/right rotation)
            pitch: Head pitch angle in degrees (up/down rotation)

        Returns:
            True if person is looking at display, False if angles are None or out of range
        """
        # Handle None values (no head pose data available)
        if yaw is None or pitch is None:
            return False
        return abs(yaw) < self.yaw_threshold and abs(pitch) < self.pitch_threshold

    def update(
        self,
        person_id: int,
        yaw: float,
        pitch: float,
        roll: float,
        timestamp: Optional[float] = None
    ) -> Tuple[AttentionState, bool]:
        """
        Update attention state for a person based on head pose.

        Args:
            person_id: Persistent person ID from database
            yaw: Head yaw angle in degrees
            pitch: Head pitch angle in degrees
            roll: Head roll angle in degrees (not used for attention)
            timestamp: Current timestamp (default: time.time())

        Returns:
            Tuple of (current_state, is_new_qualified_impression)
        """
        if timestamp is None:
            timestamp = time.time()

        # Get or create person attention record
        if person_id not in self._persons:
            self._persons[person_id] = PersonAttention(
                person_id=person_id,
                first_seen=timestamp
            )

        person = self._persons[person_id]
        person.last_seen = timestamp
        person.last_yaw = yaw
        person.last_pitch = pitch
        person.last_roll = roll

        is_looking = self._is_looking(yaw, pitch)
        new_qualified = False

        # State machine transitions
        if person.state == AttentionState.NOT_LOOKING:
            if is_looking:
                # Transition to LOOKING
                person.state = AttentionState.LOOKING
                person.looking_start_time = timestamp
                logger.debug(
                    "attention_state_change",
                    person_id=person_id,
                    old_state="NOT_LOOKING",
                    new_state="LOOKING",
                    yaw=f"{yaw:.1f}",
                    pitch=f"{pitch:.1f}"
                )

        elif person.state == AttentionState.LOOKING:
            if not is_looking:
                # Looked away - back to NOT_LOOKING
                if person.looking_start_time is not None:
                    duration = timestamp - person.looking_start_time
                    person.total_attention_time += duration
                person.state = AttentionState.NOT_LOOKING
                person.looking_start_time = None
                logger.debug(
                    "attention_state_change",
                    person_id=person_id,
                    old_state="LOOKING",
                    new_state="NOT_LOOKING"
                )
            elif person.looking_start_time is not None:
                duration = timestamp - person.looking_start_time
                if duration >= self.engagement_threshold:
                    # Transition to ENGAGED (qualified impression!)
                    person.state = AttentionState.ENGAGED
                    person.qualified_impressions += 1
                    new_qualified = True
                    logger.info(
                        "qualified_impression",
                        person_id=person_id,
                        total_impressions=person.qualified_impressions,
                        duration=f"{duration:.1f}s"
                    )

        elif person.state == AttentionState.ENGAGED:
            if not is_looking:
                # Looked away - back to NOT_LOOKING
                if person.looking_start_time is not None:
                    duration = timestamp - person.looking_start_time
                    person.total_attention_time += duration
                person.state = AttentionState.NOT_LOOKING
                person.looking_start_time = None
                logger.debug(
                    "attention_state_change",
                    person_id=person_id,
                    old_state="ENGAGED",
                    new_state="NOT_LOOKING",
                    total_attention_time=f"{person.total_attention_time:.1f}s"
                )
            # If still looking, stay ENGAGED (already counted impression)

        return person.state, new_qualified

    def get_person_attention(self, person_id: int) -> Optional[PersonAttention]:
        """Get attention data for a specific person."""
        return self._persons.get(person_id)

    def get_all_attention(self) -> Dict[int, PersonAttention]:
        """Get attention data for all tracked persons."""
        return self._persons.copy()

    def get_summary(self) -> Dict:
        """
        Get summary statistics for all tracked persons.

        Returns:
            Dictionary with aggregate attention metrics
        """
        if not self._persons:
            return {
                "total_tracked": 0,
                "currently_looking": 0,
                "currently_engaged": 0,
                "total_qualified_impressions": 0,
                "avg_attention_time": 0.0,
                "avg_dwell_time": 0.0
            }

        currently_looking = sum(
            1 for p in self._persons.values()
            if p.state == AttentionState.LOOKING
        )
        currently_engaged = sum(
            1 for p in self._persons.values()
            if p.state == AttentionState.ENGAGED
        )
        total_impressions = sum(p.qualified_impressions for p in self._persons.values())

        # Calculate averages (include current looking time for accuracy)
        current_time = time.time()
        attention_times = []
        dwell_times = []

        for p in self._persons.values():
            att_time = p.total_attention_time
            if p.looking_start_time is not None:
                att_time += current_time - p.looking_start_time
            attention_times.append(att_time)
            dwell_times.append(p.dwell_time)

        return {
            "total_tracked": len(self._persons),
            "currently_looking": currently_looking,
            "currently_engaged": currently_engaged,
            "total_qualified_impressions": total_impressions,
            "avg_attention_time": sum(attention_times) / len(attention_times) if attention_times else 0.0,
            "avg_dwell_time": sum(dwell_times) / len(dwell_times) if dwell_times else 0.0
        }

    def remove_person(self, person_id: int) -> Optional[PersonAttention]:
        """
        Remove a person from tracking (e.g., when they leave frame).

        Returns the final attention data for the person.
        """
        return self._persons.pop(person_id, None)

    def cleanup_stale(self, max_age_seconds: float = 60.0) -> int:
        """
        Remove persons not seen recently.

        Args:
            max_age_seconds: Remove persons not seen for this long

        Returns:
            Number of persons removed
        """
        current_time = time.time()
        stale_ids = [
            pid for pid, p in self._persons.items()
            if current_time - p.last_seen > max_age_seconds
        ]

        for pid in stale_ids:
            self._persons.pop(pid)

        if stale_ids:
            logger.debug("attention_cleanup", removed_count=len(stale_ids))

        return len(stale_ids)
