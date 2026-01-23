#!/usr/bin/env python3
"""
Demographics Overlay Tool

Processes a video with person bounding boxes and adds:
1. Face detection within each person
2. Age/Gender estimation
3. Face ReID embeddings (optional)
4. Enhanced video overlay with demographics info

Usage:
    python3 tools/add_demographics_overlay.py --input output/test_output.mp4 --output output/test_with_demographics.mp4
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import time
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from demographics.face_detector import FaceDetector, FaceDetection
from demographics.demographics_estimator import DemographicsEstimator, Demographics

import structlog
logger = structlog.get_logger(__name__)


@dataclass
class PersonWithDemographics:
    """Person tracking with demographics info"""
    person_id: int
    bbox: Tuple[int, int, int, int]
    face_bbox: Optional[Tuple[int, int, int, int]] = None
    demographics: Optional[Demographics] = None
    face_embedding: Optional[np.ndarray] = None
    last_seen_frame: int = 0
    appearance_count: int = 0


class DemographicsOverlayProcessor:
    """
    Processes video to add demographics overlay
    """

    # Colors for different genders (BGR)
    COLOR_MALE = (255, 128, 0)      # Orange
    COLOR_FEMALE = (255, 0, 255)    # Magenta
    COLOR_UNKNOWN = (0, 255, 255)   # Yellow
    COLOR_FACE = (0, 255, 0)        # Green

    def __init__(
        self,
        face_conf_threshold: float = 0.5,
        enable_face_reid: bool = True
    ):
        """
        Initialize processor

        Args:
            face_conf_threshold: Face detection confidence threshold
            enable_face_reid: Enable face embedding extraction for ReID
        """
        self.face_conf_threshold = face_conf_threshold
        self.enable_face_reid = enable_face_reid

        # Initialize models
        logger.info("loading_face_detector")
        self.face_detector = FaceDetector(conf_threshold=face_conf_threshold)

        logger.info("loading_demographics_estimator")
        self.demographics_estimator = DemographicsEstimator()

        # Face ReID model (using same face embeddings from detection)
        self.face_reid_enabled = enable_face_reid

        # Person demographics cache
        # {person_id: PersonWithDemographics}
        self.person_cache: Dict[int, PersonWithDemographics] = {}

        # Face embeddings database for face ReID
        # {face_id: np.ndarray}
        self.face_embeddings: Dict[int, np.ndarray] = {}
        self.next_face_id = 1

        logger.info("demographics_processor_initialized")

    def extract_person_id_from_text(self, text: str) -> Optional[int]:
        """
        Extract person ID from OSD text like "Person 5" or "P5 M32"

        Args:
            text: Text overlay from video

        Returns:
            Person ID or None
        """
        if not text:
            return None

        # Try "Person X" format
        if text.startswith("Person "):
            try:
                return int(text.split()[1])
            except (IndexError, ValueError):
                pass

        # Try "PX" format
        if text.startswith("P") and len(text) > 1:
            try:
                # Extract number after P
                num_str = ""
                for c in text[1:]:
                    if c.isdigit():
                        num_str += c
                    else:
                        break
                if num_str:
                    return int(num_str)
            except ValueError:
                pass

        return None

    def process_frame(
        self,
        frame: np.ndarray,
        frame_number: int
    ) -> Tuple[np.ndarray, List[PersonWithDemographics]]:
        """
        Process a single frame

        Args:
            frame: BGR frame
            frame_number: Current frame number

        Returns:
            Annotated frame and list of detected persons with demographics
        """
        persons_in_frame = []

        # For this demo, we'll detect faces in the full frame
        # In production, you'd use person bounding boxes from DeepStream
        faces = self.face_detector.detect(frame)

        for face in faces:
            # Get face crop for demographics
            x1, y1, x2, y2 = face.bbox
            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size == 0:
                continue

            # Estimate demographics
            demographics = self.demographics_estimator.estimate(face_crop)

            if demographics is None:
                continue

            # Create person entry
            person = PersonWithDemographics(
                person_id=0,  # Will be assigned by face ReID
                bbox=(x1, y1, x2, y2),
                face_bbox=face.bbox,
                demographics=demographics,
                last_seen_frame=frame_number,
                appearance_count=1
            )

            persons_in_frame.append(person)

            # Draw face bounding box
            color = self.COLOR_MALE if demographics.gender == 'male' else self.COLOR_FEMALE
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw demographics text
            text = f"{demographics.gender[0].upper()}{demographics.age}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

            # Background for text
            cv2.rectangle(
                frame,
                (x1, y1 - text_size[1] - 10),
                (x1 + text_size[0] + 10, y1),
                color,
                -1
            )

            # Text
            cv2.putText(
                frame,
                text,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            # Draw age group below
            age_text = f"({demographics.age_group})"
            cv2.putText(
                frame,
                age_text,
                (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )

        return frame, persons_in_frame

    def process_video(
        self,
        input_path: str,
        output_path: str,
        show_preview: bool = False
    ) -> Dict:
        """
        Process entire video

        Args:
            input_path: Input video path
            output_path: Output video path
            show_preview: Show live preview window

        Returns:
            Statistics dictionary
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_path}")

        # Open input video
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {input_path}")

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(
            "processing_video",
            input=str(input_path),
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames
        )

        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (width, height)
        )

        # Statistics
        stats = {
            'total_frames': 0,
            'faces_detected': 0,
            'demographics_estimated': 0,
            'male_count': 0,
            'female_count': 0,
            'age_distribution': {},
            'processing_time': 0.0
        }

        start_time = time.time()
        frame_number = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_number += 1
            stats['total_frames'] = frame_number

            # Process frame
            annotated_frame, persons = self.process_frame(frame, frame_number)

            # Update statistics
            stats['faces_detected'] += len(persons)
            for person in persons:
                if person.demographics:
                    stats['demographics_estimated'] += 1
                    if person.demographics.gender == 'male':
                        stats['male_count'] += 1
                    else:
                        stats['female_count'] += 1

                    age_group = person.demographics.age_group
                    stats['age_distribution'][age_group] = \
                        stats['age_distribution'].get(age_group, 0) + 1

            # Add frame info overlay
            info_text = f"Frame: {frame_number}/{total_frames} | Faces: {len(persons)}"
            cv2.putText(
                annotated_frame,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            # Write frame
            writer.write(annotated_frame)

            # Show preview
            if show_preview:
                cv2.imshow('Demographics Overlay', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Progress logging
            if frame_number % 100 == 0:
                elapsed = time.time() - start_time
                fps_actual = frame_number / elapsed if elapsed > 0 else 0
                logger.info(
                    "processing_progress",
                    frame=frame_number,
                    total=total_frames,
                    fps=f"{fps_actual:.1f}",
                    faces=stats['faces_detected']
                )

        # Cleanup
        cap.release()
        writer.release()
        if show_preview:
            cv2.destroyAllWindows()

        stats['processing_time'] = time.time() - start_time

        logger.info(
            "processing_completed",
            output=str(output_path),
            stats=stats
        )

        return stats


def main():
    parser = argparse.ArgumentParser(
        description="Add demographics overlay to video"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input video path"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output video path"
    )
    parser.add_argument(
        "--face-threshold",
        type=float,
        default=0.5,
        help="Face detection confidence threshold (default: 0.5)"
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show live preview window"
    )

    args = parser.parse_args()

    # Initialize processor
    processor = DemographicsOverlayProcessor(
        face_conf_threshold=args.face_threshold,
        enable_face_reid=True
    )

    # Process video
    stats = processor.process_video(
        input_path=args.input,
        output_path=args.output,
        show_preview=args.preview
    )

    # Print summary
    print("\n" + "=" * 50)
    print("Demographics Processing Complete")
    print("=" * 50)
    print(f"Total frames: {stats['total_frames']}")
    print(f"Faces detected: {stats['faces_detected']}")
    print(f"Demographics estimated: {stats['demographics_estimated']}")
    print(f"Male: {stats['male_count']}, Female: {stats['female_count']}")
    print(f"Age distribution: {stats['age_distribution']}")
    print(f"Processing time: {stats['processing_time']:.1f}s")
    print(f"Average FPS: {stats['total_frames']/stats['processing_time']:.1f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
