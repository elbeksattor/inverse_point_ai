#!/usr/bin/env python3
"""
DeepStream Pipeline with Integrated Demographics
Professional production-ready implementation using DeepStream Secondary GIE

Pipeline flow:
filesrc → qtdemux → h264parse → nvv4l2decoder → nvstreammux
→ nvinfer (PeopleNet: Person + Face) → nvtracker (NvDCF+ReID)
→ nvinfer (Demographics SGIE) → nvvideoconvert → nvdsosd
→ nvvideoconvert → capsfilter → nvv4l2h264enc → filesink

Features:
- Real-time person detection and tracking
- Face detection (via PeopleNet class 2)
- Age/Gender estimation (via SGIE on face crops)
- Person ReID with global verification
- Demographics linked to persons
"""

import sys
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import pyds
import numpy as np
from pathlib import Path
import time
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
import structlog

# Import our modules
sys.path.append(str(Path(__file__).parent.parent))
from database.person_database import PersonDatabase
from tracking.global_reid_verifier import GlobalReIDVerifier

logger = structlog.get_logger(__name__)


@dataclass
class DemographicsResult:
    """Demographics estimation result"""
    age: int
    gender: str  # 'male' or 'female'
    gender_confidence: float
    age_group: str

    @staticmethod
    def get_age_group(age: int) -> str:
        """Convert age to marketing age group"""
        if age < 18:
            return "under_18"
        elif age < 25:
            return "18-24"
        elif age < 35:
            return "25-34"
        elif age < 45:
            return "35-44"
        elif age < 55:
            return "45-54"
        elif age < 65:
            return "55-64"
        else:
            return "65+"


@dataclass
class PersonWithDemographics:
    """Person tracking data with demographics"""
    person_id: int
    tracker_id: int
    bbox: Tuple[float, float, float, float]
    demographics: Optional[DemographicsResult] = None
    face_bbox: Optional[Tuple[float, float, float, float]] = None
    last_seen_frame: int = 0
    demographics_updated_frame: int = 0
    appearance_count: int = 0


class DeepStreamDemographicsPipeline:
    """
    Production-ready DeepStream pipeline with integrated demographics

    Uses DeepStream's native Secondary GIE for efficient GPU-accelerated
    demographics estimation on detected faces.
    """

    # Class IDs from PeopleNet
    CLASS_PERSON = 0
    CLASS_BAG = 1
    CLASS_FACE = 2

    # Demographics SGIE unique ID
    DEMOGRAPHICS_GIE_ID = 3

    # FPS optimization constants
    REID_VERIFY_INTERVAL = 5
    REID_UPDATE_INTERVAL = 10
    STABLE_TRACK_FRAMES = 5
    HIGH_CONFIDENCE_THRESHOLD = 0.70
    DEMOGRAPHICS_UPDATE_INTERVAL = 30  # Re-estimate every 30 frames

    # Display colors (BGR)
    COLOR_MALE = (255, 128, 0)      # Orange
    COLOR_FEMALE = (255, 0, 255)    # Magenta
    COLOR_PERSON = (0, 255, 0)      # Green
    COLOR_FACE = (0, 255, 255)      # Yellow

    def __init__(
        self,
        video_path: str,
        output_path: str,
        detector_config: str,
        tracker_config: str,
        demographics_config: str = None,
        db_path: str = "output/database/person_database.db",
        index_path: str = "output/database/person_embeddings.index",
        enable_demographics: bool = True
    ):
        """
        Initialize DeepStream pipeline with demographics

        Args:
            video_path: Input video file path
            output_path: Output video file path
            detector_config: Path to nvinfer config (PeopleNet with face)
            tracker_config: Path to nvtracker config (NvDCF)
            demographics_config: Path to demographics SGIE config
            db_path: Person database path
            index_path: FAISS index path
            enable_demographics: Enable demographics estimation
        """
        self.video_path = Path(video_path)
        self.output_path = Path(output_path)
        self.detector_config = Path(detector_config)
        self.tracker_config = Path(tracker_config)
        self.enable_demographics = enable_demographics

        # Default demographics config
        if demographics_config:
            self.demographics_config = Path(demographics_config)
        else:
            self.demographics_config = Path(__file__).parent.parent.parent / \
                "configs/demographics_sgie_config.txt"

        # Validate paths
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")
        if not self.detector_config.exists():
            raise FileNotFoundError(f"Detector config not found: {self.detector_config}")
        if not self.tracker_config.exists():
            raise FileNotFoundError(f"Tracker config not found: {self.tracker_config}")
        if enable_demographics and not self.demographics_config.exists():
            raise FileNotFoundError(f"Demographics config not found: {self.demographics_config}")

        # Initialize GStreamer
        Gst.init(None)

        # Create output directory
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize RE-ID database
        self.person_db = PersonDatabase(
            db_path=db_path,
            index_path=index_path,
            embedding_dim=256,
            use_gpu=False
        )

        # Pipeline elements
        self.pipeline = None
        self.loop = None

        # Statistics
        self.frame_count = 0
        self.person_count = 0
        self.face_count = 0
        self.fps = 0.0
        self.start_time = None

        # Tracker ID to Person ID mapping
        self.tracker_to_person_map: Dict[int, int] = {}

        # Person tracking with demographics
        # {person_id: PersonWithDemographics}
        self.person_data: Dict[int, PersonWithDemographics] = {}

        # Face to Person association (for linking demographics)
        # {(frame, face_tracker_id): person_id}
        self.face_to_person: Dict[Tuple[int, int], int] = {}

        # Current frame faces and persons for association
        self.current_frame_persons: List[Tuple[int, Tuple[float, float, float, float]]] = []
        self.current_frame_faces: List[Tuple[int, Tuple[float, float, float, float]]] = []

        # ID correction statistics
        self.id_corrections = 0

        # FPS optimization tracking
        self.tracker_stability: Dict[int, Dict] = {}
        self.confidence_cache: Dict[int, Dict] = {}
        self.frames_skipped = 0
        self.tracker_history: Dict[int, Dict] = {}

        # Global ReID Verifier
        self.global_verifier = GlobalReIDVerifier(
            confident_threshold=0.75,
            possible_threshold=0.65,
            new_person_threshold=0.60,
            temporal_window=5
        )

        # Demographics statistics
        self.demographics_stats = {
            'male_count': 0,
            'female_count': 0,
            'age_distribution': {},
            'total_detections': 0
        }

        logger.info(
            "demographics_pipeline_initialized",
            video_path=str(self.video_path),
            output_path=str(self.output_path),
            enable_demographics=enable_demographics
        )

    def build_pipeline(self) -> bool:
        """Build GStreamer pipeline with demographics SGIE"""
        logger.info("building_demographics_pipeline")

        # Create pipeline
        self.pipeline = Gst.Pipeline()
        if not self.pipeline:
            logger.error("failed_to_create_pipeline")
            return False

        # ==================== Source Elements ====================
        source = Gst.ElementFactory.make("filesrc", "file-source")
        if not source:
            logger.error("failed_to_create_filesrc")
            return False
        source.set_property("location", str(self.video_path))

        qtdemux = Gst.ElementFactory.make("qtdemux", "qtdemux")
        if not qtdemux:
            logger.error("failed_to_create_qtdemux")
            return False

        h264parser = Gst.ElementFactory.make("h264parse", "h264-parser")
        if not h264parser:
            logger.error("failed_to_create_h264parse")
            return False

        decoder = Gst.ElementFactory.make("nvv4l2decoder", "nvv4l2-decoder")
        if not decoder:
            logger.error("failed_to_create_nvv4l2decoder")
            return False

        streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
        if not streammux:
            logger.error("failed_to_create_nvstreammux")
            return False
        streammux.set_property("width", 1920)
        streammux.set_property("height", 1080)
        streammux.set_property("batch-size", 1)
        streammux.set_property("batched-push-timeout", 4000000)

        # ==================== Primary GIE (Person + Face Detection) ====================
        pgie = Gst.ElementFactory.make("nvinfer", "primary-infer")
        if not pgie:
            logger.error("failed_to_create_nvinfer_pgie")
            return False
        pgie.set_property("config-file-path", str(self.detector_config))

        # ==================== Tracker ====================
        tracker = Gst.ElementFactory.make("nvtracker", "tracker")
        if not tracker:
            logger.error("failed_to_create_nvtracker")
            return False
        tracker.set_property("ll-lib-file", "/opt/nvidia/deepstream/deepstream-7.1/lib/libnvds_nvmultiobjecttracker.so")
        tracker.set_property("ll-config-file", str(self.tracker_config))
        tracker.set_property("tracker-width", 640)
        tracker.set_property("tracker-height", 384)
        tracker.set_property("gpu-id", 0)

        # ==================== Secondary GIE (Demographics) ====================
        sgie = None
        if self.enable_demographics:
            sgie = Gst.ElementFactory.make("nvinfer", "demographics-infer")
            if not sgie:
                logger.warning("failed_to_create_demographics_sgie_continuing_without")
                self.enable_demographics = False
            else:
                sgie.set_property("config-file-path", str(self.demographics_config))
                logger.info("demographics_sgie_created")

        # ==================== Display Elements ====================
        nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
        if not nvvidconv:
            logger.error("failed_to_create_nvvideoconvert")
            return False

        nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
        if not nvosd:
            logger.error("failed_to_create_nvdsosd")
            return False
        nvosd.set_property("process-mode", 1)  # GPU mode
        nvosd.set_property("display-text", 1)

        nvvidconv2 = Gst.ElementFactory.make("nvvideoconvert", "convertor2")
        if not nvvidconv2:
            logger.error("failed_to_create_nvvideoconvert2")
            return False

        # ==================== Encoder Elements ====================
        capsfilter = Gst.ElementFactory.make("capsfilter", "capsfilter")
        if not capsfilter:
            logger.error("failed_to_create_capsfilter")
            return False
        caps = Gst.Caps.from_string("video/x-raw, format=I420")
        capsfilter.set_property("caps", caps)

        encoder = Gst.ElementFactory.make("nvv4l2h264enc", "h264-encoder")
        if encoder:
            encoder.set_property("bitrate", 4000000)
            encoder.set_property("preset-level", 1)
            encoder.set_property("insert-sps-pps", 1)
            encoder.set_property("bufapi-version", 1)
        else:
            encoder = Gst.ElementFactory.make("x264enc", "h264-encoder")
            if not encoder:
                logger.error("failed_to_create_encoder")
                return False
            encoder.set_property("bitrate", 4000)
            encoder.set_property("speed-preset", "ultrafast")

        h264parser2 = Gst.ElementFactory.make("h264parse", "h264-parser2")
        if not h264parser2:
            logger.error("failed_to_create_h264parse2")
            return False

        qtmux = Gst.ElementFactory.make("qtmux", "qtmux")
        if not qtmux:
            logger.error("failed_to_create_qtmux")
            return False

        sink = Gst.ElementFactory.make("filesink", "filesink")
        if not sink:
            logger.error("failed_to_create_filesink")
            return False
        sink.set_property("location", str(self.output_path))
        sink.set_property("sync", 0)
        sink.set_property("async", 0)

        # ==================== Add Elements to Pipeline ====================
        self.pipeline.add(source)
        self.pipeline.add(qtdemux)
        self.pipeline.add(h264parser)
        self.pipeline.add(decoder)
        self.pipeline.add(streammux)
        self.pipeline.add(pgie)
        self.pipeline.add(tracker)
        if sgie:
            self.pipeline.add(sgie)
        self.pipeline.add(nvvidconv)
        self.pipeline.add(nvosd)
        self.pipeline.add(nvvidconv2)
        self.pipeline.add(capsfilter)
        self.pipeline.add(encoder)
        self.pipeline.add(h264parser2)
        self.pipeline.add(qtmux)
        self.pipeline.add(sink)

        # ==================== Link Elements ====================
        if not source.link(qtdemux):
            logger.error("failed_to_link_source_qtdemux")
            return False

        qtdemux.connect("pad-added", self._on_pad_added, h264parser)

        if not h264parser.link(decoder):
            logger.error("failed_to_link_h264parser_decoder")
            return False

        # Connect decoder to streammux
        sinkpad = streammux.get_request_pad("sink_0")
        srcpad = decoder.get_static_pad("src")
        if srcpad.link(sinkpad) != Gst.PadLinkReturn.OK:
            logger.error("failed_to_link_decoder_streammux")
            return False

        if not streammux.link(pgie):
            logger.error("failed_to_link_streammux_pgie")
            return False

        if not pgie.link(tracker):
            logger.error("failed_to_link_pgie_tracker")
            return False

        if sgie:
            if not tracker.link(sgie):
                logger.error("failed_to_link_tracker_sgie")
                return False
            if not sgie.link(nvvidconv):
                logger.error("failed_to_link_sgie_nvvidconv")
                return False
        else:
            if not tracker.link(nvvidconv):
                logger.error("failed_to_link_tracker_nvvidconv")
                return False

        if not nvvidconv.link(nvosd):
            logger.error("failed_to_link_nvvidconv_nvosd")
            return False

        if not nvosd.link(nvvidconv2):
            logger.error("failed_to_link_nvosd_nvvidconv2")
            return False

        if not nvvidconv2.link(capsfilter):
            logger.error("failed_to_link_nvvidconv2_capsfilter")
            return False

        if not capsfilter.link(encoder):
            logger.error("failed_to_link_capsfilter_encoder")
            return False

        if not encoder.link(h264parser2):
            logger.error("failed_to_link_encoder_h264parser2")
            return False

        if not h264parser2.link(qtmux):
            logger.error("failed_to_link_h264parser2_qtmux")
            return False

        if not qtmux.link(sink):
            logger.error("failed_to_link_qtmux_sink")
            return False

        # ==================== Add Probes ====================
        # Probe after tracker (before SGIE) for person tracking
        tracker_src_pad = tracker.get_static_pad("src")
        if tracker_src_pad:
            tracker_src_pad.add_probe(
                Gst.PadProbeType.BUFFER,
                self._tracker_probe_callback,
                0
            )

        # Probe after SGIE for demographics extraction
        if sgie:
            sgie_src_pad = sgie.get_static_pad("src")
            if sgie_src_pad:
                sgie_src_pad.add_probe(
                    Gst.PadProbeType.BUFFER,
                    self._demographics_probe_callback,
                    0
                )

        # Bus message handler
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._bus_call)

        logger.info("pipeline_built_successfully", with_demographics=self.enable_demographics)
        return True

    def _on_pad_added(self, element, pad, data):
        """Callback for dynamic pad linking"""
        sink_pad = data.get_static_pad("sink")
        if not sink_pad.is_linked():
            pad.link(sink_pad)

    def _tracker_probe_callback(self, pad, info, u_data):
        """
        Probe callback after tracker - handles person tracking and ReID
        """
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if not batch_meta:
            return Gst.PadProbeReturn.OK

        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            self.frame_count += 1

            # Calculate FPS
            if self.start_time is None:
                self.start_time = time.time()
            else:
                elapsed = time.time() - self.start_time
                if elapsed > 0:
                    self.fps = self.frame_count / elapsed

            # Reset per-frame tracking
            self.current_frame_persons = []
            self.current_frame_faces = []

            # Process objects
            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break

                # Get bounding box
                bbox = (
                    obj_meta.rect_params.left,
                    obj_meta.rect_params.top,
                    obj_meta.rect_params.width,
                    obj_meta.rect_params.height
                )

                if obj_meta.class_id == self.CLASS_PERSON:
                    # Process person
                    self._process_person(obj_meta, bbox, frame_meta)
                    self.current_frame_persons.append((obj_meta.object_id, bbox))

                elif obj_meta.class_id == self.CLASS_FACE:
                    # Track face for demographics association
                    self.face_count += 1
                    self.current_frame_faces.append((obj_meta.object_id, bbox))

                    # Set face display color
                    obj_meta.rect_params.border_color.set(1.0, 1.0, 0.0, 1.0)  # Yellow
                    obj_meta.rect_params.border_width = 2

                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            # Associate faces to persons
            self._associate_faces_to_persons()

            # Add frame info overlay
            self._add_frame_overlay(batch_meta, frame_meta)

            # Logging
            if self.frame_count % 100 == 0:
                logger.info(
                    "frame_progress",
                    frame=self.frame_count,
                    fps=f"{self.fps:.1f}",
                    persons=self.person_count,
                    faces=self.face_count
                )

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

    def _process_person(self, obj_meta, bbox, frame_meta):
        """Process person detection with ReID"""
        tracker_id = obj_meta.object_id
        confidence = obj_meta.confidence

        # Extract ReID embedding
        embedding = self._extract_reid_embedding(obj_meta)

        if embedding is not None:
            # Check for cached high-confidence match
            skip_verification = False
            cached_person_id = None

            if tracker_id in self.tracker_stability:
                stability = self.tracker_stability[tracker_id]
                frames_tracked = self.frame_count - stability['first_seen']

                if tracker_id in self.confidence_cache:
                    cache = self.confidence_cache[tracker_id]
                    frames_since_verify = self.frame_count - cache['frame']

                    if (frames_tracked >= self.STABLE_TRACK_FRAMES and
                        cache['similarity'] >= self.HIGH_CONFIDENCE_THRESHOLD and
                        frames_since_verify < self.REID_VERIFY_INTERVAL):
                        skip_verification = True
                        cached_person_id = cache['person_id']
                        self.frames_skipped += 1

            if skip_verification and cached_person_id is not None:
                person_id = cached_person_id
                is_new = False
                similarity = self.confidence_cache[tracker_id]['similarity']
                self.tracker_to_person_map[tracker_id] = person_id

                if self.frame_count % self.REID_UPDATE_INTERVAL == 0:
                    record = self.global_verifier.get_person_record(person_id)
                    if record:
                        record.update_embedding(embedding)
            else:
                # Full verification
                person_id, is_new, similarity = self.person_db.match_or_create_person(
                    embedding=embedding,
                    threshold=0.65,
                    confidence=confidence,
                    bbox=bbox,
                    frame_number=self.frame_count,
                    camera_id=1
                )

                # Global verification
                verified_id, was_corrected, verified_sim, reason = \
                    self.global_verifier.verify_assignment(
                        tracker_id=tracker_id,
                        proposed_person_id=person_id,
                        embedding=embedding,
                        frame=self.frame_count,
                        bbox=bbox,
                        is_new_person=is_new
                    )

                if was_corrected:
                    self.id_corrections += 1
                    if is_new:
                        self.person_db._rollback_new_person(person_id)
                        is_new = False
                    person_id = verified_id
                    similarity = verified_sim

                # Register/update in global verifier
                if is_new:
                    self.global_verifier.register_person(
                        person_id=person_id,
                        frame=self.frame_count,
                        embedding=embedding,
                        bbox=bbox,
                        confidence=confidence
                    )
                else:
                    record = self.global_verifier.get_person_record(person_id)
                    if record:
                        record.update(
                            frame=self.frame_count,
                            embedding=embedding,
                            bbox=bbox,
                            confidence=confidence,
                            similarity=similarity
                        )
                    else:
                        self.global_verifier.register_person(
                            person_id=person_id,
                            frame=self.frame_count,
                            embedding=embedding,
                            bbox=bbox,
                            confidence=confidence
                        )

                self.tracker_to_person_map[tracker_id] = person_id
                self.tracker_history[tracker_id] = {
                    'person_id': person_id,
                    'embedding': embedding.copy(),
                    'last_frame': self.frame_count
                }

                # Update stability tracking
                if tracker_id not in self.tracker_stability:
                    self.tracker_stability[tracker_id] = {
                        'first_seen': self.frame_count,
                        'last_verified': self.frame_count,
                        'stable': False
                    }
                else:
                    self.tracker_stability[tracker_id]['last_verified'] = self.frame_count
                    frames_tracked = self.frame_count - self.tracker_stability[tracker_id]['first_seen']
                    if frames_tracked >= self.STABLE_TRACK_FRAMES:
                        self.tracker_stability[tracker_id]['stable'] = True

                # Update confidence cache
                self.confidence_cache[tracker_id] = {
                    'person_id': person_id,
                    'similarity': similarity,
                    'frame': self.frame_count
                }

                if is_new:
                    self.person_count += 1
                    # Initialize person data
                    self.person_data[person_id] = PersonWithDemographics(
                        person_id=person_id,
                        tracker_id=tracker_id,
                        bbox=bbox,
                        last_seen_frame=self.frame_count
                    )
                    logger.info(
                        "new_person_detected",
                        person_id=person_id,
                        tracker_id=tracker_id,
                        frame=self.frame_count
                    )

            # Update person data
            if person_id in self.person_data:
                self.person_data[person_id].bbox = bbox
                self.person_data[person_id].last_seen_frame = self.frame_count
                self.person_data[person_id].appearance_count += 1

            # Update display text
            self._update_person_display(obj_meta, person_id)
        else:
            # No embedding - use cached mapping
            if tracker_id in self.tracker_to_person_map:
                person_id = self.tracker_to_person_map[tracker_id]
                self._update_person_display(obj_meta, person_id)
            else:
                obj_meta.text_params.display_text = f"Track {tracker_id}"

    def _update_person_display(self, obj_meta, person_id: int):
        """Update person display text with demographics if available"""
        if person_id in self.person_data:
            person = self.person_data[person_id]
            if person.demographics:
                demo = person.demographics
                display_text = f"P{person_id} {demo.gender[0].upper()}{demo.age}"
                # Set color based on gender
                if demo.gender == 'male':
                    obj_meta.rect_params.border_color.set(1.0, 0.5, 0.0, 1.0)  # Orange
                else:
                    obj_meta.rect_params.border_color.set(1.0, 0.0, 1.0, 1.0)  # Magenta
            else:
                display_text = f"Person {person_id}"
                obj_meta.rect_params.border_color.set(0.0, 1.0, 0.0, 1.0)  # Green
        else:
            display_text = f"Person {person_id}"
            obj_meta.rect_params.border_color.set(0.0, 1.0, 0.0, 1.0)  # Green

        obj_meta.text_params.display_text = display_text
        obj_meta.rect_params.border_width = 3

    def _associate_faces_to_persons(self):
        """Associate detected faces to their parent persons based on bbox overlap"""
        for face_id, face_bbox in self.current_frame_faces:
            face_center_x = face_bbox[0] + face_bbox[2] / 2
            face_center_y = face_bbox[1] + face_bbox[3] / 2

            best_person_id = None
            best_overlap = 0

            for person_tracker_id, person_bbox in self.current_frame_persons:
                # Check if face center is inside person bbox
                if (person_bbox[0] <= face_center_x <= person_bbox[0] + person_bbox[2] and
                    person_bbox[1] <= face_center_y <= person_bbox[1] + person_bbox[3]):

                    # Calculate overlap ratio
                    overlap = self._calculate_overlap(face_bbox, person_bbox)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        if person_tracker_id in self.tracker_to_person_map:
                            best_person_id = self.tracker_to_person_map[person_tracker_id]

            if best_person_id is not None:
                self.face_to_person[(self.frame_count, face_id)] = best_person_id

    def _calculate_overlap(self, bbox1, bbox2) -> float:
        """Calculate IoU overlap between two bboxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
        y2 = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = bbox1[2] * bbox1[3]
        area2 = bbox2[2] * bbox2[3]
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _demographics_probe_callback(self, pad, info, u_data):
        """
        Probe callback after demographics SGIE - extracts age/gender results
        """
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if not batch_meta:
            return Gst.PadProbeReturn.OK

        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break

                # Process face objects with demographics results
                if obj_meta.class_id == self.CLASS_FACE:
                    self._extract_demographics(obj_meta)

                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

    def _extract_demographics(self, face_obj_meta):
        """Extract demographics from SGIE tensor output"""
        # Get classifier metadata from face object
        l_class = face_obj_meta.classifier_meta_list
        while l_class is not None:
            try:
                class_meta = pyds.NvDsClassifierMeta.cast(l_class.data)
            except StopIteration:
                break

            # Check if this is from demographics SGIE
            if class_meta.unique_component_id == self.DEMOGRAPHICS_GIE_ID:
                # Extract tensor output
                l_label = class_meta.label_info_list
                if l_label:
                    try:
                        label_info = pyds.NvDsLabelInfo.cast(l_label.data)

                        # Parse demographics from tensor
                        demographics = self._parse_demographics_tensor(face_obj_meta)

                        if demographics:
                            # Find associated person
                            face_id = face_obj_meta.object_id
                            person_id = self.face_to_person.get((self.frame_count, face_id))

                            if person_id and person_id in self.person_data:
                                self.person_data[person_id].demographics = demographics
                                self.person_data[person_id].demographics_updated_frame = self.frame_count

                                # Update display
                                face_obj_meta.text_params.display_text = \
                                    f"{demographics.gender[0].upper()}{demographics.age}"

                                # Update statistics
                                self.demographics_stats['total_detections'] += 1
                                if demographics.gender == 'male':
                                    self.demographics_stats['male_count'] += 1
                                else:
                                    self.demographics_stats['female_count'] += 1

                                age_group = demographics.age_group
                                self.demographics_stats['age_distribution'][age_group] = \
                                    self.demographics_stats['age_distribution'].get(age_group, 0) + 1

                                logger.debug(
                                    "demographics_extracted",
                                    person_id=person_id,
                                    age=demographics.age,
                                    gender=demographics.gender,
                                    age_group=demographics.age_group
                                )
                    except StopIteration:
                        pass

            try:
                l_class = l_class.next
            except StopIteration:
                break

    def _parse_demographics_tensor(self, face_obj_meta) -> Optional[DemographicsResult]:
        """
        Parse demographics from SGIE output tensor

        GenderAge model output: [gender_female, gender_male, age/100]
        """
        # Try to get tensor output metadata
        l_user = face_obj_meta.obj_user_meta_list
        while l_user is not None:
            try:
                user_meta = pyds.NvDsUserMeta.cast(l_user.data)
            except StopIteration:
                break

            if user_meta.base_meta.meta_type == pyds.NVDS_TENSOR_OUTPUT_META:
                try:
                    tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)

                    # Get output tensor
                    for i in range(tensor_meta.num_output_layers):
                        layer = pyds.get_nvds_LayerInfo(tensor_meta, i)

                        if layer.dataType == 0:  # FP32
                            # Get tensor data
                            ptr = pyds.get_ptr(layer.buffer)
                            data = np.ctypeslib.as_array(ptr, shape=(3,))

                            # Parse gender (softmax of first 2 values)
                            gender_logits = data[0:2]
                            gender_probs = np.exp(gender_logits) / np.sum(np.exp(gender_logits))
                            gender_idx = np.argmax(gender_probs)
                            gender = 'female' if gender_idx == 0 else 'male'
                            gender_confidence = float(gender_probs[gender_idx])

                            # Parse age
                            age = int(data[2] * 100)
                            age = max(1, min(100, age))

                            age_group = DemographicsResult.get_age_group(age)

                            return DemographicsResult(
                                age=age,
                                gender=gender,
                                gender_confidence=gender_confidence,
                                age_group=age_group
                            )

                except Exception as e:
                    logger.warning("failed_to_parse_demographics_tensor", error=str(e))

            try:
                l_user = l_user.next
            except StopIteration:
                break

        return None

    def _extract_reid_embedding(self, obj_meta) -> Optional[np.ndarray]:
        """Extract ReID embedding from tracker metadata"""
        l_user = obj_meta.obj_user_meta_list
        while l_user is not None:
            try:
                user_meta = pyds.NvDsUserMeta.cast(l_user.data)
            except StopIteration:
                break

            if user_meta.base_meta.meta_type == pyds.NVDS_TRACKER_OBJ_REID_META:
                try:
                    reid_meta = pyds.NvDsObjReid.cast(user_meta.user_meta_data)
                    if reid_meta.featureSize > 0:
                        embedding = reid_meta.get_host_reid_vector()
                        if embedding is not None and len(embedding) > 0:
                            return embedding
                except Exception:
                    pass

            try:
                l_user = l_user.next
            except StopIteration:
                break

        return None

    def _add_frame_overlay(self, batch_meta, frame_meta):
        """Add frame information overlay"""
        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 1

        py_nvosd_text_params = display_meta.text_params[0]
        py_nvosd_text_params.display_text = \
            f"Frame: {self.frame_count} | FPS: {self.fps:.1f} | Persons: {self.person_count} | Faces: {self.face_count}"
        py_nvosd_text_params.x_offset = 10
        py_nvosd_text_params.y_offset = 12
        py_nvosd_text_params.font_params.font_name = "Serif"
        py_nvosd_text_params.font_params.font_size = 14
        py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
        py_nvosd_text_params.set_bg_clr = 1
        py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)

        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

    def _bus_call(self, bus, message):
        """Handle GStreamer bus messages"""
        t = message.type
        if t == Gst.MessageType.EOS:
            logger.info("end_of_stream")
            self.loop.quit()
        elif t == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            logger.warning("gstreamer_warning", error=err.message)
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            logger.error("gstreamer_error", error=err.message, debug=debug)
            self.loop.quit()
        return True

    def run(self) -> bool:
        """Run the pipeline"""
        if not self.build_pipeline():
            logger.error("pipeline_build_failed")
            return False

        self.loop = GLib.MainLoop()

        logger.info("starting_demographics_pipeline")
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            logger.error("failed_to_set_pipeline_to_playing")
            return False

        try:
            self.loop.run()
        except KeyboardInterrupt:
            logger.info("keyboard_interrupt")
        except Exception as e:
            logger.error("pipeline_error", error=str(e))

        logger.info("stopping_pipeline")
        self.pipeline.set_state(Gst.State.NULL)

        # Save database
        self.person_db.close()

        # Print statistics
        verifier_stats = self.global_verifier.get_stats()
        logger.info(
            "pipeline_completed",
            total_frames=self.frame_count,
            unique_persons=self.person_count,
            total_faces=self.face_count,
            avg_fps=self.fps,
            output_path=str(self.output_path),
            id_corrections=self.id_corrections,
            frames_skipped=self.frames_skipped,
            demographics_stats=self.demographics_stats
        )

        return True


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="DeepStream Demographics Pipeline")
    parser.add_argument("--video", required=True, help="Input video file")
    parser.add_argument("--output", required=True, help="Output video file")
    parser.add_argument(
        "--detector-config",
        default="configs/peoplenet_with_face_config.txt",
        help="Detector config file (PeopleNet with face detection)"
    )
    parser.add_argument(
        "--tracker-config",
        default="configs/nvdcf_tracker_config.yml",
        help="Tracker config file"
    )
    parser.add_argument(
        "--demographics-config",
        default="configs/demographics_sgie_config.txt",
        help="Demographics SGIE config file"
    )
    parser.add_argument(
        "--db-path",
        default="output/database/person_database.db",
        help="Person database path"
    )
    parser.add_argument(
        "--no-demographics",
        action="store_true",
        help="Disable demographics estimation"
    )

    args = parser.parse_args()

    # Setup logger
    from utils.logger import setup_logger
    setup_logger(log_level="INFO")

    # Create and run pipeline
    pipeline = DeepStreamDemographicsPipeline(
        video_path=args.video,
        output_path=args.output,
        detector_config=args.detector_config,
        tracker_config=args.tracker_config,
        demographics_config=args.demographics_config,
        db_path=args.db_path,
        enable_demographics=not args.no_demographics
    )

    success = pipeline.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
