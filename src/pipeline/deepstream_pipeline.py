#!/usr/bin/env python3
"""
DeepStream Pipeline for Person Detection, Tracking, and RE-ID
Uses GStreamer Python bindings (GObject introspection)
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
import structlog

# Import our RE-ID database and Global Verifier
sys.path.append(str(Path(__file__).parent.parent))
from database.person_database import PersonDatabase
from tracking.global_reid_verifier import GlobalReIDVerifier

# Import demographics module (optional - graceful fallback if not available)
try:
    from demographics.face_detector import FaceDetector
    from demographics.demographics_estimator import DemographicsEstimator, HeadPoseEstimator
    DEMOGRAPHICS_AVAILABLE = True
except ImportError:
    DEMOGRAPHICS_AVAILABLE = False

logger = structlog.get_logger(__name__)


class DeepStreamPipeline:
    """
    DeepStream pipeline for person analytics with RE-ID

    Pipeline flow:
    filesrc → qtdemux → h264parse → nvv4l2decoder → nvstreammux
    → nvinfer (YOLOv8) → nvtracker (NvDCF+ReID) → nvvideoconvert
    → nvdsosd → nvvideoconvert → capsfilter → nvv4l2h264enc → filesink

    Optimizations (v2.1):
    - ReID verification every N frames (not every frame)
    - Cached person ID lookups for stable tracks
    - Reduced logging frequency
    - GPU-accelerated FAISS when available
    """

    # ============================================================
    # FPS OPTIMIZATION CONSTANTS
    # ============================================================
    # Process ReID verification every N frames for established tracks
    REID_VERIFY_INTERVAL = 5  # Verify every 5 frames (was every frame)

    # New person detection is always done immediately
    # But re-verification of known persons can be less frequent
    REID_UPDATE_INTERVAL = 10  # Update embeddings every 10 frames

    # Minimum frames before a track is considered "stable"
    STABLE_TRACK_FRAMES = 5  # Reduced from 10 to allow faster stabilization

    # Skip verification for high-confidence matches
    # Lowered threshold since our matching is 0.75-0.99 range
    HIGH_CONFIDENCE_THRESHOLD = 0.70

    def __init__(
        self,
        video_path: str,
        output_path: str,
        detector_config: str,
        tracker_config: str,
        db_path: str = "output/database/person_database.db",
        index_path: str = "output/database/person_embeddings.index",
        optimize_fps: bool = True  # Enable FPS optimizations
    ):
        """
        Initialize DeepStream pipeline

        Args:
            video_path: Input video file path
            output_path: Output video file path
            detector_config: Path to nvinfer config (YOLOv8)
            tracker_config: Path to nvtracker config (NvDCF)
            db_path: Person database path
            index_path: FAISS index path
            optimize_fps: Enable FPS optimizations (default: True)
        """
        self.video_path = Path(video_path)
        self.output_path = Path(output_path)
        self.detector_config = Path(detector_config)
        self.tracker_config = Path(tracker_config)
        self.optimize_fps = optimize_fps

        # Validate paths
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")
        if not self.detector_config.exists():
            raise FileNotFoundError(f"Detector config not found: {self.detector_config}")
        if not self.tracker_config.exists():
            raise FileNotFoundError(f"Tracker config not found: {self.tracker_config}")

        # Initialize GStreamer
        Gst.init(None)

        # Create output directory
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize RE-ID database
        # Note: ResNet50 Market-1501 model outputs 256-dim embeddings
        # CPU FAISS is fast enough for our use case (<100 persons)
        self.person_db = PersonDatabase(
            db_path=db_path,
            index_path=index_path,
            embedding_dim=256,  # Match ReID model output
            use_gpu=False  # CPU is sufficient for small person counts
        )

        # Pipeline elements
        self.pipeline = None
        self.loop = None

        # Statistics
        self.frame_count = 0
        self.person_count = 0
        self.fps = 0.0
        self.start_time = None

        # Tracker ID to Person ID mapping
        self.tracker_to_person_map: Dict[int, int] = {}

        # ID Switch Detection: Store recent embeddings per tracker_id
        # Format: {tracker_id: {'person_id': int, 'embedding': np.ndarray, 'last_frame': int}}
        self.tracker_history: Dict[int, Dict] = {}

        # ID correction statistics
        self.id_corrections = 0

        # ============================================================
        # FPS OPTIMIZATION: Track stability per tracker
        # ============================================================
        # {tracker_id: {'first_seen': frame, 'last_verified': frame, 'stable': bool}}
        self.tracker_stability: Dict[int, Dict] = {}

        # Cache for recent high-confidence matches (avoid re-verification)
        # {tracker_id: {'person_id': int, 'similarity': float, 'frame': int}}
        self.confidence_cache: Dict[int, Dict] = {}

        # Frames skipped due to optimization
        self.frames_skipped = 0

        # Global ReID Verifier - provides robust person identification
        # independent of NvDCF tracker ID assignments
        self.global_verifier = GlobalReIDVerifier(
            confident_threshold=0.75,  # High confidence match
            possible_threshold=0.65,   # Possible match
            new_person_threshold=0.60, # Below this, might be new person
            temporal_window=5          # Frames for smoothing
        )

        # ============================================================
        # DEMOGRAPHICS MODULE (Optional - Disabled in v2.1)
        # Frame extraction via pyds.get_nvds_buf_surface causes CUDA errors
        # Demographics will be implemented via secondary inference in v3.0
        # ============================================================
        self.face_detector = None
        self.demographics_estimator = None
        self.head_pose_estimator = None
        self.enable_demographics = False

        # Demographics loading disabled to avoid CUDA conflicts
        # TODO: Implement as secondary DeepStream inference
        # if DEMOGRAPHICS_AVAILABLE:
        #     try:
        #         self.face_detector = FaceDetector(conf_threshold=0.5)
        #         self.demographics_estimator = DemographicsEstimator()
        #         self.head_pose_estimator = HeadPoseEstimator()
        #         self.enable_demographics = True
        #         logger.info("demographics_module_loaded")
        #     except Exception as e:
        #         logger.warning("demographics_module_failed", error=str(e))
        #         self.enable_demographics = False

        # Demographics cache per person (avoid re-estimating every frame)
        # {person_id: {'age': int, 'gender': str, 'age_group': str, 'last_update': frame}}
        self.person_demographics: Dict[int, Dict] = {}
        self.DEMOGRAPHICS_UPDATE_INTERVAL = 30  # Re-estimate every 30 frames

        # Attention tracking per person
        # {person_id: {'total_attention_time': float, 'attention_frames': int}}
        self.person_attention: Dict[int, Dict] = {}

        logger.info(
            "deepstream_pipeline_initialized",
            video_path=str(self.video_path),
            output_path=str(self.output_path),
            optimize_fps=optimize_fps,
            demographics_enabled=self.enable_demographics
        )

    def build_pipeline(self):
        """Build GStreamer pipeline"""
        logger.info("building_gstreamer_pipeline")

        # Create pipeline
        self.pipeline = Gst.Pipeline()

        if not self.pipeline:
            logger.error("failed_to_create_pipeline")
            return False

        # Create elements
        logger.info("creating_pipeline_elements")

        # Source: file input
        source = Gst.ElementFactory.make("filesrc", "file-source")
        if not source:
            logger.error("failed_to_create_filesrc")
            return False
        source.set_property("location", str(self.video_path))

        # Demuxer for MP4
        qtdemux = Gst.ElementFactory.make("qtdemux", "qtdemux")
        if not qtdemux:
            logger.error("failed_to_create_qtdemux")
            return False

        # H264 parser
        h264parser = Gst.ElementFactory.make("h264parse", "h264-parser")
        if not h264parser:
            logger.error("failed_to_create_h264parse")
            return False

        # Hardware decoder (NVDEC)
        decoder = Gst.ElementFactory.make("nvv4l2decoder", "nvv4l2-decoder")
        if not decoder:
            logger.error("failed_to_create_nvv4l2decoder")
            return False

        # Stream muxer (batching)
        streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
        if not streammux:
            logger.error("failed_to_create_nvstreammux")
            return False
        streammux.set_property("width", 1920)
        streammux.set_property("height", 1080)
        streammux.set_property("batch-size", 1)
        streammux.set_property("batched-push-timeout", 4000000)

        # Primary inference (YOLOv8 person detection)
        pgie = Gst.ElementFactory.make("nvinfer", "primary-infer")
        if not pgie:
            logger.error("failed_to_create_nvinfer")
            return False
        pgie.set_property("config-file-path", str(self.detector_config))

        # Tracker (NvDCF with ReID)
        tracker = Gst.ElementFactory.make("nvtracker", "tracker")
        if not tracker:
            logger.error("failed_to_create_nvtracker")
            return False
        tracker.set_property("ll-lib-file", "/opt/nvidia/deepstream/deepstream-7.1/lib/libnvds_nvmultiobjecttracker.so")
        tracker.set_property("ll-config-file", str(self.tracker_config))
        tracker.set_property("tracker-width", 640)
        tracker.set_property("tracker-height", 384)
        tracker.set_property("gpu-id", 0)
        # Note: enable-batch-process and enable-past-frame are set via ll-config-file (YAML)

        # Video converter (for OSD)
        nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
        if not nvvidconv:
            logger.error("failed_to_create_nvvideoconvert")
            return False

        # On-Screen Display (draw bounding boxes and text)
        nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
        if not nvosd:
            logger.error("failed_to_create_nvdsosd")
            return False
        # FPS OPTIMIZATION: Use GPU mode for OSD (faster)
        nvosd.set_property("process-mode", 1)  # 0=CPU, 1=GPU (faster)
        nvosd.set_property("display-text", 1)

        # Video converter (for encoder)
        nvvidconv2 = Gst.ElementFactory.make("nvvideoconvert", "convertor2")
        if not nvvidconv2:
            logger.error("failed_to_create_nvvideoconvert2")
            return False

        # Caps filter for raw video (CPU memory for software encoder)
        capsfilter = Gst.ElementFactory.make("capsfilter", "capsfilter")
        if not capsfilter:
            logger.error("failed_to_create_capsfilter")
            return False
        caps = Gst.Caps.from_string("video/x-raw, format=I420")
        capsfilter.set_property("caps", caps)

        # Try hardware encoder first, fallback to software
        encoder = Gst.ElementFactory.make("nvv4l2h264enc", "h264-encoder")
        if encoder:
            logger.info("using_hardware_encoder_nvv4l2h264enc")
            encoder.set_property("bitrate", 4000000)
            encoder.set_property("preset-level", 1)
            encoder.set_property("insert-sps-pps", 1)
            encoder.set_property("bufapi-version", 1)
        else:
            # Fallback to software encoder (x264enc)
            logger.info("hardware_encoder_not_available_using_x264enc")
            encoder = Gst.ElementFactory.make("x264enc", "h264-encoder")
            if not encoder:
                logger.error("failed_to_create_any_h264_encoder")
                return False
            encoder.set_property("bitrate", 4000)  # kbps for x264enc
            encoder.set_property("speed-preset", "ultrafast")
            encoder.set_property("tune", "zerolatency")

        # H264 parser (for muxing)
        h264parser2 = Gst.ElementFactory.make("h264parse", "h264-parser2")
        if not h264parser2:
            logger.error("failed_to_create_h264parse2")
            return False

        # MP4 muxer
        qtmux = Gst.ElementFactory.make("qtmux", "qtmux")
        if not qtmux:
            logger.error("failed_to_create_qtmux")
            return False

        # File sink
        sink = Gst.ElementFactory.make("filesink", "filesink")
        if not sink:
            logger.error("failed_to_create_filesink")
            return False
        sink.set_property("location", str(self.output_path))
        sink.set_property("sync", 0)
        sink.set_property("async", 0)

        # Add elements to pipeline
        logger.info("adding_elements_to_pipeline")
        self.pipeline.add(source)
        self.pipeline.add(qtdemux)
        self.pipeline.add(h264parser)
        self.pipeline.add(decoder)
        self.pipeline.add(streammux)
        self.pipeline.add(pgie)
        self.pipeline.add(tracker)
        self.pipeline.add(nvvidconv)
        self.pipeline.add(nvosd)
        self.pipeline.add(nvvidconv2)
        self.pipeline.add(capsfilter)
        self.pipeline.add(encoder)
        self.pipeline.add(h264parser2)
        self.pipeline.add(qtmux)
        self.pipeline.add(sink)

        # Link elements
        logger.info("linking_pipeline_elements")

        if not source.link(qtdemux):
            logger.error("failed_to_link_source_qtdemux")
            return False

        # Connect qtdemux to h264parser dynamically (qtdemux creates pads on-demand)
        qtdemux.connect("pad-added", self.on_pad_added, h264parser)

        if not h264parser.link(decoder):
            logger.error("failed_to_link_h264parser_decoder")
            return False

        # Connect decoder to streammux sink pad
        sinkpad = streammux.get_request_pad("sink_0")
        if not sinkpad:
            logger.error("failed_to_get_streammux_sink_pad")
            return False

        srcpad = decoder.get_static_pad("src")
        if not srcpad:
            logger.error("failed_to_get_decoder_src_pad")
            return False

        if srcpad.link(sinkpad) != Gst.PadLinkReturn.OK:
            logger.error("failed_to_link_decoder_streammux")
            return False

        if not streammux.link(pgie):
            logger.error("failed_to_link_streammux_pgie")
            return False

        if not pgie.link(tracker):
            logger.error("failed_to_link_pgie_tracker")
            return False

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

        # Add probe to tracker src pad for metadata extraction
        tracker_src_pad = tracker.get_static_pad("src")
        if not tracker_src_pad:
            logger.error("failed_to_get_tracker_src_pad")
            return False

        tracker_src_pad.add_probe(
            Gst.PadProbeType.BUFFER,
            self.tracker_src_pad_buffer_probe,
            0
        )

        # Add bus message handler
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.bus_call)

        logger.info("pipeline_built_successfully")
        return True

    def on_pad_added(self, element, pad, data):
        """Callback for dynamically linking qtdemux to h264parser"""
        sink_pad = data.get_static_pad("sink")
        if not sink_pad.is_linked():
            pad.link(sink_pad)
            logger.debug("qtdemux_pad_linked")

    def tracker_src_pad_buffer_probe(self, pad, info, u_data):
        """
        Buffer probe callback to extract tracker metadata and ReID embeddings
        This is where we integrate with PersonDatabase
        """
        # Get buffer
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK

        # Get batch metadata
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if not batch_meta:
            return Gst.PadProbeReturn.OK

        # Iterate through frames in batch
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

            # Count objects in this frame for debugging
            num_objects = frame_meta.num_obj_meta
            # FPS OPTIMIZATION: Reduce logging frequency
            if self.frame_count % 100 == 0:  # Log every 100 frames (was 50)
                logger.info("frame_objects", frame=self.frame_count, num_objects=num_objects, fps=f"{self.fps:.1f}")

            # ============================================================
            # DEMOGRAPHICS: Frame extraction disabled in v2.1
            # pyds.get_nvds_buf_surface causes CUDA errors in DS 7.1
            # TODO: Use secondary inference or appsink branch for demographics
            # ============================================================
            frame_image = None
            # Demographics module is loaded but frame extraction disabled
            # Demographics will be processed via external means (e.g., appsink)

            # Iterate through detected objects
            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break

                # Debug: log all detected objects
                if self.frame_count <= 10:  # First 10 frames only
                    logger.info(
                        "object_detected",
                        frame=self.frame_count,
                        class_id=obj_meta.class_id,
                        confidence=obj_meta.confidence,
                        tracker_id=obj_meta.object_id
                    )

                # Only process person class (class_id 0 for PeopleNet)
                if obj_meta.class_id == 0:
                    tracker_id = obj_meta.object_id
                    confidence = obj_meta.confidence

                    # Get bounding box
                    bbox = (
                        obj_meta.rect_params.left,
                        obj_meta.rect_params.top,
                        obj_meta.rect_params.width,
                        obj_meta.rect_params.height
                    )

                    # Try to extract ReID embedding from tracker metadata
                    embedding = self.extract_reid_embedding(obj_meta)

                    if embedding is not None:
                        # ============================================================
                        # FPS-OPTIMIZED Global ReID Verification System (v2.1)
                        # Key optimizations:
                        # 1. Skip verification for stable, high-confidence tracks
                        # 2. Reduce verification frequency for known persons
                        # 3. Cache recent results to avoid redundant computation
                        # ============================================================

                        # Check if we should skip full verification (FPS optimization)
                        skip_verification = False
                        cached_person_id = None

                        if self.optimize_fps and tracker_id in self.tracker_stability:
                            stability = self.tracker_stability[tracker_id]
                            frames_tracked = self.frame_count - stability['first_seen']

                            # Check confidence cache
                            if tracker_id in self.confidence_cache:
                                cache = self.confidence_cache[tracker_id]
                                frames_since_verify = self.frame_count - cache['frame']

                                # Skip if: stable track + high confidence + recent verification
                                if (frames_tracked >= self.STABLE_TRACK_FRAMES and
                                    cache['similarity'] >= self.HIGH_CONFIDENCE_THRESHOLD and
                                    frames_since_verify < self.REID_VERIFY_INTERVAL):
                                    skip_verification = True
                                    cached_person_id = cache['person_id']
                                    self.frames_skipped += 1

                        if skip_verification and cached_person_id is not None:
                            # Use cached result - no database/verifier calls needed
                            person_id = cached_person_id
                            is_new = False
                            similarity = self.confidence_cache[tracker_id]['similarity']

                            # Update tracker mapping
                            self.tracker_to_person_map[tracker_id] = person_id

                            # Occasionally update embedding (less frequently)
                            if self.frame_count % self.REID_UPDATE_INTERVAL == 0:
                                record = self.global_verifier.get_person_record(person_id)
                                if record:
                                    record.update_embedding(embedding)
                        else:
                            # Full verification path (new tracks or periodic re-verification)

                            # Step 1: Get initial person assignment from database
                            person_id, is_new, similarity = self.person_db.match_or_create_person(
                                embedding=embedding,
                                threshold=0.65,
                                confidence=confidence,
                                bbox=bbox,
                                frame_number=self.frame_count,
                                camera_id=1
                            )

                            # Step 2: Verify assignment using Global ReID Verifier
                            verified_id, was_corrected, verified_sim, reason = \
                                self.global_verifier.verify_assignment(
                                    tracker_id=tracker_id,
                                    proposed_person_id=person_id,
                                    embedding=embedding,
                                    frame=self.frame_count,
                                    bbox=bbox,
                                    is_new_person=is_new
                                )

                            # Step 3: Handle corrections
                            if was_corrected:
                                self.id_corrections += 1

                                if is_new:
                                    # Rollback the incorrectly created new person
                                    self.person_db._rollback_new_person(person_id)
                                    is_new = False

                                person_id = verified_id
                                similarity = verified_sim

                                # Log corrections (reduced frequency in optimized mode)
                                if not self.optimize_fps or self.id_corrections <= 20 or self.id_corrections % 10 == 0:
                                    logger.info(
                                        "global_reid_correction",
                                        tracker_id=tracker_id,
                                        corrected_to_person_id=person_id,
                                        similarity=similarity,
                                        reason=reason,
                                        frame=self.frame_count,
                                        total_corrections=self.id_corrections
                                    )

                            # Step 4: Register/update person in global verifier
                            if is_new:
                                self.global_verifier.register_person(
                                    person_id=person_id,
                                    frame=self.frame_count,
                                    embedding=embedding,
                                    bbox=bbox,
                                    confidence=confidence
                                )
                            else:
                                # Update existing person's record
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
                                    # Person exists in DB but not in verifier - register it
                                    self.global_verifier.register_person(
                                        person_id=person_id,
                                        frame=self.frame_count,
                                        embedding=embedding,
                                        bbox=bbox,
                                        confidence=confidence
                                    )

                            # Update tracker → person mapping
                            self.tracker_to_person_map[tracker_id] = person_id

                            # Update tracker history for legacy compatibility
                            self.tracker_history[tracker_id] = {
                                'person_id': person_id,
                                'embedding': embedding.copy(),
                                'last_frame': self.frame_count
                            }

                            # ============================================================
                            # FPS OPTIMIZATION: Update stability tracking and cache
                            # ============================================================
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
                                logger.info(
                                    "new_person_detected",
                                    person_id=person_id,
                                    tracker_id=tracker_id,
                                    frame=self.frame_count,
                                    similarity=similarity
                                )
                            elif self.frame_count % 100 == 0:
                                # Log re-identified persons occasionally
                                logger.info(
                                    "person_reidentified",
                                    person_id=person_id,
                                    tracker_id=tracker_id,
                                    frame=self.frame_count,
                                    similarity=similarity
                                )

                        # Update display text to show persistent person ID
                        # Include demographics if available
                        display_text = f"Person {person_id}"
                        if person_id in self.person_demographics:
                            demo = self.person_demographics[person_id]
                            display_text = f"P{person_id} {demo['gender'][0].upper()}{demo['age']}"

                        obj_meta.text_params.display_text = display_text
                    else:
                        # No embedding yet (tracker warming up)
                        # Use tracker ID temporarily
                        if tracker_id in self.tracker_to_person_map:
                            person_id = self.tracker_to_person_map[tracker_id]
                            display_text = f"Person {person_id}"
                            if person_id in self.person_demographics:
                                demo = self.person_demographics[person_id]
                                display_text = f"P{person_id} {demo['gender'][0].upper()}{demo['age']}"
                            obj_meta.text_params.display_text = display_text
                        else:
                            obj_meta.text_params.display_text = f"Track {tracker_id}"

                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            # Display FPS on frame
            display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            display_meta.num_labels = 1
            py_nvosd_text_params = display_meta.text_params[0]
            py_nvosd_text_params.display_text = f"Frame: {self.frame_count} | FPS: {self.fps:.2f} | Unique Persons: {self.person_count}"
            py_nvosd_text_params.x_offset = 10
            py_nvosd_text_params.y_offset = 12
            py_nvosd_text_params.font_params.font_name = "Serif"
            py_nvosd_text_params.font_params.font_size = 14
            py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
            py_nvosd_text_params.set_bg_clr = 1
            py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)
            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

    def extract_reid_embedding(self, obj_meta) -> Optional[np.ndarray]:
        """
        Extract ReID embedding from tracker metadata

        The NvDCF tracker with ReID enabled attaches embedding vectors to
        object metadata with type NVDS_TRACKER_OBJ_REID_META.

        Args:
            obj_meta: NvDsObjectMeta containing tracker information

        Returns:
            256-dim numpy array or None if not available
        """
        # Access user metadata list attached to this object
        l_user = obj_meta.obj_user_meta_list
        while l_user is not None:
            try:
                user_meta = pyds.NvDsUserMeta.cast(l_user.data)
            except StopIteration:
                break

            # Check if this is object-level ReID metadata from tracker
            if user_meta.base_meta.meta_type == pyds.NVDS_TRACKER_OBJ_REID_META:
                try:
                    # Cast to NvDsObjReid structure
                    reid_meta = pyds.NvDsObjReid.cast(user_meta.user_meta_data)

                    # Check if we have valid feature data
                    if reid_meta.featureSize > 0:
                        # Get the ReID embedding as numpy array
                        embedding = reid_meta.get_host_reid_vector()

                        if embedding is not None and len(embedding) > 0:
                            # Log first successful extraction
                            if self.frame_count <= 5:
                                logger.info(
                                    "reid_embedding_extracted",
                                    frame=self.frame_count,
                                    tracker_id=obj_meta.object_id,
                                    embedding_dim=len(embedding),
                                    embedding_norm=float(np.linalg.norm(embedding))
                                )
                            return embedding

                except Exception as e:
                    if self.frame_count <= 10:
                        logger.warning(
                            "failed_to_cast_reid_meta",
                            error=str(e),
                            frame=self.frame_count
                        )

            try:
                l_user = l_user.next
            except StopIteration:
                break

        # If no ReID metadata found, log occasionally for debugging
        if self.frame_count <= 10 and obj_meta.class_id == 0:
            logger.debug(
                "no_reid_metadata",
                frame=self.frame_count,
                tracker_id=obj_meta.object_id
            )

        return None

    def process_demographics(
        self,
        person_id: int,
        person_bbox: Tuple[int, int, int, int],
        frame_image: np.ndarray
    ) -> Optional[Dict]:
        """
        Process demographics for a person (face detection + age/gender estimation)

        Args:
            person_id: Person ID from RE-ID
            person_bbox: Person bounding box (x1, y1, x2, y2)
            frame_image: Full frame BGR image

        Returns:
            Demographics dict or None
        """
        if not self.enable_demographics or self.face_detector is None:
            return None

        # Check if we need to update demographics for this person
        if person_id in self.person_demographics:
            cached = self.person_demographics[person_id]
            frames_since_update = self.frame_count - cached.get('last_update', 0)
            if frames_since_update < self.DEMOGRAPHICS_UPDATE_INTERVAL:
                # Return cached demographics
                return cached

        try:
            # Detect faces within person bounding box
            faces = self.face_detector.detect_in_person_crop(
                frame_image,
                person_bbox,
                expand_ratio=0.1
            )

            if not faces:
                return None

            # Take the largest/most confident face
            best_face = max(faces, key=lambda f: f.area * f.confidence)

            # Crop face for demographics
            x1, y1, x2, y2 = best_face.bbox
            face_crop = frame_image[y1:y2, x1:x2]

            if face_crop.size == 0:
                return None

            # Estimate demographics
            demographics = self.demographics_estimator.estimate(face_crop)

            if demographics is None:
                return None

            # Calculate attention score if landmarks available
            attention_score = 0.0
            if best_face.landmarks is not None:
                yaw, pitch, roll = self.head_pose_estimator.estimate_from_landmarks(
                    best_face.landmarks,
                    (frame_image.shape[1], frame_image.shape[0])
                )
                attention_score = self.head_pose_estimator.calculate_attention_score(yaw, pitch)

            # Build result
            result = {
                'age': demographics.age,
                'gender': demographics.gender,
                'age_group': demographics.age_group,
                'gender_confidence': demographics.gender_confidence,
                'attention_score': attention_score,
                'face_confidence': best_face.confidence,
                'last_update': self.frame_count
            }

            # Cache the result
            self.person_demographics[person_id] = result

            # Update attention tracking
            if person_id not in self.person_attention:
                self.person_attention[person_id] = {
                    'total_attention_time': 0.0,
                    'attention_frames': 0,
                    'high_attention_frames': 0
                }

            self.person_attention[person_id]['attention_frames'] += 1
            if attention_score > 0.5:
                self.person_attention[person_id]['high_attention_frames'] += 1
                # Assuming 30 FPS, each high-attention frame = 1/30 seconds
                self.person_attention[person_id]['total_attention_time'] += 1.0 / 30.0

            # Log new demographics detection
            logger.info(
                "demographics_detected",
                person_id=person_id,
                age=demographics.age,
                gender=demographics.gender,
                age_group=demographics.age_group,
                attention_score=attention_score,
                frame=self.frame_count
            )

            return result

        except Exception as e:
            logger.warning("demographics_processing_error", error=str(e), person_id=person_id)
            return None

    def detect_and_correct_id_switch(
        self,
        tracker_id: int,
        current_embedding: np.ndarray,
        frame_number: int
    ) -> Optional[int]:
        """
        Detect and correct ID switches caused by trajectory merging.

        When two people pass close to each other, the tracker may swap their IDs.
        This function detects such switches using TWO methods:

        Method 1: Embedding Drift Detection
        - If the same tracker_id suddenly has a very different embedding,
          the tracker likely switched to a different person.

        Method 2: Database Mismatch Detection (NEW - fixes Person 1 ↔ Person 6 swap)
        - Even if embeddings are consistent within a tracker, check if the
          database finds a DIFFERENT person as best match.
        - This catches cases where NvDCF swapped tracker_ids between two people.

        Args:
            tracker_id: Current tracker ID from NvDCF
            current_embedding: Current ReID embedding
            frame_number: Current frame number

        Returns:
            Corrected person_id if ID switch detected, None otherwise
        """
        # Normalize current embedding
        current_norm = current_embedding / (np.linalg.norm(current_embedding) + 1e-8)
        current_normalized = current_norm.reshape(1, -1).astype('float32')

        # Check if we have history for this tracker_id
        if tracker_id not in self.tracker_history:
            return None

        history = self.tracker_history[tracker_id]
        prev_embedding = history['embedding']
        prev_person_id = history['person_id']
        prev_frame = history['last_frame']

        # Skip if frames are too far apart (person likely left and returned)
        frame_gap = frame_number - prev_frame
        if frame_gap > 30:  # More than 1 second gap at 30fps
            return None

        # Calculate similarity between current and previous embedding
        prev_norm = prev_embedding / (np.linalg.norm(prev_embedding) + 1e-8)
        embedding_similarity = float(np.dot(current_norm, prev_norm))

        # ==============================================================
        # METHOD 1: Embedding Drift Detection (original logic)
        # If same tracker has drastically different embedding
        # ==============================================================
        EMBEDDING_DRIFT_THRESHOLD = 0.70

        if embedding_similarity < EMBEDDING_DRIFT_THRESHOLD:
            # Potential ID switch - embedding changed drastically
            if self.person_db.index.ntotal > 0:
                similarities, indices = self.person_db.index.search(
                    current_normalized,
                    k=min(5, self.person_db.index.ntotal)
                )

                best_similarity = similarities[0][0]
                best_idx = int(indices[0][0])
                best_person_id = self.person_db.index_to_person_id[best_idx]

                if best_similarity > 0.70 and best_person_id != prev_person_id:
                    self.id_corrections += 1
                    logger.info(
                        "id_switch_corrected_drift",
                        tracker_id=tracker_id,
                        old_person_id=prev_person_id,
                        new_person_id=best_person_id,
                        frame=frame_number,
                        embedding_similarity=embedding_similarity,
                        db_match_similarity=best_similarity,
                        total_corrections=self.id_corrections
                    )
                    return best_person_id

        # ==============================================================
        # METHOD 2: Database Mismatch Detection (NEW)
        # Even with consistent embedding, check if DB suggests different person
        # This catches trajectory swaps where tracker follows wrong person
        # ==============================================================
        DB_MISMATCH_THRESHOLD = 0.75  # Must be confident it's a different person
        CONSISTENCY_CHECK_INTERVAL = 10  # Check every N frames for efficiency

        if frame_gap <= 5 and self.person_db.index.ntotal > 1:
            # Only check when tracking is active and we have multiple persons
            similarities, indices = self.person_db.index.search(
                current_normalized,
                k=min(5, self.person_db.index.ntotal)
            )

            best_similarity = similarities[0][0]
            best_idx = int(indices[0][0])
            best_person_id = self.person_db.index_to_person_id[best_idx]

            # Check if database says this is a DIFFERENT person with high confidence
            if best_person_id != prev_person_id and best_similarity > DB_MISMATCH_THRESHOLD:
                # Verify: also check similarity to the PREVIOUS person's embeddings
                # Get the previous person's average embedding from database
                prev_person_similarity = self._get_similarity_to_person(
                    current_normalized, prev_person_id
                )

                # If current embedding matches a different person MUCH better
                # than the person this tracker was assigned to, it's an ID switch
                SWITCH_MARGIN = 0.10  # Best match must be 10% better

                if prev_person_similarity is not None:
                    if best_similarity - prev_person_similarity > SWITCH_MARGIN:
                        self.id_corrections += 1
                        logger.info(
                            "id_switch_corrected_mismatch",
                            tracker_id=tracker_id,
                            old_person_id=prev_person_id,
                            new_person_id=best_person_id,
                            frame=frame_number,
                            embedding_consistency=embedding_similarity,
                            db_best_match_similarity=best_similarity,
                            db_prev_person_similarity=prev_person_similarity,
                            switch_margin=best_similarity - prev_person_similarity,
                            total_corrections=self.id_corrections
                        )
                        return best_person_id

        return None

    def _get_similarity_to_person(
        self,
        embedding: np.ndarray,
        person_id: int
    ) -> Optional[float]:
        """
        Get similarity between an embedding and a specific person's stored embeddings.

        Args:
            embedding: Query embedding (normalized, shape 1x256)
            person_id: Person ID to compare against

        Returns:
            Similarity score or None if person not found
        """
        # Find all embeddings for this person in the database
        person_indices = [
            idx for idx, pid in self.person_db.index_to_person_id.items()
            if pid == person_id
        ]

        if not person_indices:
            return None

        # Get the average similarity across this person's embeddings
        max_similarity = 0.0
        for idx in person_indices[:10]:  # Limit to recent 10 embeddings
            if idx < self.person_db.index.ntotal:
                stored_embedding = self.person_db.index.reconstruct(idx)
                similarity = float(np.dot(embedding.flatten(), stored_embedding))
                max_similarity = max(max_similarity, similarity)

        return max_similarity

    def bus_call(self, bus, message):
        """Handle GStreamer bus messages"""
        t = message.type
        if t == Gst.MessageType.EOS:
            logger.info("end_of_stream")
            self.loop.quit()
        elif t == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            logger.warning("gstreamer_warning", error=err.message, debug=debug)
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            logger.error("gstreamer_error", error=err.message, debug=debug)
            self.loop.quit()
        return True

    def run(self):
        """Run the pipeline"""
        # Build pipeline
        if not self.build_pipeline():
            logger.error("pipeline_build_failed")
            return False

        # Create event loop
        self.loop = GLib.MainLoop()

        # Start pipeline
        logger.info("starting_pipeline")
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            logger.error("failed_to_set_pipeline_to_playing")
            return False

        # Run event loop
        try:
            self.loop.run()
        except KeyboardInterrupt:
            logger.info("keyboard_interrupt")
        except Exception as e:
            logger.error("pipeline_error", error=str(e))

        # Cleanup
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
            avg_fps=self.fps,
            output_path=str(self.output_path),
            id_corrections=self.id_corrections,
            verifier_stats=verifier_stats,
            frames_skipped=self.frames_skipped if self.optimize_fps else 0,
            optimization_enabled=self.optimize_fps
        )

        return True


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="DeepStream Person Analytics Pipeline")
    parser.add_argument("--video", required=True, help="Input video file")
    parser.add_argument("--output", required=True, help="Output video file")
    parser.add_argument(
        "--detector-config",
        default="configs/yolov8_detector_config.txt",
        help="Detector config file"
    )
    parser.add_argument(
        "--tracker-config",
        default="configs/nvdcf_tracker_config.yml",
        help="Tracker config file"
    )
    parser.add_argument(
        "--db-path",
        default="output/database/person_database.db",
        help="Person database path"
    )

    args = parser.parse_args()

    # Setup logger
    from utils.logger import setup_logger
    setup_logger(log_level="INFO")

    # Create and run pipeline
    pipeline = DeepStreamPipeline(
        video_path=args.video,
        output_path=args.output,
        detector_config=args.detector_config,
        tracker_config=args.tracker_config,
        db_path=args.db_path
    )

    success = pipeline.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
