#!/usr/bin/env python3
"""
IP AI v3 - DeepStream Camera Pipeline

GPU-accelerated video pipeline for person detection and tracking
using NVIDIA DeepStream SDK 7.1.
"""

import sys
import os
import time
import ctypes
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from threading import Thread, Event
import numpy as np

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

import pyds

import structlog

# Import attention tracker and head pose estimator
from ..analytics import AttentionTracker, AttentionState, HeadPoseONNX

logger = structlog.get_logger(__name__)


@dataclass
class DetectionResult:
    """Single detection result."""
    class_id: int
    class_name: str
    confidence: float
    tracker_id: int
    bbox: tuple  # (left, top, width, height)
    frame_num: int
    source_id: int
    embedding: Optional[np.ndarray] = None  # RE-ID embedding from tracker (256-dim)
    person_id: Optional[int] = None  # Persistent person ID from database
    # Demographics from SGIE (for face detections or associated persons)
    age: Optional[int] = None
    gender: Optional[str] = None  # "Male" or "Female"
    age_group: Optional[str] = None  # "0-17", "18-29", "30-44", "45-59", "60+"
    # Head pose from SGIE-2
    head_yaw: Optional[float] = None
    head_pitch: Optional[float] = None
    head_roll: Optional[float] = None
    # Attention state
    attention_state: Optional[str] = None  # "NOT_LOOKING", "LOOKING", "ENGAGED"


@dataclass
class FrameMetadata:
    """Metadata for a single frame."""
    frame_num: int
    source_id: int
    timestamp: float
    detections: List[DetectionResult] = field(default_factory=list)
    fps: float = 0.0


class CameraPipeline:
    """
    DeepStream pipeline for USB camera with person detection.

    Pipeline structure:
    v4l2src -> nvvideoconvert -> nvstreammux -> nvinfer (PeopleNet) ->
    nvtracker -> nvvideoconvert -> nvdsosd -> [display/encode]
    """

    # Class labels for PeopleNet
    PGIE_CLASSES = ['person', 'bag', 'face']

    def __init__(
        self,
        camera_device: str = "/dev/video0",
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        pgie_config: str = "configs/pgie_config.txt",
        tracker_config: Optional[str] = "configs/tracker_config.txt",
        sgie_config: Optional[str] = None,  # Demographics SGIE config
        headpose_model_path: Optional[str] = None,  # Head pose ONNX model path
        output_file: Optional[str] = None,
        display: bool = True,
        on_frame_callback: Optional[Callable[[FrameMetadata], None]] = None,
        person_database: Optional[Any] = None,  # PersonDatabase instance
        reid_threshold: float = 0.50,  # Cosine similarity threshold for RE-ID (adaptive)
        # Attention tracking thresholds
        yaw_threshold: float = 30.0,
        pitch_threshold: float = 20.0,
        engagement_time: float = 2.0
    ):
        """
        Initialize camera pipeline.

        Args:
            camera_device: V4L2 camera device path
            width: Frame width
            height: Frame height
            fps: Target framerate
            pgie_config: Path to primary GIE config file
            tracker_config: Path to tracker config file (None to disable)
            sgie_config: Path to demographics SGIE config file (None to disable)
            headpose_model_path: Path to head pose ONNX model (None to disable)
            output_file: Path to save output video (None for no save)
            display: Whether to display output (requires display)
            on_frame_callback: Callback function for frame metadata
            person_database: PersonDatabase instance for persistent RE-ID
            reid_threshold: Cosine similarity threshold for person matching
            yaw_threshold: Max yaw angle (degrees) to consider "looking"
            pitch_threshold: Max pitch angle (degrees) to consider "looking"
            engagement_time: Seconds of looking before marking as "engaged"
        """
        self.camera_device = camera_device
        self.width = width
        self.height = height
        self.fps = fps
        self.pgie_config = pgie_config
        self.tracker_config = tracker_config
        self.sgie_config = sgie_config
        self.headpose_model_path = headpose_model_path
        self.output_file = output_file
        self.display = display
        self.on_frame_callback = on_frame_callback
        self.person_database = person_database
        self.reid_threshold = reid_threshold
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.engagement_time = engagement_time

        # Pipeline state
        self.pipeline = None
        self.loop = None
        self.running = False
        self._stop_event = Event()

        # FPS calculation
        self._frame_count = 0
        self._start_time = 0
        self._fps = 0.0

        # Track ID to Person ID cache (for fast lookup)
        self._tracker_to_person: Dict[int, int] = {}

        # Demographics cache: person_id -> (gender, age, age_group)
        self._person_demographics: Dict[int, Tuple[str, int, str]] = {}

        # Head pose cache: face_bbox -> (yaw, pitch, roll) for current frame
        self._frame_face_headpose: Dict[Tuple[float, float, float, float], Tuple[float, float, float]] = {}

        # Initialize head pose estimator (ONNX Runtime) and attention tracker
        self.headpose_estimator: Optional[HeadPoseONNX] = None
        self.attention_tracker: Optional[AttentionTracker] = None
        if self.headpose_model_path and os.path.exists(self.headpose_model_path):
            # Initialize ONNX Runtime head pose estimator
            self.headpose_estimator = HeadPoseONNX(
                model_path=self.headpose_model_path,
                use_gpu=False  # CPU inference is sufficient for face crops
            )
            logger.info(
                "headpose_estimator_initialized",
                model_path=self.headpose_model_path
            )

            # Initialize attention tracker
            self.attention_tracker = AttentionTracker(
                yaw_threshold=self.yaw_threshold,
                pitch_threshold=self.pitch_threshold,
                engagement_threshold=self.engagement_time
            )
            logger.info(
                "attention_tracker_initialized",
                yaw_threshold=self.yaw_threshold,
                pitch_threshold=self.pitch_threshold,
                engagement_time=self.engagement_time
            )

        # Initialize GStreamer
        Gst.init(None)

        logger.info(
            "camera_pipeline_initialized",
            camera=camera_device,
            resolution=f"{width}x{height}",
            fps=fps
        )

    @staticmethod
    def _parse_demographics(tensor_data: np.ndarray) -> tuple:
        """
        Parse demographics from InsightFace GenderAge model output.

        Args:
            tensor_data: fc1 output tensor of shape [3] containing:
                - [0]: female logit
                - [1]: male logit
                - [2]: age / 100

        Returns:
            Tuple of (age, gender, age_group)
        """
        if tensor_data is None or len(tensor_data) < 3:
            return None, None, None

        # Extract gender from InsightFace GenderAge model
        # Output format: [gender_logit, -gender_logit, age/100]
        # Positive gender_logit = Male, Negative = Female
        gender_logit = tensor_data[0]
        gender = "Male" if gender_logit > 0 else "Female"

        # Extract age (scaled by 100)
        age = int(tensor_data[2] * 100)
        age = max(0, min(100, age))  # Clamp to 0-100

        # Determine age group
        if age < 18:
            age_group = "0-17"
        elif age < 30:
            age_group = "18-29"
        elif age < 45:
            age_group = "30-44"
        elif age < 60:
            age_group = "45-59"
        else:
            age_group = "60+"

        return age, gender, age_group

    @staticmethod
    def _parse_headpose_euler(tensor_data: np.ndarray) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Parse head pose from direct Euler angle output.

        Args:
            tensor_data: Output tensor of shape [3] containing [yaw, roll, pitch] in degrees
                         (WHENet output order)

        Returns:
            Tuple of (yaw, pitch, roll) in degrees, or (None, None, None) if invalid
        """
        if tensor_data is None or len(tensor_data) < 3:
            return None, None, None

        # WHENet outputs [yaw, roll, pitch] - reorder to standard [yaw, pitch, roll]
        yaw = float(tensor_data[0])
        pitch = float(tensor_data[2])  # WHENet: index 2 is pitch
        roll = float(tensor_data[1])   # WHENet: index 1 is roll

        # Clamp to valid range (-180 to 180)
        yaw = max(-180.0, min(180.0, yaw))
        pitch = max(-180.0, min(180.0, pitch))
        roll = max(-180.0, min(180.0, roll))

        return yaw, pitch, roll

    @staticmethod
    def _parse_headpose_6d(tensor_data: np.ndarray) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Parse head pose from 6D rotation representation (first two columns of rotation matrix).

        Args:
            tensor_data: Output tensor of shape [6] containing rotation matrix columns

        Returns:
            Tuple of (yaw, pitch, roll) in degrees, or (None, None, None) if invalid
        """
        if tensor_data is None or len(tensor_data) < 6:
            return None, None, None

        try:
            # 6D representation: [r11, r21, r31, r12, r22, r32]
            # First column of rotation matrix
            r1 = np.array([tensor_data[0], tensor_data[1], tensor_data[2]])
            # Second column
            r2 = np.array([tensor_data[3], tensor_data[4], tensor_data[5]])

            # Gram-Schmidt orthogonalization
            r1 = r1 / (np.linalg.norm(r1) + 1e-8)
            r2 = r2 - np.dot(r1, r2) * r1
            r2 = r2 / (np.linalg.norm(r2) + 1e-8)

            # Third column via cross product
            r3 = np.cross(r1, r2)

            # Construct rotation matrix
            R = np.column_stack([r1, r2, r3])

            # Convert to Euler angles (ZYX convention)
            # pitch = arcsin(-r31)
            # yaw = atan2(r21, r11)
            # roll = atan2(r32, r33)
            pitch = np.arcsin(-np.clip(R[2, 0], -1, 1))
            yaw = np.arctan2(R[1, 0], R[0, 0])
            roll = np.arctan2(R[2, 1], R[2, 2])

            # Convert to degrees
            yaw_deg = float(np.degrees(yaw))
            pitch_deg = float(np.degrees(pitch))
            roll_deg = float(np.degrees(roll))

            return yaw_deg, pitch_deg, roll_deg

        except Exception:
            return None, None, None

    @staticmethod
    def _bbox_overlap_ratio(face_bbox: Tuple[float, float, float, float],
                            person_bbox: Tuple[float, float, float, float]) -> float:
        """
        Calculate how much of the face bbox is inside the person bbox.

        Returns ratio of face area inside person (0.0 to 1.0).
        """
        fx, fy, fw, fh = face_bbox
        px, py, pw, ph = person_bbox

        # Calculate intersection
        ix1 = max(fx, px)
        iy1 = max(fy, py)
        ix2 = min(fx + fw, px + pw)
        iy2 = min(fy + fh, py + ph)

        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0  # No intersection

        intersection = (ix2 - ix1) * (iy2 - iy1)
        face_area = fw * fh

        return intersection / face_area if face_area > 0 else 0.0

    def _create_element(self, factory_name: str, name: str) -> Gst.Element:
        """Create a GStreamer element with error checking."""
        element = Gst.ElementFactory.make(factory_name, name)
        if not element:
            raise RuntimeError(f"Failed to create element: {factory_name} ({name})")
        return element

    def _build_pipeline(self) -> Gst.Pipeline:
        """Build the DeepStream pipeline."""
        logger.info("building_pipeline")

        # Create pipeline
        pipeline = Gst.Pipeline()
        if not pipeline:
            raise RuntimeError("Failed to create pipeline")

        # ============ Source Elements ============
        # V4L2 source (USB camera)
        source = self._create_element("v4l2src", "source")
        source.set_property("device", self.camera_device)

        # Caps filter for camera format (MJPG for most USB cameras)
        caps_v4l2 = self._create_element("capsfilter", "caps_v4l2")
        caps_v4l2.set_property(
            "caps",
            Gst.Caps.from_string(
                f"image/jpeg, width={self.width}, height={self.height}, "
                f"framerate={self.fps}/1"
            )
        )

        # JPEG decoder for MJPG stream
        jpegdec = self._create_element("jpegdec", "jpegdec")

        # Video convert (CPU to GPU)
        vidconv_src = self._create_element("nvvideoconvert", "vidconv_src")

        # Caps filter for NVMM memory
        caps_nvmm = self._create_element("capsfilter", "caps_nvmm")
        caps_nvmm.set_property(
            "caps",
            Gst.Caps.from_string(
                "video/x-raw(memory:NVMM), format=NV12"
            )
        )

        # ============ Stream Muxer ============
        streammux = self._create_element("nvstreammux", "streammux")
        streammux.set_property("batch-size", 1)
        streammux.set_property("width", self.width)
        streammux.set_property("height", self.height)
        streammux.set_property("batched-push-timeout", 4000000)  # 4ms
        streammux.set_property("live-source", 1)

        # ============ Primary Inference (Person Detection) ============
        pgie = self._create_element("nvinfer", "pgie")
        pgie.set_property("config-file-path", self.pgie_config)

        # ============ Tracker (Optional) ============
        tracker = None
        if self.tracker_config and os.path.exists(self.tracker_config):
            tracker = self._create_element("nvtracker", "tracker")
            # Parse tracker config
            self._configure_tracker(tracker, self.tracker_config)

        # ============ Secondary Inference - Demographics (Optional) ============
        sgie = None
        if self.sgie_config and os.path.exists(self.sgie_config):
            sgie = self._create_element("nvinfer", "sgie")
            sgie.set_property("config-file-path", self.sgie_config)
            logger.info("sgie_demographics_enabled", config=self.sgie_config)

        # ============ Head Pose (ONNX Runtime - handled in probe callback) ============
        # Head pose estimation is now done via ONNX Runtime in the probe callback
        # instead of using a DeepStream SGIE element (which has caps negotiation issues)
        if self.headpose_estimator:
            logger.info("headpose_onnx_enabled", model_path=self.headpose_model_path)

        # ============ Video Convert for OSD ============
        nvvidconv = self._create_element("nvvideoconvert", "nvvidconv")

        # ============ On-Screen Display ============
        nvosd = self._create_element("nvdsosd", "nvosd")
        nvosd.set_property("process-mode", 0)  # CPU mode for OSD
        nvosd.set_property("display-text", 1)

        # ============ Output Elements ============
        if self.display:
            # Try EGL sink first (for display), fallback to fakesink
            try:
                sink = self._create_element("nveglglessink", "sink")
                sink.set_property("sync", 0)  # Async for better performance
            except Exception:
                logger.warning("nveglglessink_not_available_using_fakesink")
                sink = self._create_element("fakesink", "sink")
                sink.set_property("sync", 0)
        else:
            sink = self._create_element("fakesink", "sink")
            sink.set_property("sync", 0)

        # ============ Add Elements to Pipeline ============
        elements = [
            source, caps_v4l2, jpegdec, vidconv_src, caps_nvmm,
            streammux, pgie
        ]

        if tracker:
            elements.append(tracker)

        if sgie:
            elements.append(sgie)

        elements.extend([nvvidconv, nvosd, sink])

        for element in elements:
            pipeline.add(element)

        # ============ Link Elements ============
        # Link source chain (v4l2src -> caps -> jpegdec -> nvvideoconvert -> nvmm)
        if not source.link(caps_v4l2):
            raise RuntimeError("Failed to link source -> caps_v4l2")
        if not caps_v4l2.link(jpegdec):
            raise RuntimeError("Failed to link caps_v4l2 -> jpegdec")
        if not jpegdec.link(vidconv_src):
            raise RuntimeError("Failed to link jpegdec -> vidconv_src")
        if not vidconv_src.link(caps_nvmm):
            raise RuntimeError("Failed to link vidconv_src -> caps_nvmm")

        # Link to streammux (request sink pad)
        sinkpad = streammux.get_request_pad("sink_0")
        if not sinkpad:
            raise RuntimeError("Failed to get streammux sink pad")
        srcpad = caps_nvmm.get_static_pad("src")
        if srcpad.link(sinkpad) != Gst.PadLinkReturn.OK:
            raise RuntimeError("Failed to link caps_nvmm -> streammux")

        # Link inference chain
        if not streammux.link(pgie):
            raise RuntimeError("Failed to link streammux -> pgie")

        # Link inference chain: pgie -> [tracker] -> [sgie] -> nvvidconv
        last_inference_element = pgie

        if tracker:
            if not pgie.link(tracker):
                raise RuntimeError("Failed to link pgie -> tracker")
            last_inference_element = tracker

        if sgie:
            if not last_inference_element.link(sgie):
                raise RuntimeError(f"Failed to link {last_inference_element.get_name()} -> sgie")
            last_inference_element = sgie

        if not last_inference_element.link(nvvidconv):
            raise RuntimeError(f"Failed to link {last_inference_element.get_name()} -> nvvidconv")

        if not nvvidconv.link(nvosd):
            raise RuntimeError("Failed to link nvvidconv -> nvosd")
        if not nvosd.link(sink):
            raise RuntimeError("Failed to link nvosd -> sink")

        # ============ Add Probes ============
        # Add probe on OSD sink pad for metadata extraction
        osd_sink_pad = nvosd.get_static_pad("sink")
        if not osd_sink_pad:
            raise RuntimeError("Failed to get OSD sink pad")

        osd_sink_pad.add_probe(
            Gst.PadProbeType.BUFFER,
            self._osd_sink_pad_buffer_probe,
            0
        )

        logger.info(
            "pipeline_built",
            tracker_enabled=tracker is not None,
            sgie_demographics_enabled=sgie is not None,
            headpose_onnx_enabled=self.headpose_estimator is not None
        )

        return pipeline

    def _configure_tracker(self, tracker: Gst.Element, config_path: str):
        """Configure tracker from config file."""
        # Read tracker config
        config = {}
        with open(config_path, 'r') as f:
            section = None
            for line in f:
                line = line.strip()
                if line.startswith('[') and line.endswith(']'):
                    section = line[1:-1]
                elif '=' in line and section == 'tracker':
                    key, value = line.split('=', 1)
                    config[key.strip()] = value.strip()

        # Apply tracker properties (only properties supported by nvtracker GstElement)
        # Note: enable-batch-process and enable-past-frame are set in YAML ll-config-file
        if 'tracker-width' in config:
            tracker.set_property("tracker-width", int(config['tracker-width']))
        if 'tracker-height' in config:
            tracker.set_property("tracker-height", int(config['tracker-height']))
        if 'll-lib-file' in config:
            tracker.set_property("ll-lib-file", config['ll-lib-file'])
        if 'll-config-file' in config:
            tracker.set_property("ll-config-file", config['ll-config-file'])
        if 'display-tracking-id' in config:
            tracker.set_property(
                "display-tracking-id",
                int(config['display-tracking-id'])
            )
        if 'gpu-id' in config:
            tracker.set_property("gpu-id", int(config['gpu-id']))

        logger.info("tracker_configured", config=config)

    def _osd_sink_pad_buffer_probe(
        self,
        pad: Gst.Pad,
        info: Gst.PadProbeInfo,
        u_data
    ) -> Gst.PadProbeReturn:
        """
        Probe callback for extracting frame metadata.

        This is called for every frame and extracts detection results.
        """
        # Get batch metadata
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if not batch_meta:
            return Gst.PadProbeReturn.OK

        # Update FPS calculation
        self._frame_count += 1
        if self._frame_count == 1:
            self._start_time = time.time()
        elif self._frame_count % 30 == 0:
            elapsed = time.time() - self._start_time
            if elapsed > 0:
                self._fps = self._frame_count / elapsed

        # Iterate through frames in batch
        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            frame_data = FrameMetadata(
                frame_num=frame_meta.frame_num,
                source_id=frame_meta.source_id,
                timestamp=time.time(),
                fps=self._fps
            )

            # Count persons in this frame
            person_count = 0

            # ============ PASS 1: Collect face demographics ============
            # Store face demographics by bbox: (left, top, width, height) -> (gender, age, age_group)
            frame_face_demographics: Dict[Tuple[float, float, float, float], Tuple[str, int, str]] = {}

            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break

                # Only process faces in first pass
                if obj_meta.class_id == 2 and self.sgie_config:
                    rect = obj_meta.rect_params
                    face_bbox = (rect.left, rect.top, rect.width, rect.height)

                    # Extract demographics from tensor metadata
                    try:
                        l_user = obj_meta.obj_user_meta_list
                        while l_user is not None:
                            try:
                                user_meta = pyds.NvDsUserMeta.cast(l_user.data)
                                if user_meta.base_meta.meta_type == pyds.NVDSINFER_TENSOR_OUTPUT_META:
                                    tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
                                    for layer_idx in range(tensor_meta.num_output_layers):
                                        layer_info = pyds.get_nvds_LayerInfo(tensor_meta, layer_idx)
                                        if layer_info.layerName == "fc1":
                                            ptr = ctypes.cast(
                                                pyds.get_ptr(layer_info.buffer),
                                                ctypes.POINTER(ctypes.c_float)
                                            )
                                            raw_values = [ptr[i] for i in range(min(3, layer_info.dims.numElements))]
                                            tensor_data = np.array(raw_values[:3])
                                            age, gender, age_group = self._parse_demographics(tensor_data)
                                            if gender is not None and age is not None:
                                                frame_face_demographics[face_bbox] = (gender, age, age_group)
                                            break
                                l_user = l_user.next
                            except StopIteration:
                                break
                    except Exception:
                        pass

                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            # ============ PASS 1.5: Collect face head pose ============
            # Store face head pose by bbox: (left, top, width, height) -> (yaw, pitch, roll)
            frame_face_headpose: Dict[Tuple[float, float, float, float], Tuple[float, float, float]] = {}

            if self.headpose_estimator:
                face_count_pass15 = 0
                headpose_extracted = 0

                # Get frame buffer for face crop extraction (RGBA format)
                frame_surface = None
                buffer_mapped = False
                try:
                    frame_surface = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
                    buffer_mapped = True
                except Exception as e:
                    if frame_meta.frame_num % 30 == 0:
                        logger.warning("failed_to_get_frame_surface", error=str(e))

                try:
                    if frame_surface is not None:
                        # Iterate through faces and extract head pose
                        l_obj = frame_meta.obj_meta_list
                        while l_obj is not None:
                            try:
                                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                            except StopIteration:
                                break

                            # Only process faces for head pose
                            if obj_meta.class_id == 2:
                                face_count_pass15 += 1
                                rect = obj_meta.rect_params
                                face_bbox = (rect.left, rect.top, rect.width, rect.height)

                                # Extract face crop and run ONNX inference
                                try:
                                    left = max(0, int(rect.left))
                                    top = max(0, int(rect.top))
                                    width = int(rect.width)
                                    height = int(rect.height)

                                    # Ensure we don't exceed frame bounds
                                    frame_h, frame_w = frame_surface.shape[:2]
                                    if left + width > frame_w:
                                        width = frame_w - left
                                    if top + height > frame_h:
                                        height = frame_h - top

                                    if width > 32 and height > 32:
                                        # Extract face crop (RGBA format from DeepStream)
                                        # Make a copy to avoid holding reference to GPU buffer
                                        face_rgba = frame_surface[top:top+height, left:left+width, :].copy()
                                        # Convert RGBA to BGR for ONNX model
                                        face_bgr = face_rgba[:, :, [2, 1, 0]]

                                        # Run head pose inference via ONNX Runtime
                                        result = self.headpose_estimator.estimate(face_bgr)
                                        if result is not None:
                                            yaw, pitch, roll = result
                                            frame_face_headpose[face_bbox] = (yaw, pitch, roll)
                                            headpose_extracted += 1

                                            if frame_meta.frame_num % 30 == 0:
                                                logger.info(
                                                    "headpose_extracted",
                                                    yaw=f"{yaw:.1f}",
                                                    pitch=f"{pitch:.1f}",
                                                    roll=f"{roll:.1f}"
                                                )
                                except Exception as e:
                                    if frame_meta.frame_num % 30 == 0:
                                        logger.warning("headpose_crop_error", error=str(e))

                            try:
                                l_obj = l_obj.next
                            except StopIteration:
                                break
                finally:
                    # ALWAYS unmap buffer (required for Jetson to prevent memory leaks)
                    if buffer_mapped:
                        try:
                            pyds.unmap_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
                        except Exception:
                            pass

                # Log summary every 30 frames
                if frame_meta.frame_num % 30 == 0:
                    logger.info(
                        "headpose_pass15_summary",
                        faces_found=face_count_pass15,
                        headpose_extracted=headpose_extracted
                    )

            # ============ PASS 2: Process all objects ============
            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break

                # Get detection info
                class_id = obj_meta.class_id
                class_name = self.PGIE_CLASSES[class_id] if class_id < len(self.PGIE_CLASSES) else "unknown"
                confidence = obj_meta.confidence
                # DeepStream uses UNTRACKED_OBJECT_ID (0xFFFFFFFFFFFFFFFF) for untracked objects
                raw_tracker_id = obj_meta.object_id
                # Convert to valid track ID (0 = untracked, >0 = valid track)
                tracker_id = raw_tracker_id if raw_tracker_id < 0xFFFFFFFF else 0

                # Count persons
                if class_name == 'person':
                    person_count += 1

                # Get bounding box
                rect = obj_meta.rect_params
                bbox = (rect.left, rect.top, rect.width, rect.height)

                # Extract RE-ID embedding from tracker if available
                embedding = None
                person_id = None
                try:
                    # Try to get user meta with RE-ID embedding
                    l_user = obj_meta.obj_user_meta_list
                    while l_user is not None:
                        try:
                            user_meta = pyds.NvDsUserMeta.cast(l_user.data)
                            # Check for RE-ID metadata
                            if user_meta.base_meta.meta_type == pyds.NvDsMetaType.NVDS_TRACKER_OBJ_REID_META:
                                reid_meta = pyds.NvDsObjReid.cast(user_meta.user_meta_data)
                                feature_size = reid_meta.featureSize
                                if feature_size > 0:
                                    # Get RE-ID embedding vector (256-dim)
                                    embedding = np.array(reid_meta.get_host_reid_vector(), dtype=np.float32)
                                    # Debug: Log embedding statistics
                                    emb_norm = np.linalg.norm(embedding)
                                    self.logger.debug(
                                        "reid_embedding_extracted",
                                        tracker_id=tracker_id,
                                        embedding_size=len(embedding),
                                        embedding_norm=f"{emb_norm:.4f}",
                                        embedding_mean=f"{np.mean(embedding):.4f}",
                                        embedding_std=f"{np.std(embedding):.4f}"
                                    )
                            l_user = l_user.next
                        except StopIteration:
                            break
                except Exception:
                    pass  # Embedding not available

                # Match or create person in database if embedding available
                if embedding is not None and self.person_database is not None and tracker_id > 0:
                    # Check cache first
                    if tracker_id in self._tracker_to_person:
                        person_id = self._tracker_to_person[tracker_id]
                    else:
                        # Query database for match
                        person_id, is_new, similarity = self.person_database.match_or_create_person(
                            embedding=embedding,
                            threshold=self.reid_threshold,
                            confidence=confidence,
                            tracker_id=tracker_id,
                            bbox=bbox,
                            frame_number=frame_meta.frame_num,
                            camera_id=frame_meta.source_id
                        )
                        # Cache the mapping
                        self._tracker_to_person[tracker_id] = person_id

                # Get demographics from pass 1 cache (for faces)
                age = None
                gender = None
                age_group = None
                if class_id == 2 and bbox in frame_face_demographics:
                    gender, age, age_group = frame_face_demographics[bbox]
                    logger.info(
                        "demographics_parsed",
                        age=age,
                        gender=gender,
                        age_group=age_group
                    )

                # Get head pose from pass 1.5 cache (for faces)
                head_yaw = None
                head_pitch = None
                head_roll = None
                attention_state = None

                if class_id == 2 and bbox in frame_face_headpose:
                    head_yaw, head_pitch, head_roll = frame_face_headpose[bbox]

                # For persons, find head pose from overlapping face
                if class_id == 0 and person_id is not None and self.attention_tracker is not None:
                    for face_bbox, (f_yaw, f_pitch, f_roll) in frame_face_headpose.items():
                        overlap = self._bbox_overlap_ratio(face_bbox, bbox)
                        if overlap > 0.5:  # Face is >50% inside person bbox
                            head_yaw, head_pitch, head_roll = f_yaw, f_pitch, f_roll
                            # Update attention tracker
                            attention_state = self.attention_tracker.update(
                                person_id, head_yaw, head_pitch, head_roll
                            )
                            break

                    # If no face detected but person exists, mark as not looking
                    if head_yaw is None and person_id is not None:
                        attention_state = self.attention_tracker.update(
                            person_id, None, None, None
                        )

                detection = DetectionResult(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    tracker_id=tracker_id,
                    bbox=bbox,
                    frame_num=frame_meta.frame_num,
                    source_id=frame_meta.source_id,
                    embedding=embedding,
                    person_id=person_id,
                    age=age,
                    gender=gender,
                    age_group=age_group,
                    head_yaw=head_yaw,
                    head_pitch=head_pitch,
                    head_roll=head_roll,
                    attention_state=attention_state
                )

                frame_data.detections.append(detection)

                # Update display text based on detection type
                if class_id == 2:
                    # Face - show demographics and head pose if available (cyan color)
                    if age is not None or gender is not None:
                        gender_char = gender[0] if gender else "?"
                        age_str = str(age) if age is not None else "?"
                        display_text = f"{gender_char}{age_str} {confidence:.2f}"
                    else:
                        display_text = f"face {confidence:.2f}"
                    # Add head pose info if available
                    if head_yaw is not None:
                        display_text += f" Y:{head_yaw:.0f}"
                    obj_meta.text_params.font_params.font_color.set(0.0, 1.0, 1.0, 1.0)  # Cyan

                elif class_id == 0 and person_id is not None:
                    # Person with ID - check for associated demographics
                    person_demo = None

                    # First, try to find overlapping face from this frame
                    for face_bbox, (f_gender, f_age, f_age_group) in frame_face_demographics.items():
                        overlap = self._bbox_overlap_ratio(face_bbox, bbox)
                        if overlap > 0.5:  # Face is >50% inside person bbox
                            person_demo = (f_gender, f_age, f_age_group)
                            # Cache for future frames
                            self._person_demographics[person_id] = person_demo
                            break

                    # If no match this frame, use cached demographics
                    if person_demo is None and person_id in self._person_demographics:
                        person_demo = self._person_demographics[person_id]

                    # Build display text with demographics and attention state
                    if person_demo:
                        gender_char = person_demo[0][0]  # First char of gender
                        age_val = person_demo[1]
                        display_text = f"P{person_id} {gender_char}{age_val}"
                    else:
                        display_text = f"P{person_id}"

                    # Add head pose (Yaw/Pitch) if available
                    if head_yaw is not None and head_pitch is not None:
                        display_text += f" Y:{head_yaw:.0f} P:{head_pitch:.0f}"

                    # Add attention state label and set color based on state
                    if attention_state == "ENGAGED":
                        display_text += " [ENGAGED]"
                        display_text += f" {confidence:.2f}"
                        # Green color for engaged
                        obj_meta.text_params.font_params.font_color.set(0.0, 1.0, 0.0, 1.0)
                        # Green border
                        obj_meta.rect_params.border_color.set(0.0, 1.0, 0.0, 1.0)
                        obj_meta.rect_params.border_width = 4
                    elif attention_state == "LOOKING":
                        display_text += " [LOOKING]"
                        display_text += f" {confidence:.2f}"
                        # Yellow color for looking
                        obj_meta.text_params.font_params.font_color.set(1.0, 1.0, 0.0, 1.0)
                        # Yellow border
                        obj_meta.rect_params.border_color.set(1.0, 1.0, 0.0, 1.0)
                        obj_meta.rect_params.border_width = 3
                    else:
                        display_text += f" {confidence:.2f}"
                        # Cyan-green for default (with demographics) or green (without)
                        if person_demo:
                            obj_meta.text_params.font_params.font_color.set(0.0, 1.0, 0.5, 1.0)
                        else:
                            obj_meta.text_params.font_params.font_color.set(0.0, 1.0, 0.0, 1.0)
                        # Default border (blue)
                        obj_meta.rect_params.border_color.set(0.0, 0.5, 1.0, 1.0)
                        obj_meta.rect_params.border_width = 2

                elif tracker_id > 0:
                    display_text = f"#{tracker_id} {class_name}"
                    # Add head pose for tracked persons without person_id
                    if class_id == 0:  # Person class
                        for face_bbox, (f_yaw, f_pitch, f_roll) in frame_face_headpose.items():
                            overlap = self._bbox_overlap_ratio(face_bbox, bbox)
                            if overlap > 0.5:
                                display_text += f" Y:{f_yaw:.0f} P:{f_pitch:.0f}"
                                break
                    display_text += f" {confidence:.2f}"
                    obj_meta.text_params.font_params.font_color.set(0.0, 1.0, 0.0, 1.0)  # Green
                else:
                    display_text = f"{class_name}"
                    # Add head pose for untracked persons
                    if class_id == 0:  # Person class
                        for face_bbox, (f_yaw, f_pitch, f_roll) in frame_face_headpose.items():
                            overlap = self._bbox_overlap_ratio(face_bbox, bbox)
                            if overlap > 0.5:
                                display_text += f" Y:{f_yaw:.0f} P:{f_pitch:.0f}"
                                break
                    display_text += f" {confidence:.2f}"
                    obj_meta.text_params.font_params.font_color.set(1.0, 1.0, 0.0, 1.0)  # Yellow

                obj_meta.text_params.display_text = display_text

                # Style the text for better visibility
                obj_meta.text_params.font_params.font_name = "Serif"
                obj_meta.text_params.font_params.font_size = 18  # Larger font
                obj_meta.text_params.set_bg_clr = 1
                obj_meta.text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.8)  # More opaque background

                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            # ============ Add Statistics Overlay ============
            # Acquire display meta from pool
            display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)

            # Calculate elapsed time
            elapsed = time.time() - self._start_time if self._start_time > 0 else 0

            # Get active Person IDs and Track IDs for display
            person_detections = [d for d in frame_data.detections if d.class_name == 'person']
            person_ids = [d.person_id for d in person_detections if d.person_id is not None]
            track_ids = [d.tracker_id for d in person_detections if d.tracker_id > 0]

            # Format Person IDs (persistent)
            person_str = ",".join(str(pid) for pid in person_ids[:5]) if person_ids else "none"
            if len(person_ids) > 5:
                person_str += "..."

            # Get unique persons count from database
            unique_count = len(set(person_ids)) if person_ids else 0

            # Get attention metrics from tracker
            looking_count = 0
            engaged_count = 0
            qualified_impressions = 0
            if self.attention_tracker:
                summary = self.attention_tracker.get_summary()
                looking_count = summary.get('currently_looking', 0)
                engaged_count = summary.get('currently_engaged', 0)
                qualified_impressions = summary.get('total_qualified_impressions', 0)

            # Create statistics text with RE-ID info and attention metrics
            if self.attention_tracker:
                stats_text = (
                    f"IP AI v3 | Frame: {frame_meta.frame_num:6d} | "
                    f"FPS: {self._fps:5.1f} | "
                    f"Persons: {person_count:2d} | "
                    f"Look: {looking_count}+{engaged_count} | "
                    f"QI: {qualified_impressions} | "
                    f"Time: {elapsed:6.1f}s"
                )
            else:
                stats_text = (
                    f"IP AI v3 | Frame: {frame_meta.frame_num:6d} | "
                    f"FPS: {self._fps:5.1f} | "
                    f"Persons: {person_count:2d} | "
                    f"IDs: [P{person_str}] | "
                    f"Unique: {unique_count} | "
                    f"Time: {elapsed:6.1f}s"
                )

            # Configure text parameters for statistics overlay
            display_meta.num_labels = 1
            py_nvosd_text_params = display_meta.text_params[0]

            py_nvosd_text_params.display_text = stats_text
            py_nvosd_text_params.x_offset = 10
            py_nvosd_text_params.y_offset = 12

            # Font settings (larger for visibility)
            py_nvosd_text_params.font_params.font_name = "Serif"
            py_nvosd_text_params.font_params.font_size = 16
            py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 0.0, 1.0)  # Yellow

            # Background settings
            py_nvosd_text_params.set_bg_clr = 1
            py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.7)  # Semi-transparent black

            # Add display meta to frame
            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

            # Call user callback
            if self.on_frame_callback:
                self.on_frame_callback(frame_data)

            # Log periodically
            if frame_meta.frame_num % 100 == 0:
                log_kwargs = {
                    "frame": frame_meta.frame_num,
                    "persons": person_count,
                    "fps": f"{self._fps:.1f}"
                }
                if self.attention_tracker:
                    log_kwargs["looking"] = looking_count
                    log_kwargs["engaged"] = engaged_count
                    log_kwargs["qi"] = qualified_impressions
                logger.info("frame_processed", **log_kwargs)

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

    def _bus_call(self, bus: Gst.Bus, message: Gst.Message, loop: GLib.MainLoop):
        """Handle bus messages."""
        t = message.type

        if t == Gst.MessageType.EOS:
            logger.info("end_of_stream")
            loop.quit()

        elif t == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            logger.warning("pipeline_warning", error=str(err), debug=debug)

        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            logger.error("pipeline_error", error=str(err), debug=debug)
            loop.quit()

        elif t == Gst.MessageType.STATE_CHANGED:
            if message.src == self.pipeline:
                old_state, new_state, pending = message.parse_state_changed()
                logger.debug(
                    "state_changed",
                    old=Gst.Element.state_get_name(old_state),
                    new=Gst.Element.state_get_name(new_state)
                )

        return True

    def start(self):
        """Start the pipeline."""
        if self.running:
            logger.warning("pipeline_already_running")
            return

        logger.info("starting_pipeline")

        # Build pipeline
        self.pipeline = self._build_pipeline()

        # Create main loop
        self.loop = GLib.MainLoop()

        # Add bus watch
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._bus_call, self.loop)

        # Start pipeline
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("Failed to start pipeline")

        self.running = True
        self._stop_event.clear()

        logger.info("pipeline_started")

        # Run main loop
        try:
            self.loop.run()
        except KeyboardInterrupt:
            logger.info("keyboard_interrupt")
        finally:
            self.stop()

    def start_async(self) -> Thread:
        """Start pipeline in a background thread."""
        thread = Thread(target=self.start, daemon=True)
        thread.start()
        return thread

    def stop(self):
        """Stop the pipeline."""
        if not self.running:
            return

        logger.info("stopping_pipeline")

        self._stop_event.set()

        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)

        if self.loop and self.loop.is_running():
            self.loop.quit()

        self.running = False

        # Print final stats
        elapsed = time.time() - self._start_time if self._start_time > 0 else 0
        avg_fps = self._frame_count / elapsed if elapsed > 0 else 0

        logger.info(
            "pipeline_stopped",
            total_frames=self._frame_count,
            elapsed_seconds=f"{elapsed:.1f}",
            avg_fps=f"{avg_fps:.1f}"
        )

    def get_fps(self) -> float:
        """Get current FPS."""
        return self._fps

    def get_frame_count(self) -> int:
        """Get total frames processed."""
        return self._frame_count


def main():
    """Test the camera pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="IP AI v3 Camera Pipeline Test")
    parser.add_argument(
        "--camera", "-c",
        default="/dev/video0",
        help="Camera device path"
    )
    parser.add_argument(
        "--width", "-W",
        type=int, default=1280,
        help="Frame width"
    )
    parser.add_argument(
        "--height", "-H",
        type=int, default=720,
        help="Frame height"
    )
    parser.add_argument(
        "--fps", "-f",
        type=int, default=30,
        help="Target framerate"
    )
    parser.add_argument(
        "--pgie-config",
        default="configs/pgie_config.txt",
        help="Primary GIE config path"
    )
    parser.add_argument(
        "--tracker-config",
        default="configs/tracker_config.txt",
        help="Tracker config path (use 'none' to disable)"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable display output"
    )

    args = parser.parse_args()

    # Set up logging
    from ..utils.logger import setup_logger
    setup_logger(level=20)  # INFO

    # Frame callback for testing
    def on_frame(frame_data: FrameMetadata):
        if frame_data.frame_num % 30 == 0:
            persons = [d for d in frame_data.detections if d.class_name == 'person']
            print(f"Frame {frame_data.frame_num}: {len(persons)} persons, FPS: {frame_data.fps:.1f}")

    # Handle tracker config
    tracker_config = args.tracker_config
    if tracker_config.lower() == 'none':
        tracker_config = None

    # Create and run pipeline
    pipeline = CameraPipeline(
        camera_device=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        pgie_config=args.pgie_config,
        tracker_config=tracker_config,
        display=not args.no_display,
        on_frame_callback=on_frame
    )

    try:
        pipeline.start()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
