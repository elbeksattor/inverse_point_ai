#!/usr/bin/env python3
"""
Face Detection using SCRFD (InsightFace) with TensorRT
Optimized for NVIDIA Jetson Orin Nano
"""

import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class FaceDetection:
    """Single face detection result"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    landmarks: Optional[np.ndarray] = None  # 5 facial landmarks (if available)

    @property
    def center(self) -> Tuple[int, int]:
        """Get face center point"""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> int:
        return self.width * self.height


class FaceDetector:
    """
    TensorRT-accelerated face detector using SCRFD model

    SCRFD (Sample and Computation Redistribution for Face Detection)
    is a fast and accurate face detector from InsightFace.
    """

    # SCRFD-500M input size
    INPUT_SIZE = (640, 640)

    # Detection thresholds
    CONF_THRESHOLD = 0.5
    NMS_THRESHOLD = 0.4

    # Anchor strides for SCRFD
    FEAT_STRIDE_FPN = [8, 16, 32]
    NUM_ANCHORS = 2

    def __init__(
        self,
        model_path: str = None,
        engine_path: str = None,
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        use_tensorrt: bool = True
    ):
        """
        Initialize face detector

        Args:
            model_path: Path to ONNX model file
            engine_path: Path to TensorRT engine file (preferred)
            conf_threshold: Detection confidence threshold
            nms_threshold: NMS IoU threshold
            use_tensorrt: Use TensorRT acceleration
        """
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.use_tensorrt = use_tensorrt

        # Default model paths
        default_onnx = "/home/nvidia/projects/inverse_point/ip_ai_assist_old/edge/models/demographics/det_500m.onnx"
        default_engine = "/home/nvidia/projects/inverse_point/ip_ai_assist_old/edge/models/demographics/det_500m_fp16.engine"

        self.model_path = Path(model_path) if model_path else Path(default_onnx)
        self.engine_path = Path(engine_path) if engine_path else Path(default_engine)

        self.engine = None
        self.context = None
        self.session = None

        # Initialize model
        self._load_model()

        logger.info(
            "face_detector_initialized",
            use_tensorrt=use_tensorrt,
            conf_threshold=conf_threshold
        )

    def _load_model(self):
        """Load face detection model"""
        if self.use_tensorrt and self.engine_path.exists():
            self._load_tensorrt_engine()
        elif self.model_path.exists():
            self._load_onnx_model()
        else:
            raise FileNotFoundError(
                f"No model found at {self.model_path} or {self.engine_path}"
            )

    def _load_tensorrt_engine(self):
        """Load TensorRT engine"""
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit

            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

            with open(self.engine_path, 'rb') as f:
                engine_data = f.read()

            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(engine_data)
            self.context = self.engine.create_execution_context()

            # Allocate buffers
            self._allocate_buffers()

            logger.info("tensorrt_engine_loaded", path=str(self.engine_path))

        except Exception as e:
            logger.warning("tensorrt_load_failed", error=str(e))
            self._load_onnx_model()

    def _load_onnx_model(self):
        """Load ONNX model with OpenCV DNN or ONNX Runtime"""
        try:
            import onnxruntime as ort

            # Use CUDA if available
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(
                str(self.model_path),
                providers=providers
            )
            self.use_tensorrt = False

            logger.info("onnx_model_loaded", path=str(self.model_path))

        except Exception as e:
            logger.error("onnx_load_failed", error=str(e))
            raise

    def _allocate_buffers(self):
        """Allocate CUDA buffers for TensorRT inference"""
        import pycuda.driver as cuda

        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))

            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Preprocess image for face detection

        Args:
            image: BGR image (H, W, C)

        Returns:
            Preprocessed tensor, scale factor, padding offset
        """
        h, w = image.shape[:2]
        target_h, target_w = self.INPUT_SIZE

        # Calculate scale to fit image in input size
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize image
        resized = cv2.resize(image, (new_w, new_h))

        # Pad to target size
        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2

        padded = np.full((target_h, target_w, 3), 127, dtype=np.uint8)
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        # Convert to float and normalize
        blob = padded.astype(np.float32)
        blob = (blob - 127.5) / 128.0

        # NCHW format
        blob = blob.transpose(2, 0, 1)
        blob = np.expand_dims(blob, 0)

        return blob, scale, (pad_w, pad_h)

    def detect(self, image: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces in image

        Args:
            image: BGR image (H, W, C)

        Returns:
            List of FaceDetection objects
        """
        if image is None or image.size == 0:
            return []

        # Preprocess
        blob, scale, (pad_w, pad_h) = self.preprocess(image)

        # Run inference
        if self.use_tensorrt and self.context is not None:
            outputs = self._infer_tensorrt(blob)
        elif self.session is not None:
            outputs = self._infer_onnx(blob)
        else:
            return []

        # Post-process
        detections = self._postprocess(
            outputs,
            scale,
            (pad_w, pad_h),
            image.shape[:2]
        )

        return detections

    def _infer_tensorrt(self, blob: np.ndarray) -> List[np.ndarray]:
        """Run TensorRT inference"""
        import pycuda.driver as cuda

        # Copy input to device
        np.copyto(self.inputs[0]['host'], blob.ravel())
        cuda.memcpy_htod_async(
            self.inputs[0]['device'],
            self.inputs[0]['host'],
            self.stream
        )

        # Execute
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )

        # Copy outputs to host
        outputs = []
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
            outputs.append(out['host'].copy())

        self.stream.synchronize()

        return outputs

    def _infer_onnx(self, blob: np.ndarray) -> List[np.ndarray]:
        """Run ONNX Runtime inference"""
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: blob})
        return outputs

    def _postprocess(
        self,
        outputs: List[np.ndarray],
        scale: float,
        padding: Tuple[int, int],
        orig_size: Tuple[int, int]
    ) -> List[FaceDetection]:
        """
        Post-process SCRFD outputs

        SCRFD outputs boxes and scores at 3 scales (8, 16, 32 stride)
        """
        detections = []
        pad_w, pad_h = padding
        orig_h, orig_w = orig_size

        # Parse outputs based on SCRFD structure
        # For SCRFD-500M: 9 outputs (3 scales x 3 outputs per scale)
        # Each scale has: score, bbox, kps (landmarks)

        try:
            all_boxes = []
            all_scores = []

            for idx, stride in enumerate(self.FEAT_STRIDE_FPN):
                # Get outputs for this scale
                score_idx = idx * 3
                bbox_idx = idx * 3 + 1

                if score_idx >= len(outputs) or bbox_idx >= len(outputs):
                    continue

                scores = outputs[score_idx]
                bboxes = outputs[bbox_idx]

                # Reshape if needed
                if len(scores.shape) == 4:
                    scores = scores.squeeze((0, 1))
                if len(bboxes.shape) == 4:
                    bboxes = bboxes.squeeze(0)

                # Generate anchor grid
                h, w = self.INPUT_SIZE[1] // stride, self.INPUT_SIZE[0] // stride

                for i in range(h):
                    for j in range(w):
                        for k in range(self.NUM_ANCHORS):
                            score_val = float(scores[i, j] if scores.ndim == 2 else scores[i * w + j])

                            if score_val > self.conf_threshold:
                                # Decode bbox
                                cx = (j + 0.5) * stride
                                cy = (i + 0.5) * stride

                                if bboxes.ndim >= 3:
                                    bbox = bboxes[0, i * w + j, :4]
                                else:
                                    bbox = bboxes[i * w + j, :4]

                                # Distance format (l, t, r, b) to xyxy
                                x1 = cx - bbox[0] * stride
                                y1 = cy - bbox[1] * stride
                                x2 = cx + bbox[2] * stride
                                y2 = cy + bbox[3] * stride

                                all_boxes.append([x1, y1, x2, y2])
                                all_scores.append(score_val)

            if not all_boxes:
                return []

            # Apply NMS
            boxes = np.array(all_boxes)
            scores = np.array(all_scores)

            indices = cv2.dnn.NMSBoxes(
                boxes.tolist(),
                scores.tolist(),
                self.conf_threshold,
                self.nms_threshold
            )

            if len(indices) == 0:
                return []

            # Convert back to original image coordinates
            for i in indices.flatten():
                x1, y1, x2, y2 = boxes[i]

                # Remove padding and scale
                x1 = (x1 - pad_w) / scale
                y1 = (y1 - pad_h) / scale
                x2 = (x2 - pad_w) / scale
                y2 = (y2 - pad_h) / scale

                # Clip to image bounds
                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(orig_w, int(x2))
                y2 = min(orig_h, int(y2))

                if x2 > x1 and y2 > y1:
                    detections.append(FaceDetection(
                        bbox=(x1, y1, x2, y2),
                        confidence=float(scores[i])
                    ))

        except Exception as e:
            logger.warning("face_postprocess_error", error=str(e))

        return detections

    def detect_in_person_crop(
        self,
        image: np.ndarray,
        person_bbox: Tuple[int, int, int, int],
        expand_ratio: float = 0.1
    ) -> List[FaceDetection]:
        """
        Detect faces within a person bounding box

        Args:
            image: Full frame BGR image
            person_bbox: Person bounding box (x1, y1, x2, y2)
            expand_ratio: Expand person bbox by this ratio

        Returns:
            List of face detections with coordinates in full image space
        """
        x1, y1, x2, y2 = person_bbox
        h, w = image.shape[:2]

        # Expand bbox slightly
        bw, bh = x2 - x1, y2 - y1
        x1 = max(0, int(x1 - bw * expand_ratio))
        y1 = max(0, int(y1 - bh * expand_ratio))
        x2 = min(w, int(x2 + bw * expand_ratio))
        y2 = min(h, int(y2 + bh * expand_ratio))

        # Crop person region
        person_crop = image[y1:y2, x1:x2]

        if person_crop.size == 0:
            return []

        # Detect faces in crop
        faces = self.detect(person_crop)

        # Convert coordinates back to full image space
        for face in faces:
            fx1, fy1, fx2, fy2 = face.bbox
            face.bbox = (fx1 + x1, fy1 + y1, fx2 + x1, fy2 + y1)

        return faces


def test_face_detector():
    """Test face detector with sample image"""
    import time

    # Create detector
    detector = FaceDetector(
        conf_threshold=0.5,
        use_tensorrt=True
    )

    # Test with dummy image
    test_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    # Warmup
    for _ in range(5):
        detector.detect(test_image)

    # Benchmark
    times = []
    for _ in range(20):
        start = time.perf_counter()
        faces = detector.detect(test_image)
        times.append(time.perf_counter() - start)

    avg_time = np.mean(times) * 1000
    print(f"Average inference time: {avg_time:.1f}ms")
    print(f"FPS: {1000/avg_time:.1f}")


if __name__ == "__main__":
    test_face_detector()
