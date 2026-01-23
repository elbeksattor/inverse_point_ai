#!/usr/bin/env python3
"""
Demographics Estimation (Age/Gender) using InsightFace Model
Optimized for NVIDIA Jetson Orin Nano with TensorRT
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class Demographics:
    """Demographics estimation result"""
    age: int  # Estimated age
    gender: str  # 'male' or 'female'
    gender_confidence: float  # Confidence score for gender
    age_group: str  # Age group category

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


class DemographicsEstimator:
    """
    TensorRT-accelerated age/gender estimator

    Uses InsightFace's genderage model for fast and accurate estimation.
    """

    # Model input size (InsightFace genderage)
    INPUT_SIZE = (96, 96)

    def __init__(
        self,
        model_path: str = None,
        engine_path: str = None,
        use_tensorrt: bool = True
    ):
        """
        Initialize demographics estimator

        Args:
            model_path: Path to ONNX model
            engine_path: Path to TensorRT engine
            use_tensorrt: Use TensorRT acceleration
        """
        self.use_tensorrt = use_tensorrt

        # Default paths
        default_onnx = "/home/nvidia/projects/inverse_point/ip_ai_assist_old/edge/models/demographics/genderage.onnx"
        default_engine = "/home/nvidia/projects/inverse_point/ip_ai_assist_old/edge/models/demographics/genderage_fp16.engine"

        self.model_path = Path(model_path) if model_path else Path(default_onnx)
        self.engine_path = Path(engine_path) if engine_path else Path(default_engine)

        self.engine = None
        self.context = None
        self.session = None

        # Load model
        self._load_model()

        logger.info("demographics_estimator_initialized", use_tensorrt=use_tensorrt)

    def _load_model(self):
        """Load demographics model"""
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

            self._allocate_buffers()

            logger.info("tensorrt_engine_loaded", path=str(self.engine_path))

        except Exception as e:
            logger.warning("tensorrt_load_failed", error=str(e))
            self._load_onnx_model()

    def _load_onnx_model(self):
        """Load ONNX model"""
        try:
            import onnxruntime as ort

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
        """Allocate CUDA buffers"""
        import pycuda.driver as cuda
        import tensorrt as trt

        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

        for i in range(self.engine.num_bindings):
            binding = self.engine[i]
            size = trt.volume(self.engine.get_binding_shape(i))
            dtype = trt.nptype(self.engine.get_binding_dtype(i))

            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))

            if self.engine.binding_is_input(i):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})

    def preprocess(self, face_image: np.ndarray) -> np.ndarray:
        """
        Preprocess face image for demographics estimation

        Args:
            face_image: BGR face crop

        Returns:
            Preprocessed tensor
        """
        # Resize to model input size
        face = cv2.resize(face_image, self.INPUT_SIZE)

        # Convert BGR to RGB
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        # Normalize (InsightFace style)
        face = face.astype(np.float32)
        face = (face - 127.5) / 127.5

        # NCHW format
        face = face.transpose(2, 0, 1)
        face = np.expand_dims(face, 0)

        return face

    def estimate(self, face_image: np.ndarray) -> Optional[Demographics]:
        """
        Estimate age and gender from face image

        Args:
            face_image: BGR face crop

        Returns:
            Demographics object or None if estimation fails
        """
        if face_image is None or face_image.size == 0:
            return None

        # Minimum face size check
        if face_image.shape[0] < 20 or face_image.shape[1] < 20:
            return None

        try:
            # Preprocess
            blob = self.preprocess(face_image)

            # Run inference
            if self.use_tensorrt and self.context is not None:
                output = self._infer_tensorrt(blob)
            elif self.session is not None:
                output = self._infer_onnx(blob)
            else:
                return None

            # Parse output
            return self._parse_output(output)

        except Exception as e:
            logger.warning("demographics_estimation_error", error=str(e))
            return None

    def _infer_tensorrt(self, blob: np.ndarray) -> np.ndarray:
        """TensorRT inference"""
        import pycuda.driver as cuda

        np.copyto(self.inputs[0]['host'], blob.ravel())
        cuda.memcpy_htod_async(
            self.inputs[0]['device'],
            self.inputs[0]['host'],
            self.stream
        )

        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )

        cuda.memcpy_dtoh_async(
            self.outputs[0]['host'],
            self.outputs[0]['device'],
            self.stream
        )

        self.stream.synchronize()

        return self.outputs[0]['host'].copy()

    def _infer_onnx(self, blob: np.ndarray) -> np.ndarray:
        """ONNX Runtime inference"""
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: blob})
        return outputs[0]

    def _parse_output(self, output: np.ndarray) -> Demographics:
        """
        Parse model output to get age and gender

        InsightFace genderage model output format:
        - output[0:2]: gender logits (female, male)
        - output[2]: age / 100
        """
        output = output.flatten()

        # Gender (softmax over first 2 values)
        gender_logits = output[0:2]
        gender_probs = self._softmax(gender_logits)
        gender_idx = np.argmax(gender_probs)
        gender = 'female' if gender_idx == 0 else 'male'
        gender_confidence = float(gender_probs[gender_idx])

        # Age (multiply by 100 to get actual age)
        age = int(output[2] * 100)
        age = max(1, min(100, age))  # Clamp to valid range

        # Get age group
        age_group = Demographics.get_age_group(age)

        return Demographics(
            age=age,
            gender=gender,
            gender_confidence=gender_confidence,
            age_group=age_group
        )

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Compute softmax values"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def estimate_batch(self, face_images: List[np.ndarray]) -> List[Optional[Demographics]]:
        """
        Estimate demographics for multiple faces

        Args:
            face_images: List of BGR face crops

        Returns:
            List of Demographics objects
        """
        return [self.estimate(face) for face in face_images]


class HeadPoseEstimator:
    """
    Head pose estimation for attention analysis

    Uses 6DRepNet or similar model for yaw/pitch/roll estimation.
    For attention scoring:
    - Low yaw/pitch = looking at camera = high attention
    - High yaw/pitch = looking away = low attention
    """

    # Attention thresholds (degrees)
    ATTENTION_HIGH_THRESHOLD = 20  # Looking directly (within 20 degrees)
    ATTENTION_LOW_THRESHOLD = 45   # Looking away (beyond 45 degrees)

    def __init__(self, model_path: str = None):
        """
        Initialize head pose estimator

        Note: For MVP, we'll use face landmark-based estimation
        which doesn't require an additional model.
        """
        self.model = None
        logger.info("head_pose_estimator_initialized")

    def estimate_from_landmarks(
        self,
        landmarks: np.ndarray,
        image_size: Tuple[int, int]
    ) -> Tuple[float, float, float]:
        """
        Estimate head pose from facial landmarks

        Uses PnP (Perspective-n-Point) algorithm with standard face model.

        Args:
            landmarks: 5 facial landmarks (left_eye, right_eye, nose, left_mouth, right_mouth)
            image_size: (width, height) of image

        Returns:
            Tuple of (yaw, pitch, roll) in degrees
        """
        if landmarks is None or len(landmarks) < 5:
            return (0.0, 0.0, 0.0)

        # 3D face model points (standard face geometry)
        model_points = np.array([
            [-30.0, -30.0, -30.0],  # Left eye
            [30.0, -30.0, -30.0],   # Right eye
            [0.0, 0.0, 0.0],        # Nose tip
            [-20.0, 30.0, -20.0],   # Left mouth corner
            [20.0, 30.0, -20.0]     # Right mouth corner
        ], dtype=np.float64)

        # Camera matrix (approximate)
        w, h = image_size
        focal_length = w
        camera_matrix = np.array([
            [focal_length, 0, w / 2],
            [0, focal_length, h / 2],
            [0, 0, 1]
        ], dtype=np.float64)

        # Distortion coefficients (assume no distortion)
        dist_coeffs = np.zeros((4, 1))

        try:
            # Solve PnP
            success, rotation_vec, translation_vec = cv2.solvePnP(
                model_points,
                landmarks.astype(np.float64),
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success:
                return (0.0, 0.0, 0.0)

            # Convert rotation vector to rotation matrix
            rotation_mat, _ = cv2.Rodrigues(rotation_vec)

            # Extract Euler angles
            proj_matrix = np.hstack((rotation_mat, translation_vec))
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)

            pitch = float(euler_angles[0])
            yaw = float(euler_angles[1])
            roll = float(euler_angles[2])

            return (yaw, pitch, roll)

        except Exception as e:
            logger.warning("head_pose_estimation_error", error=str(e))
            return (0.0, 0.0, 0.0)

    def calculate_attention_score(
        self,
        yaw: float,
        pitch: float
    ) -> float:
        """
        Calculate attention score based on head pose

        Args:
            yaw: Left/right rotation in degrees
            pitch: Up/down rotation in degrees

        Returns:
            Attention score 0.0-1.0 (1.0 = fully attentive)
        """
        # Combine yaw and pitch into deviation from center
        deviation = np.sqrt(yaw**2 + pitch**2)

        if deviation <= self.ATTENTION_HIGH_THRESHOLD:
            # High attention zone
            score = 1.0
        elif deviation >= self.ATTENTION_LOW_THRESHOLD:
            # Low attention zone
            score = 0.0
        else:
            # Linear interpolation in between
            range_size = self.ATTENTION_LOW_THRESHOLD - self.ATTENTION_HIGH_THRESHOLD
            score = 1.0 - (deviation - self.ATTENTION_HIGH_THRESHOLD) / range_size

        return float(np.clip(score, 0.0, 1.0))


def test_demographics():
    """Test demographics estimator"""
    import time

    # Create estimator
    estimator = DemographicsEstimator(use_tensorrt=True)

    # Test with dummy face image
    test_face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)

    # Warmup
    for _ in range(5):
        estimator.estimate(test_face)

    # Benchmark
    times = []
    for _ in range(20):
        start = time.perf_counter()
        result = estimator.estimate(test_face)
        times.append(time.perf_counter() - start)

    avg_time = np.mean(times) * 1000
    print(f"Average inference time: {avg_time:.1f}ms")
    print(f"FPS: {1000/avg_time:.1f}")

    if result:
        print(f"Result: age={result.age}, gender={result.gender}, group={result.age_group}")


if __name__ == "__main__":
    test_demographics()
