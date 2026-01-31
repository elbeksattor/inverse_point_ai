"""
Head pose estimation using ONNX Runtime.

This module provides head pose estimation using the WHENet model via ONNX Runtime,
as a workaround for DeepStream SGIE caps negotiation issues.
"""
import numpy as np
from typing import Optional, Tuple
import structlog

logger = structlog.get_logger(__name__)


class HeadPoseONNX:
    """Head pose estimation using ONNX Runtime."""

    def __init__(self, model_path: str, use_gpu: bool = False):
        """
        Initialize head pose estimator.

        Args:
            model_path: Path to WHENet ONNX model
            use_gpu: Whether to use GPU (TensorRT provider if available)
        """
        import onnxruntime as ort

        # Select execution provider
        if use_gpu:
            providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        # Filter to available providers
        available = ort.get_available_providers()
        providers = [p for p in providers if p in available]

        logger.info("headpose_onnx_initializing",
                   model_path=model_path,
                   providers=providers)

        # Suppress ONNX Runtime verbose output
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3  # Error only

        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # WHENet expects 224x224 RGB input
        self.input_size = (224, 224)

        logger.info("headpose_onnx_initialized",
                   input_name=self.input_name,
                   output_name=self.output_name,
                   input_size=self.input_size)

    def preprocess(self, face_crop: np.ndarray) -> np.ndarray:
        """
        Preprocess face crop for WHENet inference.

        The prepost model has ImageNet normalization built in,
        so we just need to resize and convert to CHW format.

        Args:
            face_crop: BGR face image from DeepStream (HWC format)

        Returns:
            Preprocessed tensor ready for inference (NCHW format)
        """
        import cv2

        # Resize to 224x224
        resized = cv2.resize(face_crop, self.input_size, interpolation=cv2.INTER_LINEAR)

        # BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # HWC to CHW, add batch dimension
        chw = np.transpose(rgb, (2, 0, 1))  # (3, 224, 224)
        batch = np.expand_dims(chw, axis=0)  # (1, 3, 224, 224)

        # Convert to float32 (prepost model expects 0-255 uint8-like values)
        tensor = batch.astype(np.float32)

        return tensor

    def estimate(self, face_crop: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """
        Estimate head pose from face crop.

        Args:
            face_crop: BGR face image from DeepStream

        Returns:
            Tuple of (yaw, pitch, roll) in degrees, or None if inference fails
        """
        try:
            # Check valid input
            if face_crop is None or face_crop.size == 0:
                return None

            if face_crop.shape[0] < 16 or face_crop.shape[1] < 16:
                return None

            # Preprocess
            tensor = self.preprocess(face_crop)

            # Run inference
            outputs = self.session.run([self.output_name], {self.input_name: tensor})

            # WHENet outputs [yaw, roll, pitch]
            yaw_roll_pitch = outputs[0][0]  # Shape: (3,)

            yaw = float(yaw_roll_pitch[0])
            roll = float(yaw_roll_pitch[1])
            pitch = float(yaw_roll_pitch[2])

            return (yaw, pitch, roll)

        except Exception as e:
            logger.warning("headpose_inference_failed", error=str(e))
            return None
