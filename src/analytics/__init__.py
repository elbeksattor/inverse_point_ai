"""
IP AI v3 - Analytics Module

Provides attention tracking, head pose estimation, and analytics aggregation.
"""

from .attention_tracker import AttentionTracker, AttentionState, PersonAttention
from .headpose_onnx import HeadPoseONNX

__all__ = ['AttentionTracker', 'AttentionState', 'PersonAttention', 'HeadPoseONNX']
