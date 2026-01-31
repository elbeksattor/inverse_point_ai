"""
Configuration management for IP AI v3.

Loads and validates YAML configuration files.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class CameraConfig:
    """Camera configuration."""
    device: str = "/dev/video0"
    width: int = 1280
    height: int = 720
    fps: int = 30


@dataclass
class DetectionConfig:
    """Person detection configuration."""
    model_path: str = "models/peoplenet/resnet34_peoplenet.engine"
    config_file: str = "configs/pgie_config.txt"
    confidence_threshold: float = 0.4
    batch_size: int = 1


@dataclass
class TrackerConfig:
    """Multi-object tracker configuration."""
    config_file: str = "configs/tracker_config.txt"
    width: int = 640
    height: int = 384
    enable_past_frame: bool = True  # Required for OSNet embeddings


@dataclass
class ReIDConfig:
    """Re-identification configuration."""
    enabled: bool = True
    faiss_index_path: str = "output/person_embeddings.index"
    sqlite_db_path: str = "output/person_database.db"
    similarity_threshold: float = 0.75
    embedding_dim: int = 128
    moving_avg_weight: float = 0.7  # Weight for old embedding
    retention_days: int = 30


@dataclass
class DemographicsConfig:
    """Demographics analysis configuration."""
    enabled: bool = True
    face_detection_model: str = "models/face/retinaface.engine"
    demographics_model: str = "models/demographics/mivolo.engine"
    confidence_threshold: float = 0.6


@dataclass
class AttentionConfig:
    """Attention detection configuration."""
    enabled: bool = True
    headpose_model: str = "models/headpose/6drepnet.engine"
    yaw_threshold: float = 30.0  # degrees
    pitch_threshold: float = 20.0  # degrees
    engagement_time: float = 2.0  # seconds for qualified impression


@dataclass
class OutputConfig:
    """Output configuration."""
    osd_enabled: bool = True
    json_output_path: str = "output/analytics/analytics.json"
    json_interval_seconds: int = 60
    log_path: str = "output/logs"
    video_output_path: Optional[str] = None


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    camera: CameraConfig = field(default_factory=CameraConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    reid: ReIDConfig = field(default_factory=ReIDConfig)
    demographics: DemographicsConfig = field(default_factory=DemographicsConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def load_config(config_path: str) -> PipelineConfig:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        PipelineConfig object

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    config = PipelineConfig()

    # Parse camera config
    if "camera" in data:
        config.camera = CameraConfig(**data["camera"])

    # Parse detection config
    if "detection" in data:
        config.detection = DetectionConfig(**data["detection"])

    # Parse tracker config
    if "tracker" in data:
        config.tracker = TrackerConfig(**data["tracker"])

    # Parse reid config
    if "reid" in data:
        config.reid = ReIDConfig(**data["reid"])

    # Parse demographics config
    if "demographics" in data:
        config.demographics = DemographicsConfig(**data["demographics"])

    # Parse attention config
    if "attention" in data:
        config.attention = AttentionConfig(**data["attention"])

    # Parse output config
    if "output" in data:
        config.output = OutputConfig(**data["output"])

    return config


def save_config(config: PipelineConfig, config_path: str) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: PipelineConfig object
        config_path: Output path
    """
    import dataclasses

    def to_dict(obj):
        if dataclasses.is_dataclass(obj):
            return {k: to_dict(v) for k, v in dataclasses.asdict(obj).items()}
        return obj

    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w") as f:
        yaml.dump(to_dict(config), f, default_flow_style=False, sort_keys=False)
