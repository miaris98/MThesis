"""Vision feature extractor backbones for CARLA multi-camera inputs."""
from src.models.backbones.cnn_backbone import CNNFeatureExtractor
from src.models.backbones.resnet_backbone import PretrainedVisionFeatureExtractor
from src.models.backbones.erfnet_backbone import ERFNetFeatureExtractor

__all__ = ["CNNFeatureExtractor", "PretrainedVisionFeatureExtractor", "ERFNetFeatureExtractor"]
