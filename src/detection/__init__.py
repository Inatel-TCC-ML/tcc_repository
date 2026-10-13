from .base import Detector
from .cnndetection import CNNDetectionDetector
from .universalFakeDetect import UnivFDDetector
from .nprDeepfakeDetect import NPRDetector
from .spaiDetector import SPAIDetector

__all__ = [
    "Detector",
    "CNNDetectionDetector",
    "UnivFDDetector",
    "NPRDetector",
    "SPAIDetector",
]
