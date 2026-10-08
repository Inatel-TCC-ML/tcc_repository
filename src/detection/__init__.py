from .base import Detector
from .cnnDetection import CNNDetectionDetector
from .universalFakeDetect import UnivFDDetector
from .nprDeepfakeDetect import NPRDetector

__all__ = ["Detector", "CNNDetectionDetector", "UnivFDDetector", "NPRDetector"]
