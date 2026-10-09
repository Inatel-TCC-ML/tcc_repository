from .base import Detector
from .cnndetection import CNNDetectionDetector
from .universalFakeDetect import UnivFDDetector
from .nprDeepfakeDetect import NPRDetector
from .bFree import BFreeDetector

__all__ = ["Detector", "CNNDetectionDetector", "UnivFDDetector", "NPRDetector", "BFreeDetector"]
