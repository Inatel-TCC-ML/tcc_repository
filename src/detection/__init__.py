from .base import Detector
from .b_free import BFreeDetector
from .cnn_detection import CNNDetectionDetector
from .npr_deepfake_detect import NPRDetector
from .spai_detector import SPAIDetector
from .universal_fake_detect import UnivFDDetector

__all__ = [
    "Detector",
    "CNNDetectionDetector",
    "UnivFDDetector",
    "NPRDetector",
    "SPAIDetector",
    "BFreeDetector"
]
