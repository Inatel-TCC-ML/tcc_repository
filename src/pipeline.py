"""Orquestra extração de keyframes, detecção e agregação de um vídeo."""

from __future__ import annotations

from statistics import fmean

from src.aggregation.base import Aggregator
from src.detection.base import Detector
from src.extraction.base import KeyframeExtractor


class VideoDetectionPipeline:
    def __init__(
        self,
        extractor: KeyframeExtractor,
        detectors: list[Detector],
        aggregator: Aggregator,
    ) -> None:
        if not detectors:
            raise ValueError("O pipeline requer ao menos um detector.")
        self.extractor = extractor
        self.detectors = detectors
        self.aggregator = aggregator

    def run(self, video_path: str) -> dict:
        keyframes = self.extractor.extract(video_path)
        scores_by_frame: dict[str, dict[str, float]] = {}
        scores_by_detector: dict[str, list[float]] = {
            detector.name: [] for detector in self.detectors
        }

        for keyframe in keyframes:
            frame_scores = {
                detector.name: detector.predict(keyframe) for detector in self.detectors
            }
            scores_by_frame[keyframe] = frame_scores
            for detector_name, score in frame_scores.items():
                scores_by_detector[detector_name].append(score)

        detector_scores = {
            detector_name: fmean(scores)
            for detector_name, scores in scores_by_detector.items()
        }
        return {
            "keyframes": keyframes,
            "scores_by_frame": scores_by_frame,
            "scores_by_detector": detector_scores,
            "final_score": self.aggregator.aggregate(detector_scores),
        }
