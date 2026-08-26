"""Orquestra extração de keyframes, detecção e agregação de um vídeo."""

from __future__ import annotations

from src.aggregation.base import Aggregator
from src.aggregation.frame_aggregators import FrameAggregator, MeanFrameAggregator
from src.detection.base import Detector
from src.extraction.base import KeyframeExtractor


class VideoDetectionPipeline:
    def __init__(
        self,
        extractor: KeyframeExtractor,
        detectors: list[Detector],
        aggregator: Aggregator,
        frame_aggregator: FrameAggregator | None = None,
    ) -> None:
        if not detectors:
            raise ValueError("O pipeline requer ao menos um detector.")
        self.extractor = extractor
        self.detectors = detectors
        self.aggregator = aggregator
        self.frame_aggregator = frame_aggregator or MeanFrameAggregator()

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
            detector_name: self.frame_aggregator.aggregate(scores)
            for detector_name, scores in scores_by_detector.items()
        }
        return {
            "keyframes": keyframes,
            "scores_by_frame": scores_by_frame,
            "scores_by_detector": detector_scores,
            "frame_aggregation": self.frame_aggregator.name,
            "final_score": self.aggregator.aggregate(detector_scores),
        }
